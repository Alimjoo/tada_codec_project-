import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torchaudio
from datasets import Audio, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCTC, Wav2Vec2BertProcessor


DATASET_ID = "piyazon/Arhip-Program-Ug"
MODEL_ID = "piyazon/ASR-cv-corpus-ug-22-2"
TARGET_SAMPLE_RATE = 16000
HOP_LENGTH = 320


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_mono_16k(audio_array, sample_rate: int) -> torch.Tensor:
    waveform = torch.as_tensor(audio_array, dtype=torch.float32)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2 and waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.transpose(0, 1)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE,
        )
        waveform = resampler(waveform)

    return waveform.squeeze(0).contiguous()


def empty_device_cache(device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device == "mps" and getattr(torch, "mps", None) is not None:
        torch.mps.empty_cache()


def get_audio_field(audio, key: str, default=None):
    if isinstance(audio, dict):
        return audio.get(key, default)
    return getattr(audio, key, default)


def extract_audio_data(audio):
    audio_array = get_audio_field(audio, "array")
    sample_rate = get_audio_field(audio, "sampling_rate")
    audio_path = get_audio_field(audio, "path")

    if audio_array is not None and sample_rate is not None:
        return audio_array, sample_rate, audio_path

    if hasattr(audio, "get_all_samples"):
        decoded = audio.get_all_samples()
        data = getattr(decoded, "data", None)
        if data is None:
            data = get_audio_field(decoded, "array")
        rate = getattr(decoded, "sample_rate", None)
        if rate is None:
            rate = get_audio_field(decoded, "sampling_rate")
        return data, rate, audio_path

    raise ValueError(f"Unsupported audio representation: {type(audio)!r}")


def decode_ctc_prediction(pred_ids, processor, blank_id, sec_per_frame):
    word_delim = getattr(processor.tokenizer, "word_delimiter_token", "|")
    letters = []
    prev_id = None
    run_start = 0

    def finalize_run(token_id, start_frame, end_frame):
        if token_id is None or token_id == blank_id:
            return

        token = processor.tokenizer.convert_ids_to_tokens(int(token_id))
        if token is None:
            return

        char = " " if token == word_delim else token.replace("▁", " ")
        start_time = start_frame * sec_per_frame
        end_time = (end_frame + 1) * sec_per_frame

        letters.append(
            {
                "char": char,
                "start": start_time,
                "end": end_time,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
            }
        )

    for frame_index, token_id in enumerate(pred_ids.tolist()):
        if prev_id is None:
            prev_id = token_id
            run_start = frame_index
        elif token_id != prev_id:
            finalize_run(prev_id, run_start, frame_index - 1)
            prev_id = token_id
            run_start = frame_index

    finalize_run(prev_id, run_start, len(pred_ids) - 1)
    transcription = processor.decode(pred_ids).strip()
    return transcription, letters


def infer_output_lengths(model, model_inputs, logits):
    if "attention_mask" in model_inputs:
        input_lengths = model_inputs["attention_mask"].sum(dim=-1)
    else:
        first_tensor = next(iter(model_inputs.values()))
        input_lengths = torch.full(
            (first_tensor.size(0),),
            first_tensor.size(1),
            device=first_tensor.device,
            dtype=torch.long,
        )

    if hasattr(model, "_get_feat_extract_output_lengths"):
        output_lengths = model._get_feat_extract_output_lengths(input_lengths)
        return output_lengths.to(dtype=torch.long).cpu()

    return torch.full(
        (logits.size(0),),
        logits.size(1),
        dtype=torch.long,
    )


def asr_batch_with_letter_timestamps(waveforms, processor, model):
    results = []
    non_empty = [waveform for waveform in waveforms if waveform.numel() > 0]
    if not non_empty:
        return [
            {
                "transcript": "",
                "letters": [],
                "num_frames": 0,
                "sec_per_frame": 0.0,
                "audio_duration_sec": 0.0,
            }
            for _ in waveforms
        ]

    processed = processor(
        waveforms,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    model_inputs = {}
    input_features = processed.get("input_features")
    if input_features is not None:
        model_inputs["input_features"] = input_features.to(model.device)
    else:
        model_inputs["input_values"] = processed["input_values"].to(model.device)
    if "attention_mask" in processed:
        model_inputs["attention_mask"] = processed["attention_mask"].to(model.device)

    with torch.inference_mode():
        logits = model(**model_inputs).logits

    blank_id = processor.tokenizer.pad_token_id
    if blank_id is None:
        blank_id = -1
    pred_ids_batch = torch.argmax(logits, dim=-1).cpu()
    output_lengths = infer_output_lengths(model, model_inputs, logits)

    for waveform, pred_ids, output_len in zip(waveforms, pred_ids_batch, output_lengths):
        if waveform.numel() == 0:
            results.append(
                {
                    "transcript": "",
                    "letters": [],
                    "num_frames": 0,
                    "sec_per_frame": 0.0,
                    "audio_duration_sec": 0.0,
                }
            )
            continue

        num_frames = max(int(output_len.item()), 1)
        trimmed_pred_ids = pred_ids[:num_frames]
        audio_duration_sec = waveform.shape[0] / TARGET_SAMPLE_RATE
        sec_per_frame = audio_duration_sec / num_frames
        transcription, letters = decode_ctc_prediction(
            trimmed_pred_ids,
            processor,
            blank_id,
            sec_per_frame,
        )
        results.append(
            {
                "transcript": transcription,
                "letters": letters,
                "num_frames": num_frames,
                "sec_per_frame": sec_per_frame,
                "audio_duration_sec": audio_duration_sec,
            }
        )

    return results


def letters_to_words(letters):
    words = []
    current_chars = []
    current_start = None
    current_end = None

    for letter in letters:
        char = letter["char"]
        if char.isspace():
            if current_chars:
                words.append(
                    {
                        "word": "".join(current_chars).strip(),
                        "start": current_start,
                        "end": current_end,
                    }
                )
                current_chars = []
                current_start = None
                current_end = None
            continue

        if current_start is None:
            current_start = letter["start"]
        current_end = letter["end"]
        current_chars.append(char)

    if current_chars:
        words.append(
            {
                "word": "".join(current_chars).strip(),
                "start": current_start,
                "end": current_end,
            }
        )

    return [word for word in words if word["word"]]


def word_to_frame_position(word) -> int:
    midpoint_sec = 0.5 * (word["start"] + word["end"])
    return max(0, int(round(midpoint_sec * TARGET_SAMPLE_RATE / HOP_LENGTH)))


def build_items(
    dataset,
    processor,
    model,
    limit: int,
    batch_size: int,
    audio_dtype: str,
    keep_metadata: bool,
):
    vocab = {"<pad>": 0}
    items = []
    store_dtype = torch.float16 if audio_dtype == "float16" else torch.float32
    selected = dataset.select(range(min(limit, len(dataset))))

    for start in tqdm(range(0, len(selected), batch_size), desc="generate_data", unit="batch"):
        batch_samples = [selected[idx] for idx in range(start, min(start + batch_size, len(selected)))]
        prepared = []
        waveforms = []
        for local_idx, sample in enumerate(batch_samples):
            audio = sample["audio"]
            audio_array, sample_rate, audio_path = extract_audio_data(audio)
            waveform = to_mono_16k(audio_array, sample_rate)
            prepared.append(
                {
                    "sample": sample,
                    "audio_path": audio_path,
                    "sample_index": start + local_idx,
                    "waveform": waveform,
                }
            )
            waveforms.append(waveform)

        asr_results = asr_batch_with_letter_timestamps(waveforms, processor, model)

        for entry, asr in zip(prepared, asr_results):
            waveform = entry["waveform"]
            words = letters_to_words(asr["letters"])
            if not words:
                continue

            word_ids = []
            positions = []
            for word in words:
                token = word["word"]
                if token not in vocab:
                    vocab[token] = len(vocab)
                word_ids.append(vocab[token])
                positions.append(word_to_frame_position(word))

            n_frames = math.ceil(waveform.numel() / HOP_LENGTH)
            clipped_positions = [min(pos, max(n_frames - 1, 0)) for pos in positions]

            items.append(
                {
                    "audio": waveform.to(dtype=store_dtype).cpu(),
                    "text_ids": torch.tensor(word_ids, dtype=torch.long),
                    "positions": torch.tensor(clipped_positions, dtype=torch.long),
                    "sample_index": entry["sample_index"],
                }
            )
            if keep_metadata:
                items[-1]["reference_text"] = entry["sample"].get("sentence", "")
                items[-1]["predicted_transcript"] = asr["transcript"]
                items[-1]["words"] = words
                items[-1]["audio_path"] = entry["audio_path"]

        del batch_samples, prepared, waveforms, asr_results
        gc.collect()
        empty_device_cache(str(model.device))

    return items, vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="arhip_program_ug_1000.pt")
    parser.add_argument("--vocab-output", type=str, default="arhip_program_ug_1000_vocab.json")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--dataset", type=str, default=DATASET_ID)
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--audio-dtype", type=str, choices=("float16", "float32"), default="float16")
    parser.add_argument("--keep-metadata", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"using device: {device}")

    dataset = load_dataset(args.dataset, split=args.split, token=args.hf_token)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))

    processor = Wav2Vec2BertProcessor.from_pretrained(args.model, token=args.hf_token)
    model = AutoModelForCTC.from_pretrained(args.model, token=args.hf_token).to(device)
    model.eval()

    items, vocab = build_items(
        dataset,
        processor,
        model,
        args.limit,
        args.batch_size,
        args.audio_dtype,
        args.keep_metadata,
    )

    output_path = Path(args.output)
    vocab_path = Path(args.vocab_output)

    torch.save(items, output_path)
    vocab_payload = {
        "dataset": args.dataset,
        "split": args.split,
        "limit": args.limit,
        "sample_rate": TARGET_SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "model": args.model,
        "vocab": vocab,
    }
    vocab_path.write_text(json.dumps(vocab_payload, ensure_ascii=False, indent=2))

    print(f"saved {len(items)} items to {output_path}")
    print(f"saved vocab with {len(vocab)} entries to {vocab_path}")


if __name__ == "__main__":
    main()
