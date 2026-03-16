import argparse
import json
import math
from pathlib import Path

import torch
import torchaudio
from datasets import Audio, load_dataset
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


def asr_with_letter_timestamps(waveform: torch.Tensor, processor, model):
    if waveform.numel() == 0:
        return {
            "transcript": "",
            "letters": [],
            "num_frames": 0,
            "sec_per_frame": 0.0,
            "audio_duration_sec": 0.0,
        }

    processed = processor(
        waveform,
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

    with torch.no_grad():
        logits = model(**model_inputs).logits

    pred_ids = torch.argmax(logits, dim=-1)[0].cpu()
    num_frames = pred_ids.shape[0]
    audio_duration_sec = waveform.shape[0] / TARGET_SAMPLE_RATE
    sec_per_frame = audio_duration_sec / max(num_frames, 1)

    blank_id = processor.tokenizer.pad_token_id
    if blank_id is None:
        blank_id = -1
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

    finalize_run(prev_id, run_start, num_frames - 1)

    transcription = processor.decode(pred_ids).strip()
    return {
        "transcript": transcription,
        "letters": letters,
        "num_frames": num_frames,
        "sec_per_frame": sec_per_frame,
        "audio_duration_sec": audio_duration_sec,
    }


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


def build_items(dataset, processor, model, limit: int):
    vocab = {"<pad>": 0}
    items = []

    for index, sample in enumerate(dataset.select(range(min(limit, len(dataset))))):
        audio = sample["audio"]
        audio_array, sample_rate, audio_path = extract_audio_data(audio)
        waveform = to_mono_16k(audio_array, sample_rate)
        asr = asr_with_letter_timestamps(waveform, processor, model)
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
                "audio": waveform,
                "text_ids": torch.tensor(word_ids, dtype=torch.long),
                "positions": torch.tensor(clipped_positions, dtype=torch.long),
                "reference_text": sample.get("sentence", ""),
                "predicted_transcript": asr["transcript"],
                "words": words,
                "audio_path": audio_path,
                "sample_index": index,
            }
        )

    return items, vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="arhip_program_ug_100.pt")
    parser.add_argument("--vocab-output", type=str, default="arhip_program_ug_100_vocab.json")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dataset", type=str, default=DATASET_ID)
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"using device: {device}")

    dataset = load_dataset(args.dataset, split=args.split, token=args.hf_token)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))

    processor = Wav2Vec2BertProcessor.from_pretrained(args.model, token=args.hf_token)
    model = AutoModelForCTC.from_pretrained(args.model, token=args.hf_token).to(device)
    model.eval()

    items, vocab = build_items(dataset, processor, model, args.limit)

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
