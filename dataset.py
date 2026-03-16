from typing import List, Dict
import torch
from torch.utils.data import Dataset


class AlignedSpeechDataset(Dataset):
    def __init__(self, items: List[Dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]



def collate_fn(batch: List[Dict]):
    B = len(batch)
    audio_lens = torch.tensor([x["audio"].numel() for x in batch], dtype=torch.long)
    text_lens = torch.tensor([x["text_ids"].numel() for x in batch], dtype=torch.long)

    max_audio = int(audio_lens.max().item())
    max_text = int(text_lens.max().item())

    audio = torch.zeros(B, max_audio)
    text_ids = torch.zeros(B, max_text, dtype=torch.long)
    positions = torch.zeros(B, max_text, dtype=torch.long)

    for i, item in enumerate(batch):
        a = item["audio"]
        t = item["text_ids"]
        p = item["positions"]

        audio[i, :a.numel()] = a
        text_ids[i, :t.numel()] = t
        positions[i, :p.numel()] = p

    return {
        "audio": audio,
        "audio_lens": audio_lens,
        "text_ids": text_ids,
        "text_lens": text_lens,
        "positions": positions,
    }
    