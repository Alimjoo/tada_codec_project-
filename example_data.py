import math
import random
import torch

sample_rate = 16000
hop_length = 320
vocab_size = 32000

items = []
for _ in range(64):
    seconds = random.randint(3, 8)
    audio_len = sample_rate * seconds
    num_tokens = random.randint(8, 30)

    audio = torch.randn(audio_len) * 0.02
    text_ids = torch.randint(1, vocab_size, (num_tokens,), dtype=torch.long)

    n_frames = math.ceil(audio_len / hop_length)
    positions = torch.linspace(1, max(2, n_frames - 2), steps=num_tokens).long()

    items.append({
        "audio": audio,
        "text_ids": text_ids,
        "positions": positions,
    })

torch.save(items, "toy_data.pt")
print("saved toy_data.pt with", len(items), "samples")
