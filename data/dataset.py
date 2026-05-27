import os
from typing import Optional

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers


def train_tokenizer(
    dataset_name: str,
    vocab_size: int = 16384,
    sample_size: int = 10000,
    save_path: str = "tokenizer.json",
):
    dataset = load_dataset(
        dataset_name, split="train", streaming=True
    )
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<|im_start|>", "<|im_end|>"],
    )

    def text_iterator():
        count = 0
        for example in dataset:
            yield example["text"]
            count += 1
            if count >= sample_size:
                break

    tokenizer.train_from_iterator(text_iterator(), trainer)
    tokenizer.save(save_path)
    return tokenizer


def load_or_train_tokenizer(
    tokenizer_path: str,
    dataset_name: str = "HuggingFaceFW/fineweb",
    vocab_size: int = 16384,
    sample_size: int = 10000,
) -> Tokenizer:
    if os.path.exists(tokenizer_path):
        return Tokenizer.from_file(tokenizer_path)
    print(f"Training tokenizer on {sample_size} samples from {dataset_name}...")
    return train_tokenizer(
        dataset_name=dataset_name,
        vocab_size=vocab_size,
        sample_size=sample_size,
        save_path=tokenizer_path,
    )


class FineWebDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        split: str = "train",
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
    ):
        super().__init__()
        if split == "validation":
            split = "train"
        self.dataset = load_dataset(
            dataset_name, split=split, streaming=True
        )
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            tokens = self.tokenizer.encode(example["text"]).ids
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                }


def create_dataloader(
    tokenizer_path: str,
    seq_len: int = 2048,
    batch_size: int = 8,
    split: str = "train",
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    vocab_size: int = 16384,
    sample_size: int = 10000,
):
    tokenizer = load_or_train_tokenizer(
        tokenizer_path=tokenizer_path,
        dataset_name=dataset_name,
        vocab_size=vocab_size,
        sample_size=sample_size,
    )
    dataset = FineWebDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        split=split,
        dataset_name=dataset_name,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
    )
    return dataloader
