import torch
from datasets import load_dataset
from tokenizers import Tokenizer, decoders


def load_or_train_tokenizer(
    tokenizer_path: str,
) -> Tokenizer:
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


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
):
    tokenizer = load_or_train_tokenizer(tokenizer_path)
    dataset = FineWebDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        split=split,
        dataset_name=dataset_name,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        prefetch_factor=4,
    )
    return dataloader
