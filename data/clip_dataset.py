import torch
from datasets import load_dataset, Image as HFImage
from transformers import CLIPImageProcessor
from tokenizers import Tokenizer


class ImageCaptionStream(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset_name: str = "lambdalabs/pokemon-blip-captions",
        split: str = "train",
        image_key: str = "image",
        text_key: str = "text",
        clip_image_processor=None,
        lm_tokenizer: Tokenizer = None,
        seq_len: int = 77,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.image_key = image_key
        self.text_key = text_key
        self.image_processor = clip_image_processor or CLIPImageProcessor.from_pretrained(
            "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
        )
        self.lm_tokenizer = lm_tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        ds = ds.cast_column(self.image_key, HFImage())
        for example in ds:
            image = example[self.image_key]
            caption = example[self.text_key]

            image_tensor = self.image_processor(image, return_tensors="pt").pixel_values[0]

            if self.lm_tokenizer is not None:
                encoded = self.lm_tokenizer.encode(caption)
                ids = encoded.ids[: self.seq_len]
                pad_len = self.seq_len - len(ids)
                if pad_len > 0:
                    ids = ids + [0] * pad_len
                input_ids = torch.tensor(ids, dtype=torch.long)
            else:
                input_ids = torch.zeros(self.seq_len, dtype=torch.long)

            yield {
                "image": image_tensor,
                "caption": caption,
                "input_ids": input_ids,
            }


def create_clip_dataloader(
    dataset_name: str = "lambdalabs/pokemon-blip-captions",
    split: str = "train",
    image_key: str = "image",
    text_key: str = "text",
    batch_size: int = 4,
    clip_image_processor=None,
    lm_tokenizer: Tokenizer = None,
    seq_len: int = 77,
):
    dataset = ImageCaptionStream(
        dataset_name=dataset_name,
        split=split,
        image_key=image_key,
        text_key=text_key,
        clip_image_processor=clip_image_processor,
        lm_tokenizer=lm_tokenizer,
        seq_len=seq_len,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
    )
