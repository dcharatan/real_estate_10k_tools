import subprocess
import sys
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from torchvision.io import encode_jpeg
from tqdm import tqdm

INPUT_IMAGE_DIR = Path("/mnt/hdd/datasets/acid/ACID/dataset_hi_res")
INPUT_METADATA_DIR = Path("/mnt/hdd/datasets/acid/ACID/ACID")
OUTPUT_DIR = Path("/mnt/sn850x/datasets/acid")

# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)


def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
    image_keys = set(
        example.name
        for example in tqdm((INPUT_IMAGE_DIR / stage).iterdir(), desc="Indexing images")
    )
    metadata_keys = set(
        example.stem
        for example in tqdm(
            (INPUT_METADATA_DIR / stage).iterdir(), desc="Indexing metadata"
        )
    )

    missing_image_keys = metadata_keys - image_keys
    if len(missing_image_keys) > 0:
        print(
            f"Found metadata but no images for {len(missing_image_keys)} examples.",
            file=sys.stderr,
        )
    missing_metadata_keys = image_keys - metadata_keys
    if len(missing_metadata_keys) > 0:
        print(
            f"Found images but no metadata for {len(missing_metadata_keys)} examples.",
            file=sys.stderr,
        )

    keys = image_keys & metadata_keys
    print(f"Found {len(keys)} keys.")
    return keys


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(
    example_path: Path,
    quality: int = 95,
) -> dict[int, UInt8[Tensor, " _"]]:
    """Load JPG images as raw bytes (do not decode)."""

    loaded = np.load(example_path / "data.npz")
    return {
        int(Path(file).stem): encode_jpeg(
            rearrange(torch.tensor(loaded[file]), "h w c -> c h w"), quality
        )
        for file in loaded.files
    }


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def load_metadata(example_path: Path) -> Metadata:
    with example_path.open("r") as f:
        lines = f.read().splitlines()

    url = lines[0]

    timestamps = []
    cameras = []

    for line in lines[1:]:
        timestamp, *camera = line.split(" ")
        timestamps.append(int(timestamp))
        cameras.append(np.fromstring(",".join(camera), sep=","))

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
    }


if __name__ == "__main__":
    for stage in ("train", "test", "validation"):
        keys = get_example_keys(stage)

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            print(
                f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB)."
            )
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        for key in keys:
            image_dir = INPUT_IMAGE_DIR / stage / key
            metadata_dir = INPUT_METADATA_DIR / stage / f"{key}.txt"
            num_bytes = get_size(image_dir)

            # Read images and metadata.
            images = load_images(image_dir)
            example = load_metadata(metadata_dir)

            # Merge the images into the example.
            try:
                example["images"] = [
                    images[timestamp.item()] for timestamp in example["timestamps"]
                ]
                assert len(images) == len(example["timestamps"])
            except (KeyError, AssertionError):
                print(f"Skipped {key}!")
                continue

            # Add the key to the example.
            example["key"] = key

            print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()
