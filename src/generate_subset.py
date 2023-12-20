import json
import os
from pathlib import Path

IN_PATH = Path.home() / "datasets/acid"
OUT_PATH = Path.home() / "datasets/acid_subset"
NUM_CHUNKS = 3

if __name__ == "__main__":
    for split in IN_PATH.iterdir():
        (OUT_PATH / split.name).mkdir(exist_ok=True, parents=True)

        # Copy over a few chunks.
        chunks = sorted(list(split.iterdir()))[:NUM_CHUNKS]
        for chunk in chunks:
            os.system(f"cp {chunk} {OUT_PATH / split.name / chunk.name}")

        # Copy over the index.
        with (split / "index.json").open("r") as f:
            index = json.load(f)

        chunk_names = set(chunk.name for chunk in chunks)
        index = {k: v for k, v in index.items() if v in chunk_names}

        with (OUT_PATH / split.name / "index.json").open("w") as f:
            json.dump(index, f)
