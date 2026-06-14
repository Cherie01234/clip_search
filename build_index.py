"""
インデックス構築スクリプト（一度だけ実行）
Usage: python build_index.py --image-dir data/images --index-dir index
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # faiss/torch の OpenMP 競合回避

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from encoder import CLIPEncoder
from indexer import FAISSIndexer

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BATCH_SIZE = 32  # VRAMに合わせて調整


def build_index(image_dir: str, index_dir: str) -> None:
    image_paths = sorted([
        p for p in Path(image_dir).rglob("*")
        if p.suffix.lower() in SUPPORTED_EXT
    ])
    print(f"対象画像: {len(image_paths)} 件")

    encoder = CLIPEncoder()
    indexer = FAISSIndexer(dim=512)

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Encoding"):
        batch = image_paths[i : i + BATCH_SIZE]

        # 読み込み失敗した画像はスキップ
        images, valid = [], []
        for p in batch:
            try:
                images.append(Image.open(p).convert("RGB"))
                valid.append(p)
            except Exception as e:
                print(f"  スキップ: {p.name} ({e})")

        if not images:
            continue

        vectors = encoder.encode_images(images)
        meta = [
            {
                "abs_path": str(p.resolve()),
                "rel_path": str(p.relative_to(image_dir)),
                "filename": p.name,
            }
            for p in valid
        ]
        indexer.add(vectors, meta)

    indexer.save(index_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--index-dir", default="index")
    args = parser.parse_args()
    build_index(args.image_dir, args.index_dir)