import faiss
import numpy as np
import json
from pathlib import Path

class FAISSIndexer:
    """
    IndexFlatIP (Inner Product) を使用。
    L2正規化済みベクトルの内積 = cosine similarity なので
    normalize → IndexFlatIP でコサイン類似度検索ができる。
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict] = []

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        if vectors.shape[1] != self.dim:
            raise ValueError(f"次元不一致: {vectors.shape[1]} != {self.dim}")
        self.index.add(vectors.astype(np.float32))
        self.metadata.extend(metadata)

    def search(self, query: np.ndarray, top_k: int = 12) -> list[dict]:
        """
        Returns: [{"score": float, "abs_path": str, "rel_path": str, "filename": str}, ...]
        score は cosine similarity (0〜1、高いほど類似)
        """
        scores, indices = self.index.search(query.astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISSが -1 を返すのは結果なし
                continue
            item = self.metadata[idx].copy()
            item["score"] = float(score)
            results.append(item)
        return results

    def save(self, save_dir: str) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        (path / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"保存完了: {self.index.ntotal} 件 → {save_dir}")

    def load(self, save_dir: str) -> None:
        path = Path(save_dir)
        self.index = faiss.read_index(str(path / "faiss.index"))
        self.metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        print(f"インデックス読み込み: {self.index.ntotal} 件")

    @property
    def total(self) -> int:
        return self.index.ntotal