import torch
import open_clip
import numpy as np
from PIL import Image
from typing import Union

class CLIPEncoder:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device).eval()
        print(f"CLIPEncoder: {model_name} on {self.device}")

    def encode_images(self, images: Union[Image.Image, list[Image.Image]]) -> np.ndarray:
        """PIL Image (単枚またはリスト) → L2正規化済み numpy array (N, 512)"""
        if isinstance(images, Image.Image):
            images = [images]

        tensors = torch.stack(
            [self.preprocess(img) for img in images]
        ).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(tensors)
            features /= features.norm(dim=-1, keepdim=True)  # cosine similarityのためL2正規化

        return features.cpu().numpy().astype(np.float32)

    def encode_text(self, texts: Union[str, list[str]]) -> np.ndarray:
        """テキスト → L2正規化済み numpy array (N, 512)
        
        CLIPは画像とテキストを同じ空間に埋め込むため、
        テキストベクトルで画像を検索できる（ゼロショット）
        """
        if isinstance(texts, str):
            texts = [texts]

        tokens = self.tokenizer(texts).to(self.device)

        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().astype(np.float32)