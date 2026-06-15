"""Generate a tiny set of simple, recognizable icon images for an instant demo.

These let you build an index and try text search without sourcing photos
(real visual search shines on real photos — these are just a runnable smoke set).

    python make_sample_images.py
    python build_index.py --image-dir sample_images --index-dir index
    streamlit run app.py     # try: "星" / "house" / "魚" / "sun"
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "sample_images"
S = 224


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (S, S), "white")
    return img, ImageDraw.Draw(img)


def sun(d):
    for a in range(0, 360, 30):
        x1, y1 = 112 + 70 * math.cos(math.radians(a)), 112 + 70 * math.sin(math.radians(a))
        x2, y2 = 112 + 95 * math.cos(math.radians(a)), 112 + 95 * math.sin(math.radians(a))
        d.line([x1, y1, x2, y2], fill=(255, 190, 0), width=7)
    d.ellipse([66, 66, 158, 158], fill=(255, 205, 0))


def star(d):
    pts = []
    for i in range(10):
        r = 95 if i % 2 == 0 else 40
        a = math.radians(-90 + i * 36)
        pts.append((112 + r * math.cos(a), 112 + r * math.sin(a)))
    d.polygon(pts, fill=(255, 195, 0))


def house(d):
    d.polygon([(112, 40), (185, 110), (39, 110)], fill=(190, 70, 60))   # roof
    d.rectangle([60, 110, 164, 185], fill=(225, 200, 150))               # body
    d.rectangle([98, 140, 126, 185], fill=(120, 80, 50))                 # door


def tree(d):
    d.rectangle([102, 130, 122, 190], fill=(120, 80, 50))               # trunk
    d.ellipse([60, 50, 164, 150], fill=(60, 160, 80))                   # canopy


def car(d):
    d.rounded_rectangle([34, 110, 190, 160], 12, fill=(40, 110, 200))   # body
    d.rounded_rectangle([70, 80, 150, 118], 10, fill=(120, 170, 230))   # cabin
    d.ellipse([55, 150, 95, 190], fill=(30, 30, 30))                    # wheels
    d.ellipse([130, 150, 170, 190], fill=(30, 30, 30))


def fish(d):
    d.ellipse([50, 85, 160, 150], fill=(240, 140, 40))                  # body
    d.polygon([(150, 117), (195, 85), (195, 150)], fill=(240, 140, 40))  # tail
    d.ellipse([78, 105, 92, 119], fill="white")                         # eye
    d.ellipse([82, 109, 90, 117], fill="black")


def heart(d):
    d.ellipse([60, 70, 116, 126], fill=(220, 50, 70))
    d.ellipse([108, 70, 164, 126], fill=(220, 50, 70))
    d.polygon([(68, 110), (156, 110), (112, 175)], fill=(220, 50, 70))


def moon(d):
    d.ellipse([60, 50, 170, 160], fill=(245, 215, 90))                  # full disc
    d.ellipse([90, 45, 200, 155], fill="white")                        # cut crescent


def flower(d):
    for a in range(0, 360, 60):
        x, y = 112 + 45 * math.cos(math.radians(a)), 112 + 45 * math.sin(math.radians(a))
        d.ellipse([x - 30, y - 30, x + 30, y + 30], fill=(230, 120, 180))
    d.ellipse([90, 90, 134, 134], fill=(255, 210, 0))


def cloud(d):
    for cx, cy, r in [(85, 130, 35), (120, 115, 45), (160, 130, 35)]:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(150, 170, 200))
    d.rectangle([85, 130, 160, 160], fill=(150, 170, 200))


ICONS = {
    "sun": sun, "star": star, "house": house, "tree": tree, "car": car,
    "fish": fish, "heart": heart, "moon": moon, "flower": flower, "cloud": cloud,
}


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in ICONS.items():
        img, d = _canvas()
        fn(d)
        img.save(OUT / f"{name}.png")
    print(f"Wrote {len(ICONS)} sample images to {OUT}")


if __name__ == "__main__":
    build()
