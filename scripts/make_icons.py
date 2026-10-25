"""Draw the app icon (indigo-violet gradient tile with a sound-wave mark) at
the sizes the PWA manifest and the Android app need. Pure Pillow, no assets.

Usage: python scripts/make_icons.py
Writes web/icons/icon-{192,512}.png, maskable-512.png and
android/app/res/mipmap-*/ic_launcher.png
"""
import os

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def draw(size, maskable=False, radius_frac=0.22):
    s = size * 4  # supersample, then downscale for smooth edges
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    grad = Image.new("RGBA", (s, s))
    a, b = (79, 70, 229), (124, 58, 237)
    px = grad.load()
    for y in range(s):
        for x in range(s):
            t = (x + y) / (2 * s)
            px[x, y] = tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3)) + (255,)
    mask = Image.new("L", (s, s), 0)
    r = 0 if maskable else int(s * radius_frac)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, s - 1, s - 1], radius=r, fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    # sound-wave bars, inside the maskable safe zone (central 60%)
    scale = 0.6 if maskable else 0.78
    cx, cy, w = s / 2, s / 2, s * scale
    heights = [0.12, 0.42, 0.78, 0.38, 0.1]
    bar = w * 0.1
    gap = (w - 5 * bar) / 4
    x0 = cx - w / 2
    for i, h in enumerate(heights):
        x = x0 + i * (bar + gap)
        hh = max(bar, h * w * 0.85)
        d.rounded_rectangle([x, cy - hh / 2, x + bar, cy + hh / 2], radius=bar / 2, fill=(255, 255, 255, 255))
    return img.resize((size, size), Image.LANCZOS)


def main():
    web = os.path.join(ROOT, "web", "icons")
    os.makedirs(web, exist_ok=True)
    draw(192).save(os.path.join(web, "icon-192.png"))
    draw(512).save(os.path.join(web, "icon-512.png"))
    draw(512, maskable=True).save(os.path.join(web, "maskable-512.png"))
    for name, px in {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}.items():
        d = os.path.join(ROOT, "android", "app", "res", f"mipmap-{name}")
        os.makedirs(d, exist_ok=True)
        draw(px).save(os.path.join(d, "ic_launcher.png"))
    print("icons written")


if __name__ == "__main__":
    main()
