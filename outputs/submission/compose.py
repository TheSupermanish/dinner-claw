"""Compose the final 1280x720 cover from a rendered frame plus a title block.

The image underneath is a real captured frame (`cover.py`), not a mock-up. The only
thing added here is text, and every number in it is quoted from
`outputs/regression-2026-09-16/`.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
SRC = HERE / "cover-az90.png"
OUT = HERE / "cover.png"

FONTS = Path("/System/Library/Fonts/Supplemental")
title_font = ImageFont.truetype(str(FONTS / "Arial Bold.ttf"), 64)
sub_font = ImageFont.truetype(str(FONTS / "Arial.ttf"), 26)
stat_font = ImageFont.truetype(str(FONTS / "Arial Bold.ttf"), 22)
foot_font = ImageFont.truetype(str(FONTS / "Arial.ttf"), 18)

image = Image.open(SRC).convert("RGB")

# Text sat straight on top of a yellow arm and a brown table and was unreadable in
# both places. Darken the two text bands with a gradient so the render still shows
# through but the type has contrast everywhere it lands.
scrim = Image.new("L", (1, image.height), 0)
for y in range(image.height):
    if y < 230:
        scrim.putpixel((0, y), int(200 * (1 - y / 230) ** 1.2))
    elif y > 470:
        scrim.putpixel((0, y), int(205 * ((y - 470) / (image.height - 470)) ** 0.9))
mask = scrim.resize((image.width, image.height))
image = Image.composite(Image.new("RGB", image.size, (0, 0, 0)), image, mask)
draw = ImageDraw.Draw(image)

draw.text((56, 54), "DINNER CLAW", font=title_font, fill=(255, 255, 255))
draw.text((60, 132), "Two SO-101 arms lifting a plate neither arm can lift alone",
          font=sub_font, fill=(214, 214, 214))

# Measured, from outputs/regression-2026-09-16/. Nothing here ran on Intel hardware.
stats = [
    "9/10 held-out, two arms",
    "0/10 one arm, kept as evidence",
    "15 deg levelness bound",
]
y = 560
for line in stats:
    draw.rectangle((56, y + 6, 62, y + 24), fill=(247, 199, 43))
    draw.text((78, y), line, font=stat_font, fill=(255, 255, 255))
    y += 34

draw.text((56, 676), "MuJoCo 3.13 | contact-gated success | negative controls at 0",
          font=foot_font, fill=(198, 198, 198))

image.save(OUT)
print(OUT, image.size)
