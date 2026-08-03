"""Export selected executed-notebook figures and deterministic GitHub artwork."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/llm-finance-matplotlib")

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
PREVIEW_DIR = ROOT / "assets" / "previews"
SELECTIONS = [
    (
        "00_start_here.ipynb",
        0,
        "learning_path.png",
        "A coherent learning path",
        "17 core labs plus one optional bridge",
    ),
    (
        "02_transformer_mechanics.ipynb",
        0,
        "attention_mechanics.png",
        "Transformer mechanics",
        "Attention implemented and inspected from first principles",
    ),
    (
        "04_point_in_time_rag.ipynb",
        0,
        "point_in_time_rag.png",
        "Point-in-time RAG",
        "Retrieval quality with temporal and metadata gates",
    ),
    (
        "11_governance_control_tower.ipynb",
        1,
        "governance_dashboard.png",
        "Governance control tower",
        "Release criteria preserve honest HOLD outcomes",
    ),
    (
        "16_capstone_earnings_intelligence.ipynb",
        0,
        "capstone_scorecard.png",
        "Governed capstone",
        "Evidence, controls, policy tests, and human review",
    ),
]
NAVY = "#0C2A40"
BLUE = "#2B6F97"
TEAL = "#19A7A0"
GOLD = "#E6B84A"
CREAM = "#F7F2EA"
SLATE = "#5B6B78"
WHITE = "#FFFFFF"


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_image(path: Path, image: Image.Image, *, format_name: str, **kwargs) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        image.save(temporary, format=format_name, **kwargs)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    properties = font_manager.FontProperties(
        family="DejaVu Sans",
        weight="bold" if bold else "normal",
    )
    return ImageFont.truetype(font_manager.findfont(properties), size)


def _notebook_images(path: Path) -> list[bytes]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    images = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            encoded = output.get("data", {}).get("image/png")
            if encoded:
                images.append(base64.b64decode(encoded))
    return images


def _frame(image_path: Path, title: str, caption: str, position: int) -> Image.Image:
    canvas = Image.new("RGB", (1200, 675), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1200, 104), fill=NAVY)
    draw.text((48, 22), title, font=_font(34, bold=True), fill=WHITE)
    draw.text((50, 67), caption, font=_font(18), fill="#DCE7ED")

    with Image.open(image_path) as source:
        chart = ImageOps.contain(source.convert("RGB"), (1100, 500))
    x = (1200 - chart.width) // 2
    y = 125 + (500 - chart.height) // 2
    canvas.paste(chart, (x, y))

    for index in range(len(SELECTIONS)):
        color = GOLD if index == position else "#C9D4DA"
        left = 510 + index * 40
        draw.rounded_rectangle((left, 642, left + 26, 651), radius=4, fill=color)
    return canvas


def _social_preview() -> Image.Image:
    canvas = Image.new("RGB", (1280, 640), NAVY)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((64, 58, 1216, 582), radius=28, fill=CREAM)
    draw.text((112, 102), "LARGE LANGUAGE MODELS", font=_font(48, bold=True), fill=NAVY)
    draw.text((112, 158), "IN FINANCE", font=_font(62, bold=True), fill=BLUE)
    draw.text(
        (114, 238),
        "Executable companion notebook suite",
        font=_font(27),
        fill=SLATE,
    )

    stages = ["Evidence", "Proposal", "Controls", "Policy", "Review"]
    colors = [BLUE, TEAL, GOLD, "#FF7A59", "#2A8C6B"]
    left = 112
    for index, (stage, color) in enumerate(zip(stages, colors, strict=True)):
        width = 184
        draw.rounded_rectangle((left, 342, left + width, 426), radius=18, fill=color)
        text_box = draw.textbbox((0, 0), stage, font=_font(22, bold=True))
        text_width = text_box[2] - text_box[0]
        draw.text(
            (left + (width - text_width) / 2, 369),
            stage,
            font=_font(22, bold=True),
            fill=WHITE if stage != "Controls" else NAVY,
        )
        if index < len(stages) - 1:
            draw.polygon(
                [(left + 190, 376), (left + 214, 384), (left + 190, 392)],
                fill=SLATE,
            )
        left += 220

    draw.text(
        (114, 486),
        "18 executed notebooks  •  deterministic fixtures  •  fail-closed governance",
        font=_font(24, bold=True),
        fill=NAVY,
    )
    draw.text(
        (114, 532),
        "PacktPublishing/LLMs-in-Finance",
        font=_font(21),
        fill=SLATE,
    )
    return canvas


def main() -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for position, (notebook, ordinal, filename, title, caption) in enumerate(SELECTIONS):
        images = _notebook_images(ROOT / "notebooks" / notebook)
        if ordinal >= len(images):
            raise ValueError(f"{notebook} has no figure at ordinal {ordinal}")
        destination = PREVIEW_DIR / filename
        _atomic_bytes(destination, images[ordinal])
        frames.append(_frame(destination, title, caption, position))

    _atomic_image(
        ROOT / "assets" / "finllm_suite_demo.gif",
        frames[0],
        format_name="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=2200,
        loop=0,
        disposal=2,
        optimize=False,
    )
    _atomic_image(
        ROOT / "assets" / "github_social_preview.png",
        _social_preview(),
        format_name="PNG",
        optimize=False,
    )
    print(f"Exported {len(SELECTIONS)} previews, one demo GIF, and one social image.")


if __name__ == "__main__":
    main()
