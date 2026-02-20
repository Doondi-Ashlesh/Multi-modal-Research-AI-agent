"""Multi-modal input handling: text, images, and PDFs."""
import base64
import io
from pathlib import Path
from typing import Any

from pypdf import PdfReader


# Supported image extensions for vision API
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def load_text(path: Path) -> str:
    """Load plain text file."""
    return path.read_text(encoding="utf-8", errors="replace")


def load_image_as_base64(path: Path) -> str:
    """Load image and return base64 string for API."""
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


def load_pdf_text(path: Path) -> str:
    """Extract text from PDF. Returns concatenated page text."""
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts) if parts else "[No text could be extracted from PDF.]"


def load_pdf_first_page_image(path: Path) -> tuple[str, str] | None:
    """Render first page of PDF as image and return (base64, media_type). Optional for vision."""
    try:
        import pdf2image  # optional: pip install pdf2image; needs poppler
        images = pdf2image.convert_from_path(path, first_page=1, last_page=1)
        if not images:
            return None
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.standard_b64encode(buf.read()).decode("utf-8")
        return b64, "image/png"
    except Exception:
        return None


def build_message_content(
    user_text: str,
    file_paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """
    Build OpenAI-style message content list supporting text + images + PDF text.
    Each file is added as appropriate: images as image_url, PDFs as extracted text.
    """
    content: list[dict[str, Any]] = []

    if user_text.strip():
        content.append({"type": "text", "text": user_text})

    if not file_paths:
        return content if content else [{"type": "text", "text": "No input provided."}]

    for path in file_paths:
        path = Path(path)
        if not path.exists():
            content.append({"type": "text", "text": f"[File not found: {path}]"})
            continue
        ext = path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            b64 = load_image_as_base64(path)
            media_type = get_image_media_type(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64}"},
            })
        elif ext == ".pdf":
            pdf_text = load_pdf_text(path)
            content.append({
                "type": "text",
                "text": f"[PDF: {path.name}]\n\n{pdf_text}",
            })
        else:
            try:
                text = load_text(path)
                content.append({"type": "text", "text": f"[File: {path.name}]\n\n{text}"})
            except Exception as e:
                content.append({"type": "text", "text": f"[Could not read {path.name}: {e}]"})

    return content
