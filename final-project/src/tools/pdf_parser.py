import fitz
import re
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    pages = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text()
        if text:
            cleaned = _clean_text(text)
            if cleaned.strip():
                pages.append({
                    "page_num": i + 1,
                    "text": cleaned,
                })
    doc.close()
    return pages


def _clean_text(text: str) -> str:
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = text.replace("\r", "").replace("\t", " ")
    text = re.sub(r"[\u200b\u00ad\u2060\ufeff]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def extract_chapters(pages: List[Dict]) -> List[Dict]:
    chapter_pattern = re.compile(
        r"第\s*\d+\s*章\s*.+",
    )
    result = []
    current_chapter = "概述"
    current_texts = []
    current_page = pages[0]["page_num"] if pages else 1

    for page in pages:
        lines = page["text"].split("\n")
        for line in lines:
            if chapter_pattern.match(line.strip()):
                if current_texts:
                    result.append({
                        "page_num": current_page,
                        "chapter": current_chapter,
                        "text": "\n".join(current_texts),
                    })
                current_chapter = line.strip()
                current_texts = [line.strip()]
                current_page = page["page_num"]
            else:
                current_texts.append(line.strip())

    if current_texts:
        result.append({
            "page_num": current_page,
            "chapter": current_chapter,
            "text": "\n".join(current_texts),
        })

    return result
