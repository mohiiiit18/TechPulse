"""TechPulse Phase 5 — Resume parsing (PyMuPDF).

Upload → raw text → structured ResumeData. Handles:
- text-based PDFs (PyMuPDF page-by-page extraction),
- scanned/image PDFs (no extractable text → explicit "scanned PDF" result,
  never fabricated content),
- encrypted / corrupt / empty PDFs (typed errors).

Privacy: parsing happens in memory only. Nothing is written to disk unless the
user explicitly opts in (the app never does by default).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

MAX_RESUME_MB = 5
ALLOWED_SUFFIX = ".pdf"


class ResumeParseError(Exception):
    """User-facing resume parsing failure (message is safe to display)."""


@dataclass
class ResumeData:
    """Structured resume content. Missing fields stay empty — never invented."""

    raw_text: str = ""
    full_name: str = ""          # best-effort, empty if not confidently found
    email: str = ""
    phone: str = ""
    education: list[str] = field(default_factory=list)
    experience: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    skills_by_category: dict[str, list[str]] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)
    n_pages: int = 0
    is_scanned: bool = False

    @property
    def all_skills(self) -> list[str]:
        return [s for skills in self.skills_by_category.values() for s in skills]


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{2,4}\)?[\s.\-]?){2,4}\d{2,4}"
)
_EDU_DEGREE_RE = re.compile(
    r"(?im)^[^\n]{0,120}?(bachelor|master|ph\.?d|doctorate|b\.?sc|b\.?a|m\.?sc|m\.?a|m\.?eng|"
    r"b\.?tech|m\.?tech|mba|associate|diploma|high school|bootcamp)[^\n]{0,120}$"
)
_BULLET_RE = re.compile(r"(?m)^\s*[•\-\u2022\u25cf\u25aa\*►▪]\s*(.+)$")
_DATE_RANGE_RE = re.compile(
    r"(19|20)\d{2}\s*(–|-|—|to|until)\s*((19|20)\d{2}|present|current|now)", re.I
)


def validate_upload(name: str, size_bytes: int) -> str | None:
    """Return an error message for invalid uploads, or None if acceptable."""
    if not name or not name.lower().endswith(ALLOWED_SUFFIX):
        return "Unsupported file type — please upload a PDF resume."
    if size_bytes <= 0:
        return "The uploaded file is empty."
    if size_bytes > MAX_RESUME_MB * 1024 * 1024:
        return f"File too large ({size_bytes / 1024 / 1024:.1f} MB) — limit is {MAX_RESUME_MB} MB."
    return None


def pdf_to_text(file_bytes: bytes) -> tuple[str, int, bool]:
    """Extract text from PDF bytes. Returns (text, page_count, is_scanned).

    Raises ResumeParseError for corrupt/encrypted PDFs. A PDF whose pages
    contain no extractable text layer is flagged scanned (image-only).
    """
    try:
        with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
            if doc.is_encrypted:
                if not doc.authenticate(""):
                    raise ResumeParseError(
                        "This PDF is password-protected — please upload an unprotected copy.")
            n_pages = doc.page_count
            if n_pages == 0:
                raise ResumeParseError("This PDF contains no pages.")
            parts = [page.get_text("text") or "" for page in doc]
    except ResumeParseError:
        raise
    except Exception as e:  # corrupt file, wrong magic bytes, etc.
        raise ResumeParseError(f"Could not read this PDF — the file may be corrupted. ({type(e).__name__})")

    text = "\n".join(parts).strip()
    if len(text) < 40:
        # Near-zero text layer ⇒ image-only scan. Empty docs get the same
        # honest "no text" treatment (message mentions scanned as the cause).
        raise ResumeParseError(
            "No selectable text was found — this PDF appears to be empty or a "
            "scanned/image-only document. Please upload a text-based PDF (export "
            "from Word/Google Docs, or run OCR first).")
    if len(text) > 200_000:
        text = text[:200_000]  # hard safety cap; resumes are far smaller
    return text, n_pages, False


def _guess_name(text: str) -> str:
    """Best-effort full name from the top of the resume. Empty if unsure.

    Heuristic: first non-empty line of the first page-ish block, 2–4 words,
    alphabetic, title-cased, and not a header/email/phone line.
    """
    for line in text.splitlines()[:8]:
        cand = line.strip()
        if not cand or "@" in cand or any(ch.isdigit() for ch in cand):
            continue
        words = cand.split()
        if not 2 <= len(words) <= 4:
            continue
        if not all(re.fullmatch(r"[A-Za-z][A-Za-z.'\-]*", w) for w in words):
            continue
        if any(k in cand.lower() for k in ("resume", "curriculum", "cv", "profile", "summary")):
            continue
        return cand.title()
    return ""


def _split_lines_block(section_text: str, line_pattern: re.Pattern) -> list[str]:
    """Extract structured items from a section: bullets, date-ranged lines or
    degree lines. Only literally-present lines are returned."""
    if not section_text:
        return []
    items: list[str] = []
    for m in _BULLET_RE.finditer(section_text):
        item = m.group(1).strip()
        if 3 < len(item) < 200:
            items.append(item)
    if items:
        return items
    for line in section_text.splitlines():
        line = line.strip()
        if 3 < len(line) < 200 and (line_pattern.search(line)):
            items.append(line)
    return items


def parse_resume(file_bytes: bytes) -> ResumeData:
    """Full pipeline: bytes → ResumeData (in-memory, nothing persisted)."""
    from src.skill_extractor import extract_certifications, extract_skills, find_sections

    text, n_pages, _ = pdf_to_text(file_bytes)

    email_m = _EMAIL_RE.search(text)
    email = email_m.group(0) if email_m else ""

    phone = ""
    # Search phone near the top of the document, outside email context.
    head = text[:600]
    for m in _PHONE_RE.finditer(head):
        digits = re.sub(r"\D", "", m.group(0))
        if 9 <= len(digits) <= 14:
            phone = m.group(0).strip()
            break

    sections = find_sections(text)

    education = _split_lines_block(sections.get("education", ""), _EDU_DEGREE_RE)
    if not education:  # degrees listed outside a labeled section
        education = [m.group(0).strip() for m in _EDU_DEGREE_RE.finditer(text)][:6]

    experience = _split_lines_block(sections.get("experience", ""), _DATE_RANGE_RE)
    if not experience:
        experience = [ln.strip() for ln in text.splitlines()
                      if _DATE_RANGE_RE.search(ln) and 3 < len(ln.strip()) < 200][:8]

    projects = _split_lines_block(sections.get("projects", ""), _DATE_RANGE_RE)

    cert_lines = extract_certifications(text)
    section_certs = _split_lines_block(sections.get("certifications", ""), _DATE_RANGE_RE)
    certifications = list(dict.fromkeys(section_certs + cert_lines))[:10]

    skills_section = sections.get("skills", "") or text
    skills_by_category = extract_skills(skills_section)

    return ResumeData(
        raw_text=text,
        full_name=_guess_name(text),
        email=email,
        phone=phone,
        education=education,
        experience=experience,
        projects=projects,
        certifications=certifications,
        skills_by_category=skills_by_category,
        sections=sections,
        n_pages=n_pages,
        is_scanned=False,
    )
