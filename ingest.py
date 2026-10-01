"""Document ingestion for the Fieldnotes app.

Used by the Streamlit app (home.py), and also runnable on its own to build the
ChromaDB index once, for example:

    python ingest.py lab2-starter-main.zip
    python ingest.py path/to/folder
"""

try:  # ChromaDB needs a newer SQLite on some hosts; harmless if pysqlite3 is absent
    __import__("pysqlite3")
    import sys as _sys

    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass

import hashlib
import html
import io
import re
import sys
import zipfile
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DB_DIR = APP_DIR / "chroma_db"
COLLECTION_NAME = "client_document_chunks"
COLLECTION_METADATA = {"hnsw:space": "cosine", "description": "Uploaded client research documents"}
MAX_FILE_BYTES = 20 * 1024 * 1024
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".docx", ".xlsx", ".pptx", ".ds_store"}
CODE_SUFFIXES = {".py", ".md", ".json", ".yml", ".yaml", ".toml", ".ipynb", ".cfg", ".ini", ".lock"}
IGNORE_NAMES = {"requirements.txt", "license", "readme", ".gitignore"}
IGNORE_DIRS = {"reusable_code", "node_modules", "__pycache__", "venv", ".venv", "__macosx"}


def wanted(name: str) -> bool:
    """False for code, config, hidden and junk files (so a whole repo can be ingested safely)."""
    path = Path(name)
    if any(part.startswith(".") for part in path.parts):
        return False
    if any(part.lower() in IGNORE_DIRS for part in path.parts[:-1]):
        return False
    if path.name.lower() in IGNORE_NAMES:
        return False
    return path.suffix.lower() not in SKIP_SUFFIXES | CODE_SUFFIXES


DOC_TYPES = ["email", "contract", "board_paper", "other"]

# Entities flagged per document (key -> pattern). Add names here for a different case.
ENTITIES = {
    "canvassian": r"canvassian",
    "jane_wu": r"jane\s+wu|\bms\.?\s+wu\b",
    "paywise": r"pay[\s-]?wise",
    "alphabear": r"alpha[\s-]?bear",
    "bravocat": r"bravo[\s-]?cat",
    "charlemont": r"charlemont",
    "deltaforce": r"delta[\s-]?force",
    "echona": r"echona",
}

# Topics flagged per chunk (key -> pattern), so searches can be narrowed deterministically.
TOPICS = {
    "coc": r"change[\s-]+(?:of|in)[\s-]+control|change of ownership|\bassign(?:ment|ed|s)?\b|"
    r"controlling interest|takeover|take-over|sale of (?:the )?(?:shares|business|company)|"
    r"(?:acquisition|acquirer|acquired) of (?:the )?(?:supplier|company|canvassian)",
    "financial": r"insolven|administrat(?:ion|or)|liquidat|receiver\b|going concern|overdue|late payment|"
    r"unpaid|\bdefault(?:ed|s)?\b|restructur|covenant|cash[\s-]?flow|financial difficult|credit (?:risk|rating)|"
    r"write[\s-]?off|bad debt|payment (?:delay|plan)|extension of (?:payment )?terms",
    "key_person": r"key[\s-]?(?:person|man|employee)|founder|resign|retention|non[\s-]?compete|notice period|"
    r"step(?:ping)? down|leav(?:e|ing) the (?:company|business)|succession|burn[\s-]?out|vesting|earn[\s-]?out",
    "legal": r"litigation|lawsuit|proceedings|dispute|\bclaim(?:s|ed)?\b|breach of|indemnit|penalt|regulator|"
    r"investigation|subpoena|class action",
    "security": r"data breach|security incident|vulnerabilit|\bcve-|ransomware|unauthori[sz]ed access|"
    r"penetration test|notifiable|privacy commissioner|\bincident\b",
    "ip": r"open[\s-]?source|\bgpl\b|licen[cs]e|intellectual property|infring|copyright|patent|escrow",
}

ENTITY_RE = {key: re.compile(pattern, re.I) for key, pattern in ENTITIES.items()}
TOPIC_RE = {key: re.compile(pattern, re.I) for key, pattern in TOPICS.items()}

_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)
DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",
    rf"\b\d{{1,2}}\s+{_MONTH}\.?\s+\d{{4}}\b",
    rf"\b{_MONTH}\.?\s+\d{{1,2}},?\s+\d{{4}}\b",
    r"\b\d{1,2}/\d{1,2}/\d{4}\b",
    rf"\b{_MONTH}\.?\s+\d{{4}}\b",
]
DATE_FORMATS = ["%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y", "%d/%m/%Y", "%B %Y", "%b %Y"]

ORG_RE = re.compile(
    r"\b((?:[A-Z][\w&'\-]*[ \t]+){0,3}[A-Z][\w&'\-]*[ \t]+(?:Pty\.?[ \t]+Ltd|Pty\.?[ \t]+Limited|Limited|Ltd|Inc|LLC))\b"
)
ORG_STOPWORDS = {"between", "and", "the", "this", "agreement", "by", "with", "of", "to", "from", "for", "dear", "party", "client"}


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if not self.skip_depth and data.strip():
            self.parts.append(data.strip())


def decode_bytes(raw: bytes, filename: str) -> str:
    encodings = ["utf-8-sig"]
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        encodings.append("utf-16")
    encodings += ["cp1252", "latin-1"]
    text = None
    for encoding in encodings:
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None or "\x00" in text:
        raise ValueError("This file does not appear to contain readable text.")
    if Path(filename).suffix.lower() in {".html", ".htm", ".xhtml"}:
        extractor = HTMLTextExtractor()
        extractor.feed(text)
        text = "\n".join(extractor.parts)
    return html.unescape(text).strip()


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


KEYWORD_TYPES = (("email", "email"), ("contract", "contract"), ("agreement", "contract"), ("board", "board_paper"), ("minutes", "board_paper"))


def _by_keyword(value: str):
    for key, label in KEYWORD_TYPES:
        if key in value:
            return label
    return None


def classify(text: str, path: str) -> str:
    parts = Path(path).parts
    label = _by_keyword("/".join(parts[:-1]).lower()) or _by_keyword(parts[-1].lower()) if parts else None
    if label:
        return label
    head = text[:1500].lower()
    if re.search(r"^(from|to|subject):", head, re.M):
        return "email"
    if re.search(r"this agreement|governing law|\bparties\b|schedule \d|\bwhereas\b", head):
        return "contract"
    if re.search(r"board (paper|meeting|minutes|pack)|\bagenda\b|resolutions?\b|\bdirectors?\b", head):
        return "board_paper"
    return "other"


def email_headers(text: str) -> dict:
    headers = {}
    head = text[:2000]
    for key in ("from", "to", "cc", "date", "subject"):
        match = re.search(rf"^{key}:\s*(.+)$", head, re.I | re.M)
        if match:
            headers[key] = match.group(1).strip()[:200]
    return headers


def parse_date(value: str) -> str:
    value = value.strip().replace(".", "")
    if not value:
        return ""
    try:
        return parsedate_to_datetime(value).date().isoformat()
    except Exception:
        pass
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def find_date(text: str) -> str:
    """Earliest date mentioned in the text (by position)."""
    found = []
    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, text):
            parsed = parse_date(match.group(0))
            if parsed:
                found.append((match.start(), parsed))
    return min(found)[1] if found else ""


def find_parties(text: str) -> str:
    found = []
    for match in ORG_RE.finditer(text[:2500]):
        words = match.group(1).split()
        while words and words[0].lower() in ORG_STOPWORDS:
            words.pop(0)
        name = " ".join(words)
        if name and name not in found:
            found.append(name)
        if len(found) == 4:
            break
    return ", ".join(found)


def extract_metadata(text: str, path: str, size_bytes: int) -> dict:
    name = Path(path).name
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    headings = [re.sub(r"^#{1,6}\s*", "", line).strip() for line in lines if re.match(r"^#{1,6}\s+", line)]
    doc_type = classify(text, path)
    headers = email_headers(text) if doc_type == "email" else {}
    title = headers.get("subject") or (headings[0] if headings else (lines[0] if lines else Path(name).stem))
    title = re.sub(r"\s+", " ", title).strip("#* _")[:160] or Path(name).stem
    meta = {
        "source_id": sha(path.casefold()),
        "source_name": name,
        "source_path": path,
        "title": title,
        "file_type": Path(name).suffix.lower().lstrip(".") or "text",
        "doc_type": doc_type,
        "doc_date": parse_date(headers.get("date", "")) or find_date(text[:3000]),
        "parties": find_parties(text),
        "email_from": headers.get("from", ""),
        "email_to": headers.get("to", ""),
        "size_bytes": size_bytes,
        "heading_count": len(headings),
        "content_hash": sha(text),
    }
    for key, pattern in ENTITY_RE.items():
        meta[f"ent_{key}"] = bool(pattern.search(text))
    return meta


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []
    chunk_size = max(200, chunk_size)
    overlap = min(max(0, overlap), chunk_size // 2)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            floor = start + int(chunk_size * 0.65)
            boundary = max(text.rfind("\n\n", floor, end), text.rfind("\n", floor, end))
            if boundary < floor:
                boundary = text.rfind(" ", floor, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def index_document(collection, raw: bytes, path: str, chunk_size: int = 1000, overlap: int = 150, force: bool = False):
    """Index one document. Returns ("indexed" | "skipped", number_of_chunks)."""
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("Files must be 20 MB or smaller.")
    text = decode_bytes(raw, path)
    if not text:
        raise ValueError("The file is empty after text extraction.")
    meta = extract_metadata(text, path, len(raw))
    existing = collection.get(where={"source_id": meta["source_id"]}, include=["metadatas"])
    if existing["ids"] and not force and existing["metadatas"][0].get("content_hash") == meta["content_hash"]:
        return "skipped", len(existing["ids"])
    pieces = chunk_text(text, chunk_size, overlap)
    if not pieces:
        raise ValueError("No text could be chunked from this file.")
    if existing["ids"]:
        collection.delete(where={"source_id": meta["source_id"]})

    meta["uploaded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header = f"[{meta['doc_type']} | {meta['title']} | {meta['source_name']}"
    header += f" | {meta['doc_date']}]\n" if meta["doc_date"] else "]\n"
    ids, documents, metadatas = [], [], []
    for index, (start, end, piece) in enumerate(pieces):
        ids.append(f"{meta['source_id']}-{meta['content_hash'][:12]}-{index}")
        documents.append(header + piece)
        chunk_meta = {**meta, "chunk_index": index, "char_start": start, "char_end": end}
        for topic, pattern in TOPIC_RE.items():
            chunk_meta[f"kw_{topic}"] = bool(pattern.search(piece))
        metadatas.append(chunk_meta)
    for offset in range(0, len(ids), 500):
        collection.upsert(
            ids=ids[offset : offset + 500],
            documents=documents[offset : offset + 500],
            metadatas=metadatas[offset : offset + 500],
        )
    return "indexed", len(pieces)


def iter_zip(data: bytes):
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir() or info.file_size > MAX_FILE_BYTES or not wanted(info.filename):
                continue
            yield info.filename, zf.read(info)


def expand_files(files):
    """files: iterable of (name, bytes). Zip files are expanded into their members."""
    for name, raw in files:
        if name.lower().endswith(".zip"):
            yield from iter_zip(raw)
        else:
            yield name, raw


def iter_folder(folder: Path):
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(folder))
        if wanted(relative):
            yield relative, path.read_bytes()


def main(argv):
    if len(argv) < 2:
        print("Usage: python ingest.py <file.zip | folder>")
        return 1
    import chromadb

    target = Path(argv[1])
    if target.is_dir():
        files = iter_folder(target)
    else:
        files = expand_files([(target.name, target.read_bytes())])
    DB_DIR.mkdir(parents=True, exist_ok=True)
    collection = chromadb.PersistentClient(path=str(DB_DIR)).get_or_create_collection(
        name=COLLECTION_NAME, metadata=COLLECTION_METADATA
    )
    indexed = skipped = chunks = 0
    failures = []
    for number, (name, raw) in enumerate(files, 1):
        try:
            status, count = index_document(collection, raw, name)
            if status == "indexed":
                indexed += 1
                chunks += count
            else:
                skipped += 1
        except Exception as error:
            failures.append(f"{name}: {error}")
        if number % 25 == 0:
            print(f"  processed {number} files…", flush=True)
    print(f"Done. Indexed {indexed} documents ({chunks} passages), skipped {skipped} already indexed, {len(failures)} failed.")
    for line in failures[:20]:
        print("  FAILED", line)
    print(f"Collection now holds {collection.count()} passages in {DB_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
