import re
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from apps.rag.schemas import Chunk, ParsedPage


TARGET_CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
HEADING_PREFIX = "标题路径："

MARKDOWN_SUFFIXES = {".md", ".markdown"}
MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")

CHINESE_NUMBER = "一二三四五六七八九十百千万"
PLAIN_HEADING_PATTERNS = [
    re.compile(rf"^第[{CHINESE_NUMBER}0-9]+[章节篇部分]\s*(.+)$"),
    re.compile(rf"^([{CHINESE_NUMBER}]+)、\s*(.+)$"),
    re.compile(rf"^（([{CHINESE_NUMBER}]+)）\s*(.+)$"),
    re.compile(r"^(\d+(?:\.\d+){0,3})[.、]?\s+(.+)$"),
]

fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TARGET_CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        "。", "！", "？",
        "；",
        ".", "!", "?",
        ";",
        "，", ",",
        " ",
        "",
    ],
)


@dataclass
class TextBlock:
    content: str
    heading_path: str | None = None


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def is_markdown_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in MARKDOWN_SUFFIXES


def heading_path(heading_stack: list[str]) -> str | None:
    if not heading_stack:
        return None
    return " > ".join(heading_stack)


def update_heading_stack(heading_stack: list[str], level: int, title: str) -> None:
    del heading_stack[level - 1 :]
    heading_stack.append(title.strip())


def is_confident_plain_heading(line: str) -> bool:
    line = line.strip()
    if not 4 <= len(line) <= 80:
        return False
    return not line.endswith(("。", "！", "？", "；", "，", ".", "!", "?", ";", ","))


def detect_plain_heading(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not is_confident_plain_heading(stripped):
        return None

    for pattern in PLAIN_HEADING_PATTERNS:
        match = pattern.match(stripped)
        if not match:
            continue

        marker = match.group(1)
        title = match.group(match.lastindex).strip()
        if pattern.pattern.startswith("^(\\d+"):
            level = marker.count(".") + 1
        elif stripped.startswith("（"):
            level = 2
        elif stripped.startswith("第") and "节" in stripped[:8]:
            level = 2
        else:
            level = 1
        return min(level, 6), title

    return None


def split_markdown_blocks(text: str) -> list[TextBlock]:
    blocks = []
    current: list[str] = []
    current_heading_path: str | None = None
    heading_stack: list[str] = []

    def flush_current() -> None:
        nonlocal current
        content = "\n".join(current).strip()
        if content:
            blocks.append(TextBlock(content=content, heading_path=current_heading_path))
        current = []

    for line in normalize_text(text).split("\n"):
        stripped = line.strip()
        if not stripped:
            flush_current()
            continue

        heading_match = MARKDOWN_HEADING.match(stripped)
        if heading_match:
            flush_current()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            update_heading_stack(heading_stack, level, title)
            current_heading_path = heading_path(heading_stack)
            current = [stripped]
            continue

        current.append(line.rstrip())

    flush_current()
    return blocks


def split_plain_blocks(text: str) -> list[TextBlock]:
    blocks = []
    current: list[str] = []
    current_heading_path: str | None = None
    heading_stack: list[str] = []

    def flush_current() -> None:
        nonlocal current
        content = "\n".join(current).strip()
        if content:
            blocks.append(TextBlock(content=content, heading_path=current_heading_path))
        current = []

    for line in normalize_text(text).split("\n"):
        stripped = line.strip()
        if not stripped:
            flush_current()
            continue

        detected_heading = detect_plain_heading(stripped)
        if detected_heading:
            flush_current()
            level, title = detected_heading
            update_heading_stack(heading_stack, level, title)
            current_heading_path = heading_path(heading_stack)
            current = [stripped]
            continue

        current.append(line.rstrip())

    flush_current()
    return blocks


def merge_small_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    merged = []
    current: TextBlock | None = None

    for block in blocks:
        if current is None:
            current = block
            continue

        same_heading = current.heading_path == block.heading_path
        candidate = f"{current.content}\n\n{block.content}".strip()
        if same_heading and len(candidate) <= TARGET_CHUNK_SIZE:
            current = TextBlock(content=candidate, heading_path=current.heading_path)
            continue

        merged.append(current)
        current = block

    if current is not None:
        merged.append(current)

    return merged


def prepend_heading_path(content: str, path: str | None) -> str:
    if not path:
        return content
    return f"{HEADING_PREFIX}{path}\n\n{content}"


def split_large_block(block: TextBlock) -> list[str]:
    if len(block.content) <= TARGET_CHUNK_SIZE:
        return [prepend_heading_path(block.content, block.heading_path)]

    chunks = []
    for chunk in fallback_splitter.split_text(block.content):
        content = chunk.strip()
        if content:
            chunks.append(prepend_heading_path(content, block.heading_path))
    return chunks


def split_text_structured(text: str, filename: str = "") -> list[str]:
    if is_markdown_file(filename):
        blocks = split_markdown_blocks(text)
    else:
        blocks = split_plain_blocks(text)

    chunks = []
    for block in merge_small_blocks(blocks):
        chunks.extend(split_large_block(block))
    return chunks


async def split_pages(pages: list[ParsedPage]) -> list[Chunk]:
    chunks = []
    chunk_index = 0

    for page in pages:
        for content in split_text_structured(page.content, page.filename):
            chunk_index += 1
            chunks.append(
                Chunk(
                    user_id=page.user_id,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    content=content,
                    document_id=page.document_id,
                    filename=page.filename,
                )
            )

    return chunks
