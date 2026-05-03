from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    text: str
    character_start: int
    character_end: int


class ChunkingService:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, content: str) -> list[TextChunk]:
        normalized = content.strip()
        if not normalized:
            return []

        chunks: list[TextChunk] = []
        start = 0
        text_length = len(normalized)
        while start < text_length:
            raw_end = min(start + self.chunk_size, text_length)
            end = self._find_clean_break(normalized, start, raw_end)
            chunk_text = normalized[start:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_index=len(chunks) + 1,
                        text=chunk_text,
                        character_start=start,
                        character_end=end,
                    )
                )
            if end >= text_length:
                break
            start = max(0, end - self.chunk_overlap)
        return chunks

    def _find_clean_break(self, text: str, start: int, raw_end: int) -> int:
        if raw_end >= len(text):
            return len(text)

        search_window = text[start:raw_end]
        for separator in ("\n\n", "\n", ". ", " "):
            idx = search_window.rfind(separator)
            if idx > self.chunk_size * 0.5:
                return start + idx + len(separator)
        return raw_end

