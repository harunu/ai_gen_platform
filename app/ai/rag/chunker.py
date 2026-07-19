# WHY THIS FILE EXISTS:
# Splits long documents into smaller chunks before embedding.
# Why chunk? Embedding models have token limits (~512 tokens for all-MiniLM-L6-v2).
# Also, smaller chunks = more precise retrieval. A 500-word chunk about one topic
# returns a more focused result than a 5000-word chapter.
#
# Strategy: Recursive Character Splitter
# Tries to split on paragraph boundaries (\n\n) first.
# Falls back to sentence boundaries (\n), then word spaces, then characters.
# This preserves semantic coherence better than fixed-size splitting.

from dataclasses import dataclass, field
from typing import List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# Separator hierarchy: try these in order, from most to least natural
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


@dataclass
class TextChunk:
    """A chunk of text with its position metadata."""
    text: str
    chunk_index: int           # 0-based position in the original document
    start_char: int            # Character offset in the original document
    end_char: int              # End character offset
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class RecursiveCharacterChunker:
    """
    Recursively splits text using a hierarchy of separators.

    Algorithm:
    1. Try to split on the first separator (e.g. "\n\n")
    2. For any resulting piece that's still too long, recursively split with
       the next separator in the hierarchy
    3. Merge pieces that are too short (below chunk_size) with the next piece
       unless adding it would exceed chunk_size — then start a new chunk
    4. Add overlap: copy the last `overlap` characters of each chunk to
       the start of the next (preserves context across chunk boundaries)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None,
    ):
        """
        Args:
            chunk_size: Target character count per chunk (not strict — may be exceeded
                        if a single sentence is longer than chunk_size).
            chunk_overlap: Characters to repeat at the start of each subsequent chunk.
                          This ensures a sentence split across chunks is fully present
                          in at least one chunk.
            separators: Custom separator hierarchy. Defaults to natural language boundaries.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the separator hierarchy.
        Returns a list of pieces that are each <= chunk_size characters.
        """
        if not text:
            return []

        # If the text fits in a single chunk, no splitting needed
        if len(text) <= self.chunk_size:
            return [text]

        # Try each separator until we find one that splits the text
        separator = separators[-1]  # fallback: split on every character
        new_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split on the chosen separator
        splits = text.split(separator) if separator else list(text)

        # Recombine splits into chunks, recursing on any split that's still too long
        result: List[str] = []
        current: List[str] = []
        current_len = 0

        for split in splits:
            split_with_sep = split + separator if separator else split

            if len(split_with_sep) > self.chunk_size:
                # This piece is too long on its own — recurse with remaining separators
                if current:
                    # Flush current buffer first
                    result.append(separator.join(current) if separator else "".join(current))
                    current = []
                    current_len = 0
                # Recurse with the next separator tier
                sub_splits = self._split_text(split_with_sep, new_separators or [""])
                result.extend(sub_splits)
                continue

            if current_len + len(split_with_sep) > self.chunk_size and current:
                # Current buffer is full — flush it
                result.append(separator.join(current) if separator else "".join(current))
                # Keep overlap: retain trailing pieces that fit within overlap size
                overlap_pieces = []
                overlap_len = 0
                for piece in reversed(current):
                    piece_len = len(piece) + len(separator)
                    if overlap_len + piece_len <= self.chunk_overlap:
                        overlap_pieces.insert(0, piece)
                        overlap_len += piece_len
                    else:
                        break
                current = overlap_pieces
                current_len = overlap_len

            current.append(split)
            current_len += len(split_with_sep)

        # Flush remaining buffer
        if current:
            result.append(separator.join(current) if separator else "".join(current))

        return [r for r in result if r.strip()]

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[TextChunk]:
        """
        Chunk a document text into a list of TextChunk objects.

        Args:
            text: The full document text.
            metadata: Optional metadata to attach to each chunk (e.g. source, title).

        Returns:
            List of TextChunk objects in document order.
        """
        if not text or not text.strip():
            return []

        raw_chunks = self._split_text(text.strip(), self.separators)

        chunks: List[TextChunk] = []
        char_offset = 0

        for i, chunk_text in enumerate(raw_chunks):
            # Find the actual position of this chunk in the original text
            start = text.find(chunk_text, char_offset)
            if start == -1:
                start = char_offset
            end = start + len(chunk_text)

            chunks.append(TextChunk(
                text=chunk_text.strip(),
                chunk_index=i,
                start_char=start,
                end_char=end,
                metadata=metadata or {},
            ))

            char_offset = max(char_offset, end - self.chunk_overlap)

        logger.info(
            "document_chunked",
            original_chars=len(text),
            num_chunks=len(chunks),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        return chunks
