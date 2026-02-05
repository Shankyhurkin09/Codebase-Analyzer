"""Token-safe code chunking using RecursiveCharacterTextSplitter."""

from langchain.text_splitter import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def chunk_code(code_files: list[dict]) -> list[dict]:
    """
    Split code files into chunks that respect token limits.
    Each code_file dict has: file (path), content.
    Returns list of dicts with: file, chunk, chunk_index.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []
    for code_file in code_files:
        file_path = code_file["file"]
        content = code_file["content"]

        if not content.strip():
            continue

        parts = splitter.split_text(content)
        for i, part in enumerate(parts):
            chunks.append({
                "file": file_path,
                "chunk": part,
                "chunk_index": i,
            })

    return chunks
