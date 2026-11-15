
from langchain_core.documents import Document


SNIPPET_CHARACTERS = 400


def snippet_text(content: str) -> str:
    """A short, single-line excerpt of a retrieved chunk."""

    compact = " ".join(content.split())
    if len(compact) <= SNIPPET_CHARACTERS:
        return compact
    return compact[:SNIPPET_CHARACTERS].rstrip() + "…"


def apply_citation_labels(citations: list[dict], labels: dict | None) -> list[dict]:
    """Attach registry identity and drop citations outside the selection.

    ``labels`` is keyed by stored filename. ``None`` leaves citations
    unchanged for callers that are not enforcing a selection.
    """

    if labels is None:
        return citations

    labeled = []
    for citation in citations:
        label = labels.get(citation.get("source"))
        if label is None:
            continue
        labeled.append({**citation, **label})
    return labeled


def build_context(
    documents: list[Document],
    max_context_chars: int = 12000
):
    """
    Build clean, deduplicated context from retrieved documents.
    Preserve source and page information for citations.
    """

    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be greater than zero.")

    if not documents:
        return "", []

    seen_chunks = set()
    context_parts = []
    citations = []
    current_length = 0

    for index, document in enumerate(documents, start=1):

        content = document.page_content.strip()

        if not content:
            continue

        metadata = document.metadata

        source = metadata.get("source", "Unknown source")
        page = metadata.get("page", "Unknown page")
        chunk_id = metadata.get("chunk_id")

        # Avoid duplicate chunks.
        chunk_key = (source, page, chunk_id, content)

        if chunk_key in seen_chunks:
            continue

        seen_chunks.add(chunk_key)

        reference = f"[Source {len(citations) + 1}]"

        formatted_chunk = (
            f"{reference}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{content}"
        )

        remaining = max_context_chars - current_length

        if len(formatted_chunk) > remaining:
            if remaining <= 0:
                break

            # Avoid adding a partial chunk that cannot be cited reliably.
            continue

        context_parts.append(formatted_chunk)

        citations.append({
            "reference": reference,
            "source": source,
            "page": page,
            "chunk_id": chunk_id,
            "snippet": snippet_text(content),
        })

        current_length += len(formatted_chunk) + 2

    return "\n\n".join(context_parts), citations

