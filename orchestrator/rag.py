"""
RAG — Retrieval-Augmented Generation (Phase 5).

The idea: before the agent answers, RETRIEVE the most relevant reference
material from a knowledge corpus, and inject it into the prompt. The agent
then answers GROUNDED in that material and CITES which document each fact
came from.

Retrieval here uses TF-IDF (from scikit-learn) — a classic, fast, fully
deterministic technique. The corpus is small (~13 business definitions),
so TF-IDF is plenty; embedding databases would be overkill for v1.

TF-IDF in one sentence: it scores how important each word is to a document,
then ranks documents by how well their words overlap with the query.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    """One retrievable chunk of the knowledge corpus."""

    doc_id: str    # short, citable id, e.g. "Churn"
    title: str
    text: str


def load_definitions(path: str = "data/definitions.md") -> list[Document]:
    """Split the definitions markdown into one Document per **term** entry.

    A definition looks like:  **Churn** - A customer is considered churned...
    Each such paragraph becomes one Document, with the bold term as its id.
    """
    raw = Path(path).read_text()
    paragraphs = raw.split("\n\n")

    documents = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        # Keep only paragraphs that start with a bold **term**
        if not paragraph.startswith("**"):
            continue

        # The term is the text between the first pair of ** **
        term_end = paragraph.find("**", 2)
        if term_end == -1:
            continue
        term = paragraph[2:term_end].strip()

        documents.append(Document(doc_id=term, title=term, text=paragraph))

    return documents


def format_context(retrieved: list[tuple[Document, float]]) -> str:
    """Turn retrieved (Document, score) pairs into labelled reference text.

    Each block is tagged with its [doc_id] so the agent can cite it.
    """
    blocks = []
    for doc, score in retrieved:
        blocks.append(f"[{doc.doc_id}]\n{doc.text}")
    return "REFERENCE MATERIAL:\n\n" + "\n\n".join(blocks)


class Retriever:
    """TF-IDF retriever over a list of Documents."""

    def __init__(self, documents: list[Document]):
        self.documents = documents

        # Build the TF-IDF model from the corpus text
        self.vectorizer = TfidfVectorizer(stop_words="english")
        corpus_texts = [doc.text for doc in documents]
        self.matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        """Return the top-k most relevant documents for a query.

        Each result is a (Document, similarity_score) pair, best first.
        """
        # Turn the query into the same TF-IDF space, then score every doc
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]

        # Pair each document with its score
        scored = []
        for doc, score in zip(self.documents, scores):
            scored.append((doc, float(score)))

        # Sort by score, highest first, and keep the top k
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]
