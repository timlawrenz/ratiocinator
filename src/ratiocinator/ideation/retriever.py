"""arXiv paper retrieval and local caching for literature-grounded ideation."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Cached representation of an arXiv paper."""

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    categories: list[str] = field(default_factory=list)
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArxivRetriever:
    """Queries arXiv API and caches results locally."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = cache_dir / "papers.jsonl"

    def search(
        self,
        query: str,
        max_results: int = 20,
        sort_by: str = "relevance",
    ) -> list[Paper]:
        """Search arXiv and cache results.

        Args:
            query: Search query (e.g., "efficient attention mechanisms").
            max_results: Maximum number of papers to return.
            sort_by: Sort order — "relevance" or "submittedDate".
        """
        import arxiv

        sort = arxiv.SortCriterion.Relevance
        if sort_by == "submittedDate":
            sort = arxiv.SortCriterion.SubmittedDate

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort,
        )

        papers = []
        for result in client.results(search):
            paper = Paper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                abstract=result.summary,
                authors=[a.name for a in result.authors],
                published=result.published.isoformat(),
                categories=result.categories,
                url=result.entry_id,
            )
            papers.append(paper)

        self._cache_papers(papers)
        logger.info("Retrieved %d papers for query: %s", len(papers), query)
        return papers

    def get_cached(self) -> list[Paper]:
        """Load all cached papers."""
        if not self._cache_file.exists():
            return []
        papers = []
        seen = set()
        for line in self._cache_file.read_text().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            if data["arxiv_id"] not in seen:
                papers.append(Paper(**data))
                seen.add(data["arxiv_id"])
        return papers

    def _cache_papers(self, papers: list[Paper]) -> None:
        """Append papers to the local JSONL cache (deduplicating)."""
        existing_ids = {p.arxiv_id for p in self.get_cached()}
        with self._cache_file.open("a") as f:
            for paper in papers:
                if paper.arxiv_id not in existing_ids:
                    f.write(json.dumps(paper.to_dict()) + "\n")
                    existing_ids.add(paper.arxiv_id)


class SemanticSearch:
    """Lightweight semantic search over paper abstracts.

    Uses sentence-transformers for embeddings and cosine similarity for ranking.
    Falls back to keyword matching if sentence-transformers is not installed.
    """

    def __init__(self, papers: list[Paper]) -> None:
        self.papers = papers
        self._embeddings = None
        self._model = None

    def search(self, query: str, top_k: int = 5) -> list[tuple[Paper, float]]:
        """Find the most relevant papers for a query.

        Returns list of (paper, score) tuples sorted by relevance.
        """
        try:
            return self._semantic_search(query, top_k)
        except ImportError:
            logger.info("sentence-transformers not available, falling back to keyword search")
            return self._keyword_search(query, top_k)

    def _semantic_search(self, query: str, top_k: int) -> list[tuple[Paper, float]]:
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

        if self._embeddings is None:
            texts = [f"{p.title}. {p.abstract}" for p in self.papers]
            self._embeddings = self._model.encode(texts, normalize_embeddings=True)

        query_emb = self._model.encode([query], normalize_embeddings=True)
        scores = (self._embeddings @ query_emb.T).flatten()

        indices = scores.argsort()[::-1][:top_k]
        return [(self.papers[i], float(scores[i])) for i in indices]

    def _keyword_search(self, query: str, top_k: int) -> list[tuple[Paper, float]]:
        """Simple keyword overlap scoring as fallback."""
        query_terms = set(query.lower().split())
        scored = []
        for paper in self.papers:
            text = f"{paper.title} {paper.abstract}".lower()
            text_terms = set(text.split())
            overlap = len(query_terms & text_terms)
            if overlap > 0:
                score = overlap / len(query_terms)
                scored.append((paper, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
