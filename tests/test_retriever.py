"""Tests for the arXiv retriever and semantic search."""

from __future__ import annotations

from ratiocinator.ideation.retriever import ArxivRetriever, Paper, SemanticSearch


def _sample_papers() -> list[Paper]:
    return [
        Paper(
            arxiv_id="2301.00001",
            title="Efficient Attention via Low-Rank Decomposition",
            abstract="We propose a low-rank attention mechanism that reduces complexity.",
            authors=["Alice"],
            published="2023-01-01T00:00:00",
        ),
        Paper(
            arxiv_id="2301.00002",
            title="GaLore: Memory-Efficient LLM Training",
            abstract="Gradient low-rank projection for memory-efficient training of LLMs.",
            authors=["Bob"],
            published="2023-06-01T00:00:00",
        ),
        Paper(
            arxiv_id="2301.00003",
            title="Convolutional Neural Networks for Image Classification",
            abstract="A survey of CNN architectures for classifying images.",
            authors=["Carol"],
            published="2022-01-01T00:00:00",
        ),
    ]


class TestPaper:
    def test_to_dict(self):
        p = _sample_papers()[0]
        d = p.to_dict()
        assert d["arxiv_id"] == "2301.00001"
        assert d["title"] == "Efficient Attention via Low-Rank Decomposition"
        assert isinstance(d["authors"], list)


class TestArxivRetriever:
    def test_cache_and_load(self, tmp_path):
        retriever = ArxivRetriever(tmp_path / "cache")
        papers = _sample_papers()
        retriever._cache_papers(papers)

        cached = retriever.get_cached()
        assert len(cached) == 3
        assert cached[0].arxiv_id == "2301.00001"

    def test_cache_deduplicates(self, tmp_path):
        retriever = ArxivRetriever(tmp_path / "cache")
        papers = _sample_papers()
        retriever._cache_papers(papers)
        retriever._cache_papers(papers)  # Cache again

        cached = retriever.get_cached()
        assert len(cached) == 3

    def test_empty_cache(self, tmp_path):
        retriever = ArxivRetriever(tmp_path / "cache")
        assert retriever.get_cached() == []


class TestSemanticSearch:
    def test_keyword_search(self):
        papers = _sample_papers()
        search = SemanticSearch(papers)
        results = search._keyword_search("attention low-rank efficient", top_k=2)
        assert len(results) > 0
        # First result should be the attention paper
        assert results[0][0].arxiv_id == "2301.00001"

    def test_keyword_search_no_matches(self):
        papers = _sample_papers()
        search = SemanticSearch(papers)
        results = search._keyword_search("xyznonexistent", top_k=5)
        assert len(results) == 0

    def test_search_falls_back_to_keyword(self):
        """Without sentence-transformers installed, search should still work."""
        papers = _sample_papers()
        search = SemanticSearch(papers)
        results = search.search("attention mechanism", top_k=2)
        assert len(results) > 0
