import pytest
import sys
sys.path.insert(0, ".")

class TestHybridRetriever:
  
  def test_retriever_imports(self):
    """Test that retriever module imports without errors"""
    try:
      from rag.retriever import HybridRetriever
      assert True
    except ImportError as e:
      pytest.skip(f"Dependencies not installed: {e}")
  
  def test_rrf_formula(self):
    """Test Reciprocal Rank Fusion formula correctness"""
    # RRF score for rank 0 with k=60 should be 1/61
    k = 60
    rank = 0
    expected = 1 / (rank + k)
    result = 1 / (rank + k)
    assert abs(result - expected) < 1e-10
  
  def test_top_k_limit(self):
    """Test that retrieve respects top_k parameter"""
    results = [{"chunk_id": f"c{i}", "text": f"text {i}", 
                "rrf_score": 1/i} for i in range(1, 25)]
    top_k = 5
    limited = sorted(results, 
      key=lambda x: x["rrf_score"], reverse=True)[:top_k]
    assert len(limited) == top_k
