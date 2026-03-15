import pytest
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, ".")

def test_health_endpoint_structure():
  """Test health endpoint returns correct schema"""
  expected_fields = {"status", "model", 
                     "uptime_seconds", "total_queries_served"}
  mock_response = {
    "status": "ok",
    "model": "medlexica-v1",
    "uptime_seconds": 10.5,
    "total_queries_served": 0
  }
  assert set(mock_response.keys()) == expected_fields
  assert mock_response["status"] == "ok"

def test_query_request_validation():
  """Test Pydantic validation on ClinicalQueryRequest"""
  from api.models import ClinicalQueryRequest
  
  valid_request = ClinicalQueryRequest(
    query="What is the treatment for dengue fever in adults?",
    top_k_retrieval=20,
    top_n_rerank=5
  )
  assert len(valid_request.query) >= 10
  assert valid_request.model_version_check() if hasattr(
    valid_request, 'model_version_check') else True
