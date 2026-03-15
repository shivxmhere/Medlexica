from pydantic import BaseModel, Field, UUID4
from typing import List, Optional
from datetime import datetime
import uuid

class SourceDocument(BaseModel):
    doc_id: str
    text: str
    source_file: str
    rerank_score: float
    chunk_id: str

class ClinicalQueryRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=1000,
        description="Clinical question or symptom description")
    top_k_retrieval: int = Field(default=20, ge=5, le=50)
    top_n_rerank: int = Field(default=5, ge=1, le=10)
    stream: bool = Field(default=False)
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "query": "First-line treatment for Type 2 diabetes in a 45-year-old Indian patient",
                "top_k_retrieval": 20,
                "top_n_rerank": 5,
                "stream": False
            }]
        }
    }

class ClinicalQueryResponse(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    response: str
    retrieved_sources: List[SourceDocument]
    hallucination_flag: bool
    hallucination_score: float
    latency_ms: float
    model_version: str = "medlexica-v1"
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    disclaimer: str = ("This is AI-generated content for educational "
        "purposes only. Always consult a qualified medical professional "
        "for diagnosis and treatment decisions.")

class HealthResponse(BaseModel):
    status: str
    model: str
    uptime_seconds: float
    total_queries_served: int

class MetricsResponse(BaseModel):
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_queries: int
    hallucination_rate_last_100: float
