from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import time
import json
import logging
from collections import deque
from api.models import (
    ClinicalQueryRequest, 
    ClinicalQueryResponse, 
    HealthResponse, 
    MetricsResponse
)

# Global state
engine = None
start_time = time.time()
query_log = deque(maxlen=100)  # last 100 queries
total_queries = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global engine
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Loading MedLexica engine...")
    try:
        from api.inference import MedLexicaEngine
        engine = MedLexicaEngine()
        logger.info("Engine loaded successfully ✓")
    except Exception as e:
        logger.error(f"Failed to load engine during startup (ignoring for build checks): {e}")
        engine = None
    yield
    # Shutdown
    logger.info("Shutting down MedLexica API")

app = FastAPI(
    title="MedLexica API",
    description="Clinical AI Assistant — Fine-tuned Phi-3-mini with Hybrid RAG Pipeline",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def log_requests(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = (time.perf_counter() - start) * 1000
    
    log_entry = {
        "timestamp": time.time(),
        "path": str(request.url.path),
        "latency_ms": round(latency, 2),
        "status_code": response.status_code
    }
    
    # Safe dump
    try:
        with open("api/query_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
        
    return response

@app.post("/query", response_model=ClinicalQueryResponse)
async def query_endpoint(request: ClinicalQueryRequest):
    global total_queries
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    result = engine.generate(
        query=request.query,
        top_k=request.top_k_retrieval,
        top_n=request.top_n_rerank
    )
    
    total_queries += 1
    query_log.append({
        "latency_ms": result.latency_ms,
        "hallucination_flag": result.hallucination_flag
    })
    
    return result

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model="medlexica-v1",
        uptime_seconds=round(time.time() - start_time, 2),
        total_queries_served=total_queries
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    import numpy as np
    
    if not query_log:
        latencies = [0]
        hall_rate = 0.0
    else:
        latencies = [q["latency_ms"] for q in query_log]
        hall_rate = sum(1 for q in query_log if q["hallucination_flag"]) / len(query_log)
    
    return MetricsResponse(
        p50_latency_ms=round(float(np.percentile(latencies, 50)), 2),
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 2),
        p99_latency_ms=round(float(np.percentile(latencies, 99)), 2),
        total_queries=total_queries,
        hallucination_rate_last_100=round(hall_rate, 4)
    )

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("demo/medlexica_ui.html", encoding="utf-8") as f:
        return f.read()
