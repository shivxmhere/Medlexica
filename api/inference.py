from functools import lru_cache
import time
import json
import logging
from typing import List
import os

class MedLexicaEngine:
    
    def __init__(self, config_path: str = ".env"):
        # Load env variables
        from dotenv import load_dotenv
        load_dotenv(config_path)
        
        self.model_version = "medlexica-v1"
        self.logger = logging.getLogger(__name__)
        
        # Load quantized model via llama-cpp-python
        from llama_cpp import Llama
        model_path = os.getenv("BASE_MODEL_PATH")
        # In a real scenario, this would load a quantized GGUF format 
        # For simplicity assuming it's correctly mapped via llama_cpp limits setup
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4,
            verbose=False
        )
        self.logger.info("LLM loaded from %s", model_path)
        
        # Load RAG components
        from rag.retriever import HybridRetriever
        from rag.reranker import CrossEncoderReranker
        
        self.retriever = HybridRetriever(
            index_dir=os.getenv("FAISS_INDEX_PATH", "./rag")
        )
        self.reranker = CrossEncoderReranker()
        
        # Load NLI model for hallucination detection
        from sentence_transformers import CrossEncoder
        self.nli_model = CrossEncoder(
            "cross-encoder/nli-distilroberta-base"
        )
        
        self.logger.info("MedLexicaEngine fully initialized")

    @lru_cache(maxsize=100)
    def _get_cached_embedding(self, text: str):
        # Cache embeddings for repeated queries
        return self.retriever.encoder.encode(text)

    def _build_prompt(self, query: str, contexts: List[dict]) -> str:
        context_text = "\n\n".join([
            f"[Source {i+1}]: {c['text']}" 
            for i, c in enumerate(contexts)
        ])
        return f"""<|system|>
You are MedLexica, a clinical AI assistant trained on Indian 
medical literature. Answer based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly.
Always include relevant ICD-10 codes. End with a disclaimer.<|end|>
<|user|>
CONTEXT:
{context_text}

QUESTION: {query}<|end|>
<|assistant|>"""

    def _check_hallucination(self, response: str, contexts: List[dict]) -> tuple:
        combined_context = " ".join([c["text"] for c in contexts])
        scores = self.nli_model.predict([(combined_context, response)])
        entailment_score = float(scores[0][2])
        # Entailment probability checks alignment securely mapped
        hallucination_flag = entailment_score < 0.5
        return hallucination_flag, entailment_score

    def generate(self, query: str, top_k: int = 20, top_n: int = 5):
        from api.models import ClinicalQueryResponse, SourceDocument
        start_time = time.perf_counter()
        
        # Stage 1: Retrieve
        candidates = self.retriever.retrieve(query, top_k=top_k)
        
        # Stage 2: Rerank
        reranked = self.reranker.rerank(query, candidates, top_n=top_n)
        
        # Stage 3: Build prompt and generate
        prompt = self._build_prompt(query, reranked)
        output = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.1,
            stop=["<|end|>", "<|user|>"]
        )
        response_text = output["choices"][0]["text"].strip()
        
        # Stage 4: Hallucination check
        hall_flag, hall_score = self._check_hallucination(response_text, reranked)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        sources = [SourceDocument(
            doc_id=c.get("chunk_id", f"chunk_{i}"),
            text=c["text"],
            source_file=c.get("source", "unknown"),
            rerank_score=c.get("rerank_score", 0.0),
            chunk_id=c.get("chunk_id", f"chunk_{i}")
        ) for i, c in enumerate(reranked)]
        
        return ClinicalQueryResponse(
            query=query,
            response=response_text,
            retrieved_sources=sources,
            hallucination_flag=hall_flag,
            hallucination_score=round(hall_score, 4),
            latency_ms=round(latency_ms, 2)
        )
