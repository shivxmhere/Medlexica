import gradio as gr
import httpx
import json

API_URL = "http://localhost:8000"

def query_medlexica(clinical_query: str, top_k: int, top_n: int):
    if len(clinical_query.strip()) < 10:
        return (
            "Please enter a more detailed query (min 10 chars)",
            {},
            0,
            False
        )
    
    try:
        response = httpx.post(
            f"{API_URL}/query",
            json={
                "query": clinical_query,
                "top_k_retrieval": top_k,
                "top_n_rerank": top_n,
                "stream": False
            },
            timeout=60.0
        )
        data = response.json()
        
        sources_display = {
            f"Source {i+1} ({s['source_file']})": {
                "text": s["text"][:300] + "...",
                "rerank_score": s["rerank_score"]
            }
            for i, s in enumerate(data["retrieved_sources"])
        }
        
        return (
            data["response"] + "\n\n" + data["disclaimer"],
            sources_display,
            data["latency_ms"],
            data["hallucination_flag"]
        )
    except Exception as e:
        return (f"Error: {str(e)}", {}, 0, False)

def format_hallucination_warning(flag: bool) -> str:
    if flag:
        return (
            "⚠️ HALLUCINATION RISK DETECTED — Response may not be "
            "fully grounded in retrieved sources. "
            "Verify with a qualified medical professional."
        )
    return "✓ Response grounded in retrieved medical literature."


with gr.Blocks(
    title="MedLexica — Clinical AI Assistant",
    theme=gr.themes.Soft()
) as demo:
    gr.Markdown("""
    # MedLexica — Clinical AI Assistant
    **Fine-tuned Phi-3-mini · Hybrid RAG (FAISS+BM25) · Cross-Encoder Reranking · 4-bit Quantized**
    
    *For educational purposes only. Not a substitute for professional medical advice.*
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Clinical Query",
                placeholder="e.g., First-line treatment for dengue fever in a 28-year-old adult in India?",
                lines=3
            )
            with gr.Row():
                top_k_slider = gr.Slider(
                    5, 50, value=20, step=5,
                    label="Retrieval candidates (top-K)"
                )
                top_n_slider = gr.Slider(
                    1, 10, value=5, step=1,
                    label="Reranked passages (top-N)"
                )
            submit_btn = gr.Button("Submit Query", variant="primary")
            
    with gr.Row():
        with gr.Column():
            response_output = gr.Textbox(
                label="MedLexica Response", lines=8, interactive=False
            )
            hallucination_output = gr.Textbox(
                label="Grounding Check", interactive=False
            )
        with gr.Column():
            sources_output = gr.JSON(label="Retrieved Sources & Scores")
            latency_output = gr.Number(
                label="Latency (ms)", precision=2
            )
            
    submit_btn.click(
        fn=lambda q, k, n: (
            *query_medlexica(q, k, n)[:3],
            format_hallucination_warning(query_medlexica(q, k, n)[3])
        ),
        inputs=[query_input, top_k_slider, top_n_slider],
        outputs=[response_output, sources_output, latency_output, hallucination_output]
    )
    
    gr.Examples(
        examples=[
            ["What is the standard dose of metformin for Type 2 diabetes in a 50-year-old Indian patient?"],
            ["Differential diagnosis for fever with rash in a 25-year-old presenting to a Delhi hospital"],
            ["First-line antibiotic for community-acquired pneumonia in India according to current guidelines"],
            ["ICD-10 code for dengue fever with warning signs"],
            ["Drug interactions between warfarin and aspirin — what monitoring is required?"]
        ],
        inputs=query_input
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
