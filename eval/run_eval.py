import json
import os
import re
from datetime import datetime
import wandb
import pandas as pd
from transformers import pipeline
import torch

from metrics import (
    compute_rouge,
    compute_bertscore,
    compute_hallucination_rate,
    compute_icd_accuracy,
    compute_latency
)

# Optional G-Eval, skipping API call strictly if no key is provided here, 
# you can weave compute_geval_score dynamically when ready over subset.

def main():
    # STEP 1 - Load hold-out set
    holdout_path = "data/eval/hold_out.jsonl"
    queries = []
    references = []
    gold_icd_codes = []
    
    icd_pattern = r'\b[A-Z]\d{2}\.?\d{0,2}\b'
    
    with open(holdout_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query = data.get('input', '')
            queries.append(query)
            ref = data.get('output', '')
            references.append(ref)
            
            # dynamically extract plausible gold ICD codes from the original correct labels
            extracted = re.findall(icd_pattern, ref)
            gold_icd_codes.append(extracted[0] if extracted else "")
            
    print(f"Loaded {len(queries)} evaluation samples")

    # STEP 2 - Define both model inference functions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 2a. Load microsoft/Phi-3-mini-4k-instruct base (no fine-tuning)
    try:
        baseline_pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", device=0 if device=="cuda" else -1, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    except Exception as e:
        print(f"Could not load baseline model immediately into VRAM: {e}")
        baseline_pipe = None

    def baseline_inference(query):
        prompt = f"<|user|>\n{query}<|end|>\n<|assistant|>\n"
        if not baseline_pipe:
            return "Failed load"
        res = baseline_pipe(prompt, max_new_tokens=150, return_full_text=False)
        return res[0]['generated_text'].strip()
        
    # 2b. Load from checkpoints/medlexica-v1-merged
    try:
        medlexica_pipe = pipeline("text-generation", model="./checkpoints/medlexica-v1-merged", device=0 if device=="cuda" else -1, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    except Exception as e:
        print(f"Could not load finetuned model (checkpoint may not exist yet): {e}")
        medlexica_pipe = None
        
    def medlexica_inference(query):
        prompt = f"<|user|>\n{query}<|end|>\n<|assistant|>\n"
        if not medlexica_pipe:
            return "Failed load - model needs to be trained and merged first"
        res = medlexica_pipe(prompt, max_new_tokens=150, return_full_text=False)
        return res[0]['generated_text'].strip()

    # STEP 3 - Generate predictions from both models
    print("Generating baseline predictions...")
    baseline_predictions = []
    medlexica_predictions = []
    
    for i, q in enumerate(queries):
        baseline_predictions.append(baseline_inference(q))
        medlexica_predictions.append(medlexica_inference(q))
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(queries)} samples")
            
    # STEP 4 - Run all metrics on BOTH models
    print("Computing metrics for baseline...")
    baseline_scores = {
        "rouge": compute_rouge(baseline_predictions, references),
        "bertscore": compute_bertscore(baseline_predictions, references),
        "hallucination": compute_hallucination_rate(baseline_predictions, queries),
        "icd_accuracy": compute_icd_accuracy(baseline_predictions, gold_icd_codes),
        "latency": compute_latency(baseline_inference, queries[:20])
    }
    
    print("Computing metrics for MedLexica...")
    medlexica_scores = {
        "rouge": compute_rouge(medlexica_predictions, references),
        "bertscore": compute_bertscore(medlexica_predictions, references),
        "hallucination": compute_hallucination_rate(medlexica_predictions, queries),
        "icd_accuracy": compute_icd_accuracy(medlexica_predictions, gold_icd_codes),
        "latency": compute_latency(medlexica_inference, queries[:20])
    }
    
    # STEP 5 - Compute deltas
    def calc_delta(baseline_val, medlexica_val, invert=False):
        if baseline_val == 0:
            return "+0.0%" if medlexica_val == 0 else "+100.0%"
        diff = ((medlexica_val - baseline_val) / baseline_val) * 100
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.1f}%"
        
    deltas = {
        "rougeL": calc_delta(baseline_scores["rouge"]["rougeL"], medlexica_scores["rouge"]["rougeL"]),
        "bertscore_f1": calc_delta(baseline_scores["bertscore"]["f1"], medlexica_scores["bertscore"]["f1"]),
        "hallucination_rate": calc_delta(baseline_scores["hallucination"]["hallucination_rate"], medlexica_scores["hallucination"]["hallucination_rate"]),
        "icd_accuracy": calc_delta(baseline_scores["icd_accuracy"]["icd10_accuracy"], medlexica_scores["icd_accuracy"]["icd10_accuracy"]),
        "p95_latency": calc_delta(baseline_scores["latency"]["p95_ms"], medlexica_scores["latency"]["p95_ms"])
    }
    
    # Determine 'Better?' flag for WandB table
    better_flags = {
        "rougeL": "Yes" if medlexica_scores["rouge"]["rougeL"] > baseline_scores["rouge"]["rougeL"] else "No",
        "bertscore_f1": "Yes" if medlexica_scores["bertscore"]["f1"] > baseline_scores["bertscore"]["f1"] else "No",
        "hallucination_rate": "Yes" if medlexica_scores["hallucination"]["hallucination_rate"] < baseline_scores["hallucination"]["hallucination_rate"] else "No",
        "icd_accuracy": "Yes" if medlexica_scores["icd_accuracy"]["icd10_accuracy"] > baseline_scores["icd_accuracy"]["icd10_accuracy"] else "No",
        "p95_latency": "Yes" if medlexica_scores["latency"]["p95_ms"] < baseline_scores["latency"]["p95_ms"] else "No"
    }

    # STEP 6 - Save scorecard.json
    scorecard = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "baseline_model": "microsoft/Phi-3-mini-4k-instruct",
        "finetuned_model": "medlexica-v1",
        "eval_samples": len(queries),
        "baseline": baseline_scores,
        "medlexica": medlexica_scores,
        "delta": deltas
    }
    
    with open("eval/scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2)
        
    # STEP 7 - Log to Weights & Biases
    wandb.init(project="medlexica-hackathon", name="eval-baseline-vs-finetuned")
    
    # Create and log table
    table = wandb.Table(columns=["Metric", "Base Phi-3", "MedLexica", "Delta", "Better?"])
    table.add_data("ROUGE-L", baseline_scores["rouge"]["rougeL"], medlexica_scores["rouge"]["rougeL"], deltas["rougeL"], better_flags["rougeL"])
    table.add_data("BERTScore F1", baseline_scores["bertscore"]["f1"], medlexica_scores["bertscore"]["f1"], deltas["bertscore_f1"], better_flags["bertscore_f1"])
    table.add_data("Hallucination Rate", baseline_scores["hallucination"]["hallucination_rate"], medlexica_scores["hallucination"]["hallucination_rate"], deltas["hallucination_rate"], better_flags["hallucination_rate"])
    table.add_data("ICD-10 Accuracy", baseline_scores["icd_accuracy"]["icd10_accuracy"], medlexica_scores["icd_accuracy"]["icd10_accuracy"], deltas["icd_accuracy"], better_flags["icd_accuracy"])
    table.add_data("P95 Latency (ms)", baseline_scores["latency"]["p95_ms"], medlexica_scores["latency"]["p95_ms"], deltas["p95_latency"], better_flags["p95_latency"])
    
    wandb.log({"eval_results_table": table})
    wandb.log({
        "eval/rouge_l": medlexica_scores["rouge"]["rougeL"],
        "eval/bertscore_f1": medlexica_scores["bertscore"]["f1"],
        "eval/hallucination_rate": medlexica_scores["hallucination"]["hallucination_rate"],
        "eval/icd_accuracy": medlexica_scores["icd_accuracy"]["icd10_accuracy"]
    })
    
    # STEP 8 - Print formatted terminal report
    # Custom format strings for centering padding
    def fmt(val):
        return f"{val:<12.4f}" if isinstance(val, float) else f"{val:<12}"
        
    print("╔══════════════════════╦══════════════╦══════════════╦══════════╗")
    print("║ Metric               ║ Base Phi-3   ║ MedLexica    ║ Delta    ║")
    print("╠══════════════════════╬══════════════╬══════════════╬══════════╣")
    print(f"║ ROUGE-L              ║ {fmt(baseline_scores['rouge']['rougeL'])} ║ {fmt(medlexica_scores['rouge']['rougeL'])} ║ {deltas['rougeL']:<8} ║")
    print(f"║ BERTScore F1         ║ {fmt(baseline_scores['bertscore']['f1'])} ║ {fmt(medlexica_scores['bertscore']['f1'])} ║ {deltas['bertscore_f1']:<8} ║")
    print(f"║ Hallucination Rate   ║ {fmt(baseline_scores['hallucination']['hallucination_rate'])} ║ {fmt(medlexica_scores['hallucination']['hallucination_rate'])} ║ {deltas['hallucination_rate']:<8} ║")
    print(f"║ ICD-10 Accuracy      ║ {fmt(baseline_scores['icd_accuracy']['icd10_accuracy'])} ║ {fmt(medlexica_scores['icd_accuracy']['icd10_accuracy'])} ║ {deltas['icd_accuracy']:<8} ║")
    print(f"║ P95 Latency (ms)     ║ {baseline_scores['latency']['p95_ms']:<12.2f} ║ {medlexica_scores['latency']['p95_ms']:<12.2f} ║ {deltas['p95_latency']:<8} ║")
    print("╚══════════════════════╩══════════════╩══════════════╩══════════╝")
    
    # STEP 9 - Save detailed per-sample CSV
    detailed_df = pd.DataFrame({
        "query_id": range(len(queries)),
        "query": queries,
        "baseline_pred": baseline_predictions,
        "medlexica_pred": medlexica_predictions,
        "baseline_bertscore": [None] * len(queries), # BERTScore Python API provides arrays naturally, but this requires refactoring to merge into CSV cleanly per-sample
        "medlexica_bertscore": [None] * len(queries), 
        "baseline_hallucinated": [int(s < 0.5) for s in baseline_scores["hallucination"]["per_sample_entailment_scores"]],
        "medlexica_hallucinated": [int(s < 0.5) for s in medlexica_scores["hallucination"]["per_sample_entailment_scores"]],
        "baseline_icd_correct": [int(bool(gold) and any(c.startswith(gold[:3]) for c in re.findall(icd_pattern, p))) for p, gold in zip(baseline_predictions, gold_icd_codes)],
        "medlexica_icd_correct": [int(bool(gold) and any(c.startswith(gold[:3]) for c in re.findall(icd_pattern, p))) for p, gold in zip(medlexica_predictions, gold_icd_codes)]
    })
    
    detailed_df.to_csv("eval/detailed_results.csv", index=False)
    print("Saved eval/detailed_results.csv")

if __name__ == "__main__":
    main()
