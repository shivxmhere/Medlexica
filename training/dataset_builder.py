import json
import random
from collections import Counter
from datasets import load_dataset
import os

def main():
    print("Loading MedMCQA dataset (train split)...")
    dataset = load_dataset("medmcqa", split="train")

    valid_subjects = ["Medicine", "Pharmacology", "Pathology", "Surgery"]
    print(f"Filtering down to subjects: {valid_subjects}...")
    
    filtered_dataset = dataset.filter(lambda x: x["subject_name"] in valid_subjects)
    
    print("Shuffling and selecting 5000 samples for deterministic split...")
    # Deterministic split using random_state=42 (seed=42)
    shuffled_dataset = filtered_dataset.shuffle(seed=42)
    subset_dataset = shuffled_dataset.select(range(5000))

    instruction_template = "You are MedLexica, a clinical AI assistant trained for Indian medical practice. Answer the following medical question accurately. Always include relevant ICD-10 codes where applicable. End every response with: DISCLAIMER: This is AI-generated content for educational purposes only. Not a substitute for professional medical advice."

    stats_subjects = []
    total_input_len = 0
    total_output_len = 0
    processed_samples = []

    for item in subset_dataset:
        stats_subjects.append(item["subject_name"])
        
        # MedMCQA provides opa, opb, opc, opd and cop (correct option index)
        opts = [item.get("opa", ""), item.get("opb", ""), item.get("opc", ""), item.get("opd", "")]
        cop_val = item.get("cop")
        
        try:
            cop_idx = int(cop_val)
            opt_text = opts[cop_idx] if 0 <= cop_idx < len(opts) else ""
        except (ValueError, TypeError):
            opt_text = ""
            
        exp_text = item.get("exp", "")
        # Handle cases where exp_text is None
        if not exp_text:
            exp_text = ""
            
        output_str = f"{opt_text}\n\nExplanation: {exp_text}".strip()
        
        out_dict = {
            "instruction": instruction_template,
            "input": (item.get("question") or "").strip(),
            "output": output_str
        }
        
        total_input_len += len(out_dict["input"].split())
        total_output_len += len(out_dict["output"].split())
        processed_samples.append(out_dict)
        
    # Split deterministically
    train_split = processed_samples[:4700]
    eval_split = processed_samples[4700:5000]

    train_path = os.path.join("data", "processed", "train_instructions.jsonl")
    eval_path = os.path.join("data", "eval", "hold_out.jsonl")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    
    with open(train_path, "w", encoding="utf-8") as f:
        for entry in train_split:
            f.write(json.dumps(entry) + "\n")
            
    with open(eval_path, "w", encoding="utf-8") as f:
        for entry in eval_split:
            f.write(json.dumps(entry) + "\n")

    print("\n--- Processing complete ---")
    print(f"Total samples processed: {len(processed_samples)}")
    
    subject_counts = Counter(stats_subjects)
    print("Subject distribution:")
    for subj, count in subject_counts.items():
        print(f"  - {subj}: {count}")
        
    avg_input_len = total_input_len / len(processed_samples)
    avg_output_len = total_output_len / len(processed_samples)
    
    print(f"Average input token length: ~{avg_input_len:.2f}")
    print(f"Average output token length: ~{avg_output_len:.2f}")
    print(f"Files saved paths:")
    print(f"  Train data -> {train_path}")
    print(f"  Hold-out eval -> {eval_path}")
    print("CRITICAL: hold_out.jsonl must NEVER be used during training.")

if __name__ == "__main__":
    main()
