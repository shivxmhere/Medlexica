import json
import logging
import spacy
import os
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Set up Python logging to BOTH console AND data/preprocess.log
os.makedirs("data", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/preprocess.log"),
        logging.StreamHandler()
    ]
)

def token_packer(samples, max_length=2048):
    """
    Groups multiple short samples into single max-token window (e.g., 2048-token windows).
    Separates samples with "<|sep|>" token.
    Returns packed samples list.
    """
    # Grouping logic to efficiently utilize fixed-length attention contexts
    packed_windows = []
    current_window = []
    current_length = 0
    separator = "<|sep|>"
    sep_length = len(separator.split())

    original_samples = len(samples)

    for sample in samples:
        # Simplistic split representation of token count
        sample_len = len(sample.split())
        
        if current_length + sample_len + sep_length > max_length and current_window:
            # Reached max length, push current window
            packed_windows.append(f" {separator} ".join(current_window))
            current_window = [sample]
            current_length = sample_len
        else:
            current_window.append(sample)
            current_length += sample_len + (sep_length if current_window else 0)
            
    # Add any remaining string arrays in the window context
    if current_window:
        packed_windows.append(f" {separator} ".join(current_window))
        
    # Log how many windows were created vs original samples
    logging.info(f"Packed {original_samples} original samples into {len(packed_windows)} packed windows.")
    return packed_windows

def main():
    logging.info("Initializing NLP models for data preprocessing...")
    # PHI scrubbing engine setup
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    
    # Load spaCy to validate sentences
    nlp = spacy.load("en_core_web_sm")

    input_file = "data/processed/train_instructions.jsonl"
    output_file = "data/processed/train_clean.jsonl"

    logging.info(f"Loading generated dataset instructions from {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f]

    valid_json_samples = []
    logging.info("Beginning filtering and PHI scrubbing pass...")
    
    for record in raw_samples:
        clean_input = record["input"]
        clean_output = record["output"]

        # PHI scrubbing required for HIPAA-equivalent compliance in Indian clinical data contexts.
        # Targets: PERSON, DATE_TIME, LOCATION -> replaces with REDACTED tags.
        if clean_input:
            in_results = analyzer.analyze(text=clean_input, entities=["PERSON", "DATE_TIME", "LOCATION"], language='en')
            if in_results:
                redacted_in = anonymizer.anonymize(
                    text=clean_input,
                    analyzer_results=in_results,
                    operators={
                        "PERSON": {"type": "replace", "new_value": "[REDACTED_NAME]"},
                        "DATE_TIME": {"type": "replace", "new_value": "[REDACTED_DATE]"},
                        "LOCATION": {"type": "replace", "new_value": "[REDACTED_LOCATION]"}
                    }
                )
                clean_input = redacted_in.text

        if clean_output:
            out_results = analyzer.analyze(text=clean_output, entities=["PERSON", "DATE_TIME", "LOCATION"], language='en')
            if out_results:
                redacted_out = anonymizer.anonymize(
                    text=clean_output,
                    analyzer_results=out_results,
                    operators={
                        "PERSON": {"type": "replace", "new_value": "[REDACTED_NAME]"},
                        "DATE_TIME": {"type": "replace", "new_value": "[REDACTED_DATE]"},
                        "LOCATION": {"type": "replace", "new_value": "[REDACTED_LOCATION]"}
                    }
                )
                clean_output = redacted_out.text
        
        # Combine blocks to evaluate overall grammatical formations and token lengths
        combined_text = f"{record['instruction']} {clean_input} {clean_output}"
        
        doc = nlp(combined_text)
        
        # Skip any sample where spaCy detects 0 sentences or shorter than 20 tokens
        sent_count = len(list(doc.sents))
        if sent_count == 0 or len(doc) < 20:
            continue
            
        valid_json_samples.append({
            "instruction": record["instruction"],
            "input": clean_input,
            "output": clean_output
        })

    logging.info(f"Finished cleaning filtering. Reduced from {len(raw_samples)} to {len(valid_json_samples)} valid samples.")
    
    # Utilizing token packing to assemble long context causal representations
    string_samples = [json.dumps(s) for s in valid_json_samples]
    packed = token_packer(string_samples, max_length=2048)

    logging.info(f"Saving fully processed data to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for p in packed:
            # Represent packed multi-text payload as single JSON structure for trainer
            f.write(json.dumps({"text": p}) + "\n")

    logging.info("Data preprocessing pipeline complete!")

if __name__ == "__main__":
    main()
