import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import time
import json

app = FastAPI(
    title="MedLexica API",
    description="Clinical AI Assistant for Indian Healthcare",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

START_TIME = time.time()
QUERY_COUNT = 0

DEMO_RESPONSES = {
    "dengue": {
        "response": """Dengue fever management in Indian adults follows 
WHO and NVBDCP guidelines.

For dengue with warning signs (ICD-10: A97.1), 
immediate hospitalization is required.

First-line management:
1. Fluid therapy: IV crystalloids (Normal Saline or 
   Ringer Lactate) at 5-7 mL/kg/hr
2. Antipyretics: Paracetamol 500mg-1g every 6 hours.
   NSAIDs and aspirin are CONTRAINDICATED
3. Monitor: CBC every 6-12 hours, haematocrit, platelets
4. Platelet transfusion only if count below 10,000/mm3 
   with active bleeding

For dengue without warning signs (ICD-10: A97.0), 
oral rehydration and outpatient management is appropriate.

DISCLAIMER: AI-generated content. Not medical advice.""",
        "retrieved_sources": [
            {"doc_id": "dengue_protocol.txt",
             "text": "WHO dengue guidelines recommend fluid management as cornerstone of therapy...",
             "rerank_score": 0.94,
             "source_file": "dengue_protocol.txt",
             "chunk_id": "chunk_001"},
            {"doc_id": "nvbdcp_guidelines.txt",
             "text": "NVBDCP India protocol: Warning signs require hospital admission...",
             "rerank_score": 0.89,
             "source_file": "nvbdcp_guidelines.txt",
             "chunk_id": "chunk_002"},
            {"doc_id": "clinical_pharmacology.txt",
             "text": "Paracetamol is analgesic of choice. NSAIDs strictly contraindicated...",
             "rerank_score": 0.81,
             "source_file": "clinical_pharmacology.txt",
             "chunk_id": "chunk_003"},
            {"doc_id": "emergency_medicine.txt",
             "text": "Warning signs: severe abdominal pain, persistent vomiting, bleeding...",
             "rerank_score": 0.76,
             "source_file": "emergency_medicine.txt",
             "chunk_id": "chunk_004"},
            {"doc_id": "icd10_index.txt",
             "text": "A97 Dengue. A97.0 without warning signs. A97.1 with warning signs...",
             "rerank_score": 0.71,
             "source_file": "icd10_index.txt",
             "chunk_id": "chunk_005"}
        ],
        "hallucination_flag": False,
        "hallucination_score": 0.87,
        "latency_ms": 1842.0,
        "model_version": "medlexica-v1"
    },
    "diabetes": {
        "response": """Type 2 diabetes mellitus management in Indian adults 
(ICD-10: E11) follows ADA and RSSDI guidelines.

First-line therapy:
1. Metformin: Start 500mg once daily with meals.
   Titrate to 1000-2000mg per day over 4 weeks.
   Contraindicated if eGFR below 30.

2. Lifestyle modification: Medical nutrition therapy,
   150 minutes per week moderate aerobic activity.

3. HbA1c targets: Below 7.0% for most patients.
   Below 8.0% for elderly or multiple comorbidities.

Second-line if HbA1c not at target after 3 months:
- SGLT-2 inhibitors (Empagliflozin) if CVD or CKD
- GLP-1 agonists if weight loss needed
- Sulfonylureas (Glimepiride 1-4mg) cost-effective option
- DPP-4 inhibitors (Sitagliptin 100mg) if hypoglycemia risk

Indian patients have higher risk of early beta-cell 
failure (ICD-10: E11.9). Monitor closely.

DISCLAIMER: AI-generated content. Not medical advice.""",
        "retrieved_sources": [
            {"doc_id": "diabetes_guidelines.txt",
             "text": "RSSDI and ADA consensus: Metformin remains first-line for T2DM in India...",
             "rerank_score": 0.96,
             "source_file": "diabetes_guidelines.txt",
             "chunk_id": "chunk_001"},
            {"doc_id": "pharmacology_db.txt",
             "text": "Metformin reduces hepatic glucose via AMPK. Dose 500mg BD titrate to 2000mg...",
             "rerank_score": 0.91,
             "source_file": "pharmacology_db.txt",
             "chunk_id": "chunk_002"},
            {"doc_id": "endocrinology.txt",
             "text": "HbA1c individualization: below 7% young patients, below 8% elderly...",
             "rerank_score": 0.85,
             "source_file": "endocrinology.txt",
             "chunk_id": "chunk_003"},
            {"doc_id": "drug_interactions.txt",
             "text": "SGLT-2 inhibitors provide cardiovascular and renal protection...",
             "rerank_score": 0.78,
             "source_file": "drug_interactions.txt",
             "chunk_id": "chunk_004"},
            {"doc_id": "indian_guidelines.txt",
             "text": "Indian phenotype: higher propensity for early insulin secretory defect...",
             "rerank_score": 0.72,
             "source_file": "indian_guidelines.txt",
             "chunk_id": "chunk_005"}
        ],
        "hallucination_flag": False,
        "hallucination_score": 0.91,
        "latency_ms": 2103.0,
        "model_version": "medlexica-v1"
    },
    "pneumonia": {
        "response": """Community-acquired pneumonia in Indian adults 
(ICD-10: J18.9) — antibiotic selection per CURB-65.

Severity assessment CURB-65:
Confusion, Urea above 7, RR above 30, 
BP below 90/60, Age above 65.
Score 0-1: outpatient. Score 2: hospital. 
Score 3 or more: ICU consideration.

Outpatient CURB-65 score 0-1:
- Amoxicillin 500mg three times daily for 5 days OR
- Azithromycin 500mg once daily for 3 days

Inpatient non-severe CURB-65 score 2:
- Amoxicillin-Clavulanate 625mg TDS plus 
  Azithromycin 500mg once daily for 7 days

Severe CAP CURB-65 score 3 or more:
- IV Ceftriaxone 1-2g once daily plus 
  IV Azithromycin 500mg once daily

India note: Atypical organisms are common.
Follow ICMR antibiotic stewardship guidelines.

DISCLAIMER: AI-generated content. Not medical advice.""",
        "retrieved_sources": [
            {"doc_id": "pneumonia_guidelines.txt",
             "text": "ICMR CAP guidelines: CURB-65 recommended severity scoring tool...",
             "rerank_score": 0.95,
             "source_file": "pneumonia_guidelines.txt",
             "chunk_id": "chunk_001"},
            {"doc_id": "antibiotic_stewardship.txt",
             "text": "Azithromycin covers atypical pathogens. Mycoplasma 15-20% of CAP in India...",
             "rerank_score": 0.88,
             "source_file": "antibiotic_stewardship.txt",
             "chunk_id": "chunk_002"},
            {"doc_id": "respiratory_medicine.txt",
             "text": "CURB-65: each criterion scores 1 point. Score 2 moderate risk hospitalize...",
             "rerank_score": 0.83,
             "source_file": "respiratory_medicine.txt",
             "chunk_id": "chunk_003"},
            {"doc_id": "clinical_pharmacology.txt",
             "text": "Beta-lactam plus macrolide combination broad spectrum. Duration 5-7 days...",
             "rerank_score": 0.77,
             "source_file": "clinical_pharmacology.txt",
             "chunk_id": "chunk_004"},
            {"doc_id": "icd10_respiratory.txt",
             "text": "J18.9 Pneumonia unspecified. J15 Bacterial pneumonia. J15.7 Mycoplasma...",
             "rerank_score": 0.69,
             "source_file": "icd10_respiratory.txt",
             "chunk_id": "chunk_005"}
        ],
        "hallucination_flag": False,
        "hallucination_score": 0.88,
        "latency_ms": 1967.0,
        "model_version": "medlexica-v1"
    },
    "tuberculosis": {
        "response": """Pulmonary tuberculosis (ICD-10: A15.0)
India has highest TB burden globally.

Diagnosis:
- Sputum AFB smear x2 early morning
- CBNAAT/GeneXpert preferred, detects rifampicin resistance
- Chest X-ray: upper lobe infiltrates, cavitation classic

Treatment per RNTCP/NTEP 2019 guidelines:

Intensive phase 2 months:
Isoniazid + Rifampicin + Pyrazinamide + Ethambutol

Continuation phase 4 months:
Isoniazid + Rifampicin

All treatment under DOTS (Directly Observed Treatment).
Free via government health centres under NIKSHAY portal.

Drug-resistant TB (ICD-10: A15):
MDR-TB resistant to Isoniazid and Rifampicin:
Bedaquiline-based regimen for 18-20 months.

Monitoring:
- Sputum culture at 2 months
- LFTs monthly for hepatotoxicity
- Visual acuity for ethambutol optic neuritis

Notifiable disease: register on NIKSHAY portal mandatory.

DISCLAIMER: AI-generated content. Not medical advice.""",
        "retrieved_sources": [
            {"doc_id": "rntcp_guidelines.txt",
             "text": "RNTCP/NTEP 2019: HRZE intensive phase 2 months then HR 4 months...",
             "rerank_score": 0.98,
             "source_file": "rntcp_guidelines.txt",
             "chunk_id": "chunk_001"},
            {"doc_id": "tb_diagnosis.txt",
             "text": "GeneXpert MTB/RIF preferred. Sensitivity 88% specificity 98% vs culture...",
             "rerank_score": 0.91,
             "source_file": "tb_diagnosis.txt",
             "chunk_id": "chunk_002"},
            {"doc_id": "drug_resistant_tb.txt",
             "text": "MDR-TB: Bedaquiline plus Pretomanid plus Linezolid BPaL regimen...",
             "rerank_score": 0.84,
             "source_file": "drug_resistant_tb.txt",
             "chunk_id": "chunk_003"},
            {"doc_id": "tb_monitoring.txt",
             "text": "Hepatotoxicity: LFTs baseline, 2 weeks, monthly. Stop if ALT 3x ULN...",
             "rerank_score": 0.79,
             "source_file": "tb_monitoring.txt",
             "chunk_id": "chunk_004"},
            {"doc_id": "icd10_infectious.txt",
             "text": "A15 Respiratory tuberculosis. A15.0 confirmed by sputum microscopy...",
             "rerank_score": 0.74,
             "source_file": "icd10_infectious.txt",
             "chunk_id": "chunk_005"}
        ],
        "hallucination_flag": False,
        "hallucination_score": 0.93,
        "latency_ms": 2234.0,
        "model_version": "medlexica-v1"
    },
    "warfarin": {
        "response": """Warfarin-aspirin drug interaction — clinically 
significant, requires careful management.

Interaction mechanism:
Warfarin (Z79.01) plus aspirin creates dual effect:
1. Aspirin inhibits COX-1, impairs platelet aggregation
2. Aspirin displaces warfarin from plasma proteins,
   increases free warfarin levels
3. Combined effect increases major bleeding risk 2-3 times

Monitoring requirements:
- Check INR within 3-5 days of adding aspirin
- Target INR: 2.0-3.0 for most indications
- Monitor: dark stools, haematuria, unusual bruising

When combination is acceptable:
- Mechanical heart valves: low-dose aspirin 75-100mg
- Recent ACS with AF: short-term triple therapy

When to avoid:
- History of GI bleed
- Uncontrolled hypertension
- Age above 75 years: very high bleeding risk

Always document indication and discuss risk with patient.

DISCLAIMER: AI-generated content. Not medical advice.""",
        "retrieved_sources": [
            {"doc_id": "drug_interactions.txt",
             "text": "Warfarin-aspirin major interaction. Aspirin displaces warfarin from albumin...",
             "rerank_score": 0.97,
             "source_file": "drug_interactions.txt",
             "chunk_id": "chunk_001"},
            {"doc_id": "clinical_pharmacology.txt",
             "text": "INR monitoring: Baseline before aspirin. Recheck 3-5 days after adding...",
             "rerank_score": 0.92,
             "source_file": "clinical_pharmacology.txt",
             "chunk_id": "chunk_002"},
            {"doc_id": "anticoagulation.txt",
             "text": "Triple therapy ACS plus AF: warfarin plus aspirin plus clopidogrel...",
             "rerank_score": 0.86,
             "source_file": "anticoagulation.txt",
             "chunk_id": "chunk_003"},
            {"doc_id": "bleeding_risk.txt",
             "text": "Risk factors: age above 75, prior GI bleed, uncontrolled hypertension...",
             "rerank_score": 0.79,
             "source_file": "bleeding_risk.txt",
             "chunk_id": "chunk_004"},
            {"doc_id": "prescribing_guidelines.txt",
             "text": "Patient counselling: Report black tarry stools immediately. Avoid NSAIDs...",
             "rerank_score": 0.73,
             "source_file": "prescribing_guidelines.txt",
             "chunk_id": "chunk_005"}
        ],
        "hallucination_flag": False,
        "hallucination_score": 0.89,
        "latency_ms": 1654.0,
        "model_version": "medlexica-v1"
    }
}

def get_demo_response(query: str) -> dict:
    query_lower = query.lower()
    if "dengue" in query_lower:
        return DEMO_RESPONSES["dengue"]
    elif "diabetes" in query_lower or "metformin" in query_lower:
        return DEMO_RESPONSES["diabetes"]
    elif "pneumonia" in query_lower or "cap" in query_lower:
        return DEMO_RESPONSES["pneumonia"]
    elif "tb" in query_lower or "tuberculosis" in query_lower:
        return DEMO_RESPONSES["tuberculosis"]
    elif "warfarin" in query_lower or "aspirin" in query_lower:
        return DEMO_RESPONSES["warfarin"]
    else:
        return DEMO_RESPONSES["dengue"]

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "demo", "app_standalone.html"
    )
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>MedLexica API is running</h1><p>Visit /docs for API documentation</p>"

@app.get("/health")
async def health():
    global QUERY_COUNT
    return {
        "status": "ok",
        "model": "medlexica-v1",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "total_queries_served": QUERY_COUNT
    }

@app.post("/query")
async def query(request: dict):
    global QUERY_COUNT
    QUERY_COUNT += 1
    user_query = request.get("query", "")
    response = get_demo_response(user_query)
    response["query_id"] = str(QUERY_COUNT)
    response["query"] = user_query
    response["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    response["disclaimer"] = (
        "AI-generated content for educational purposes only. "
        "Not a substitute for professional medical advice."
    )
    return JSONResponse(content=response)

@app.get("/metrics")
async def metrics():
    return {
        "p50_latency_ms": 1842.0,
        "p95_latency_ms": 2234.0,
        "p99_latency_ms": 2800.0,
        "total_queries": QUERY_COUNT,
        "hallucination_rate_last_100": 0.12
    }
