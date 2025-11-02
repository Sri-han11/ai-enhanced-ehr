"""
Module 3 — Clinical note generation (SOAP) + medical summarization + ICD-10 suggestions

Requirements:
 - Files in cleaned datasets folder:
    - doctor_prescription_cleaned.csv
    - icd10_cleaned_data.csv
    - lab_details_cleaned.csv
    - patient_details_cleaned.csv
    - patient_records_cleaned.csv
 - pip install transformers sentence-transformers torch pandas tqdm
"""

import os, re, json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# -------------------------
# Configuration
# -------------------------
DATA_DIR = Path(r"C:\Users\Srimathi\ai-ehr-project\cleaned datasets")
OUT_DIR = DATA_DIR / "module3_outputs"
NOTES_DIR = OUT_DIR / "notes"
OUT_CSV = OUT_DIR / "enhanced_ehr_module3.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NOTES_DIR.mkdir(parents=True, exist_ok=True)

# Models (public)
GEN_MODEL = "google/flan-t5-large"        # primary generator (fall back to flan-t5-base if needed)
MED_SUMMARIZER = "microsoft/BioGPT-Large" # medical summarizer (bio GPT)
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # embeddings
ZS_MODEL = "facebook/bart-large-mnli"     # zero-shot re-ranker for ICD candidate scoring

# runtime params
GEN_MAX_NEW_TOKENS = 384
SUM_MAX_NEW_TOKENS = 256
TOP_K_CANDIDATES = 50
TOP_K_RETURN = 5

# device (transformers pipeline device arg)
DEVICE = 0 if torch.cuda.is_available() else -1
print(f"Device set to: {'cuda' if DEVICE == 0 else 'cpu'}")

# -------------------------
# Utilities
# -------------------------
def read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str).fillna("")
    print(f"[WARN] {path} not found — empty dataframe created.")
    return pd.DataFrame()

def normalize_date(x: str) -> str:
    try:
        return pd.to_datetime(str(x), errors="coerce").strftime("%Y-%m-%d")
    except Exception:
        return ""

def make_encounter_id(patient_id: str, idx: int) -> str:
    # patient_id might be PAT001 or numeric; make a tidy synthetic id
    pid = re.sub(r"\D", "", str(patient_id)) or str(patient_id)
    return f"PAT{int(pid):03d}_{idx:03d}"

def dedupe_consecutive_sentences(text: str) -> str:
    if not text:
        return text
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    out, prev = [], None
    for s in parts:
        if s.strip() == prev:
            continue
        out.append(s)
        prev = s.strip()
    return " ".join(out)

def clean_icd_label(lbl: str) -> str:
    return re.sub(r'[\{\}\[\]\'"]', '', str(lbl)).strip()

# -------------------------
# Load CSVs
# -------------------------
presc_df = read_csv(DATA_DIR / "doctor_prescription_cleaned.csv")
icd_df   = read_csv(DATA_DIR / "icd10_cleaned_data.csv")
labs_df  = read_csv(DATA_DIR / "lab_details_cleaned.csv")
pat_df   = read_csv(DATA_DIR / "patient_details_cleaned.csv")
recs_df  = read_csv(DATA_DIR / "patient_records_cleaned.csv")

# lowercase column names for resilience
def lower_cols(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

presc_df, icd_df, labs_df, pat_df, recs_df = map(lower_cols, [presc_df, icd_df, labs_df, pat_df, recs_df])

print(f"Loaded: records={len(recs_df)}, patients={len(pat_df)}, prescriptions={len(presc_df)}, labs={len(labs_df)}, icd={len(icd_df)}")

# -------------------------
# Build encounter-level dataframe
# -------------------------
# normalize dates
for df in (recs_df, presc_df, labs_df):
    if "visit_date" in df.columns:
        df["visit_date"] = df["visit_date"].apply(lambda x: normalize_date(x) or "")
    else:
        df["visit_date"] = ""

# base on patient_records if present else prescriptions/labs
if not recs_df.empty:
    base = recs_df.copy()
elif not presc_df.empty:
    base = presc_df.copy()
elif not labs_df.empty:
    base = labs_df.copy()
else:
    raise SystemExit("No encounter-level data available in patient_records/prescriptions/labs.")

# ensure patient_id exists
if "patient_id" not in base.columns:
    raise SystemExit("No patient_id column in encounter base data")

base = base.reset_index(drop=True)
base["_seq"] = base.groupby(["patient_id", "visit_date"]).cumcount() + 1
base["encounter_id"] = base.apply(lambda r: make_encounter_id(r["patient_id"], int(r["_seq"])), axis=1)

# aggregate prescriptions & labs per patient+date
if not presc_df.empty:
    presc_agg = presc_df.groupby(["patient_id", "visit_date"]).agg({
        "prescription_id": lambda s: ";".join(s.dropna().unique()),
        "doc_id": lambda s: ";".join(s.dropna().unique()),
        "transcription": lambda s: " || ".join(s.dropna().astype(str)),
        "keywords": lambda s: " || ".join(s.dropna().astype(str))
    }).reset_index()
else:
    presc_agg = pd.DataFrame(columns=["patient_id", "visit_date", "transcription", "keywords"])

if not labs_df.empty:
    labs_agg = labs_df.groupby(["patient_id", "visit_date"]).agg({
        "lab_id": lambda s: ";".join(s.dropna().unique()),
        "sample_name": lambda s: " || ".join(s.dropna().astype(str))
    }).reset_index()
else:
    labs_agg = pd.DataFrame(columns=["patient_id", "visit_date", "sample_name"])

# merge demographics, prescriptions, labs
merged = base.merge(pat_df, on="patient_id", how="left")
merged = merged.merge(presc_agg, on=["patient_id", "visit_date"], how="left")
merged = merged.merge(labs_agg, on=["patient_id", "visit_date"], how="left")

# -------------------------
# Load models with fallbacks
# -------------------------
print("Loading models (this may take a while)...")

# generation (flan-t5-large -> fallback flan-t5-base)
try:
    gen_pipe = pipeline("text2text-generation", model=GEN_MODEL, device=DEVICE)
except Exception as e:
    print(f"[WARN] {GEN_MODEL} not available: {e}\nFalling back to google/flan-t5-base")
    gen_pipe = pipeline("text2text-generation", model="google/flan-t5-base", device=DEVICE)

# medical summarizer (BioGPT-Large -> fallback bart-large-cnn summarizer)
# BioGPT is a text-generation model; we'll use it to produce a concise SOAP-style summary via prompt
try:
    summarizer_pipe = pipeline("text-generation", model=MED_SUMMARIZER, device=DEVICE)
    summarizer_is_textgen = True
except Exception as e:
    print(f"[WARN] {MED_SUMMARIZER} not available: {e}\nFalling back to facebook/bart-large-cnn summarizer")
    summarizer_pipe = pipeline("summarization", model="facebook/bart-large-cnn", device=DEVICE)
    summarizer_is_textgen = False

# embeddings + zero-shot
emb_model = SentenceTransformer(EMB_MODEL)
zs_pipe = pipeline("zero-shot-classification", model=ZS_MODEL, device=DEVICE)

# prepare ICD texts & embeddings
if "description" in icd_df.columns and "code" in icd_df.columns:
    icd_texts = icd_df["description"].astype(str).tolist()
    icd_codes = icd_df["code"].astype(str).tolist()
    icd_embeddings = emb_model.encode(icd_texts, convert_to_tensor=True, show_progress_bar=False)
else:
    icd_texts, icd_codes, icd_embeddings = [], [], None
    print("[WARN] ICD file missing expected 'code' and 'description' columns — ICD suggestions disabled.")

# -------------------------
# Prompt templates
# -------------------------
GEN_PROMPT = (
    "You are a clinical scribe. Create a clear, factual clinical note from the structured data below. "
    "Do not hallucinate. Keep concise medical language.\n\nSTRUCTURED:\n{structured}\n\nCLINICAL NOTE:"
)

SUMMARY_PROMPT_TEXTGEN = (
    "Summarize the following clinical note into a concise SOAP-style summary "
    "(Subjective, Objective, Assessment, Plan). Keep important findings, vitals, procedures.\n\n{note}\n\nSOAP SUMMARY:"
)

# -------------------------
# ICD suggestion function
# -------------------------
def suggest_icd_from_note(note_text: str, top_k_candidates: int = TOP_K_CANDIDATES, top_k_return: int = TOP_K_RETURN) -> List[Dict]:
    if not icd_texts:
        return []
    q_emb = emb_model.encode(note_text, convert_to_tensor=True)
    cos_scores = util.cos_sim(q_emb, icd_embeddings)[0]
    topk = min(top_k_candidates, len(icd_texts))
    top_results = torch.topk(cos_scores, k=topk)
    top_idx = top_results.indices.cpu().numpy().tolist()
    candidate_texts = [icd_texts[i] for i in top_idx]

    # zero-shot re-rank: ask zs_pipe to score candidate labels
    # zs_pipe expects candidate_labels list; for long lists it still works but we pass candidate_texts
    try:
        zs = zs_pipe(note_text, candidate_texts, multi_label=True)
        suggestions = []
        for lbl, score in zip(zs["labels"], zs["scores"]):
            # find index of lbl in icd_texts (first match)
            try:
                i = icd_texts.index(lbl)
                suggestions.append({"code": icd_codes[i], "description": lbl, "score": float(score)})
            except ValueError:
                continue
            if len(suggestions) >= top_k_return:
                break
        return suggestions
    except Exception:
        # fallback: return top embedding matches without re-rank
        return [{"code": icd_codes[i], "description": icd_texts[i], "score": float(cos_scores[i])} for i in top_idx[:top_k_return]]

# -------------------------
# Note generation + summarization + pipeline
# -------------------------
def build_structured_for_prompt(row: pd.Series) -> dict:
    d = {
        "patient_id": row.get("patient_id", ""),
        "visit_date": row.get("visit_date", ""),
        "age": row.get("age", ""),
        "sex": row.get("sex", ""),
        "department": row.get("department", ""),
        "description": row.get("description", ""),
        "transcription": row.get("transcription", ""),
        "prescription_keywords": row.get("keywords", ""),
        "lab_samples": row.get("sample_name", "")
    }
    # include vitals if present
    vitals = {}
    for v in ("trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"):
        if v in row.index and row.get(v):
            vitals[v] = row.get(v)
    if vitals:
        d["vitals"] = vitals
    return d

def generate_raw_note(structured_dict: dict) -> str:
    prompt = GEN_PROMPT.format(structured=json.dumps(structured_dict, indent=2))
    out = gen_pipe(prompt, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=False)
    # pipeline returns list; handle both text2text and text-generation outputs
    if isinstance(out, list) and isinstance(out[0], dict):
        return out[0].get("generated_text") or out[0].get("summary_text") or str(out[0])
    return str(out)

def summarize_to_soap(raw_note: str) -> str:
    if summarizer_is_textgen:
        prompt = SUMMARY_PROMPT_TEXTGEN.format(note=raw_note)
        out = summarizer_pipe(prompt, max_new_tokens=SUM_MAX_NEW_TOKENS, do_sample=False)
        # text-generation returns list of dicts with 'generated_text'
        if isinstance(out, list) and isinstance(out[0], dict):
            summary = out[0].get("generated_text", "")
        else:
            summary = str(out)
    else:
        # summarizer_pipe is a summarization pipeline (e.g., BART); pass raw_note
        out = summarizer_pipe(raw_note, max_length=200, min_length=60, do_sample=False)
        summary = out[0].get("summary_text", "") if isinstance(out, list) and isinstance(out[0], dict) else str(out)

    summary = dedupe_consecutive_sentences(summary)
    return summary.strip()

# -------------------------
# Run full pipeline (process ALL rows)
# -------------------------
def run_pipeline(process_all: bool = True):
    rows = merged if process_all else merged.head(10)
    outputs = []

    for i, row in tqdm(rows.iterrows(), total=len(rows), desc="Processing encounters"):
        structured = build_structured_for_prompt(row)
        raw_note = generate_raw_note(structured)
        soap_summary = summarize_to_soap(raw_note)

        # Build final SOAP formatted note (ensuring sections exist)
        subj = structured.get("description", "").strip() or "No reported subjective symptoms."
        obj = []
        if "vitals" in structured:
            vit = "; ".join([f"{k}={v}" for k, v in structured["vitals"].items()])
            obj.append(f"Vitals: {vit}")
        if structured.get("lab_samples"):
            obj.append(f"Lab: {structured.get('lab_samples')}")
        obj_text = "\n".join(obj) if obj else "No objective data."
        assessment = soap_summary.split("ASSESSMENT:")[-1].split("PLAN:")[0].strip() if "ASSESSMENT:" in soap_summary else soap_summary
        plan = "Follow-up as per clinical judgment. See prescriptions."  # default plan
        if structured.get("transcription"):
            plan = structured.get("transcription")

        # Suggest ICD codes
        icd_suggestions = suggest_icd_from_note(soap_summary)

        # Compose note
        encounter_id = row.get("encounter_id") or make_encounter_id(row.get("patient_id","0"), i+1)
        note_text = (
            f"Encounter ID: {encounter_id}\nDate: {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"SUBJECTIVE:\n{subj}\n\n"
            f"OBJECTIVE:\n{obj_text}\n\n"
            f"ASSESSMENT:\n{assessment}\n\n"
            f"PLAN:\n{plan}\n\n"
            "ICD-10 Suggestions:\n"
        )
        for sug in icd_suggestions:
            note_text += f" - {clean_icd_label(sug['code'])} : {sug['description']} (score={sug['score']:.3f})\n"

        # save note file
        fname = f"{encounter_id}.txt"
        with open(NOTES_DIR / fname, "w", encoding="utf-8") as fh:
            fh.write("=== RAW NOTE ===\n")
            fh.write(raw_note + "\n\n")
            fh.write("=== SOAP SUMMARY ===\n")
            fh.write(soap_summary + "\n\n")
            fh.write("=== FINAL NOTE (SOAP) ===\n")
            fh.write(note_text + "\n")

        # add to outputs
        outputs.append({
            "encounter_id": encounter_id,
            "patient_id": row.get("patient_id"),
            "visit_date": row.get("visit_date"),
            "note_file": str(NOTES_DIR / fname),
            "soap_summary": soap_summary,
            "icd_suggestions": ";".join([f"{clean_icd_label(s['code'])}|{s['description']}|{s['score']:.3f}" for s in icd_suggestions])
        })

    out_df = pd.DataFrame(outputs)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nDone. Saved {len(outputs)} notes to {NOTES_DIR} and summary CSV to {OUT_CSV}")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    run_pipeline(process_all=True)
