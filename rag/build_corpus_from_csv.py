import re
import pickle
import pandas as pd
from pathlib import Path
from typing import List
from rank_bm25 import BM25Okapi

CSV_PATH = Path("rag_sample_qas_from_kis.csv")
INDEX_PATH = Path("corpus_index.pkl")

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    t = normalize(text)
    if not t:
        return []
    chunks, i = [], 0
    while i < len(t):
        chunks.append(t[i : i + max_chars])
        i += max_chars - overlap
    return chunks

def tokenize(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())

def main():
    df = pd.read_csv(CSV_PATH)
    required = {"ki_topic", "ki_text", "sample_question", "sample_ground_truth"}
    if not required.issubset(df.columns):
        raise SystemExit(f"CSV must contain columns: {sorted(required)}")

    # Build chunk dicts (no custom classes)
    chunk_dicts: List[dict] = []
    for i, row in df.iterrows():
        topic = normalize(row["ki_topic"])
        body  = normalize(row["ki_text"])
        for j, piece in enumerate(chunk_text(body)):
            chunk_dicts.append({
                "id": f"row::{i}__chunk::{j}",
                "row_id": int(i),
                "source": "csv:kis",
                "topic": topic,
                "text": piece,
            })

    if not chunk_dicts:
        raise SystemExit("No chunks were created from ki_text. Check your CSV content.")

    # BM25 over chunk texts
    corpus_tokens = [tokenize(c["text"]) for c in chunk_dicts]
    bm25 = BM25Okapi(corpus_tokens)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"chunks": chunk_dicts, "bm25": bm25}, f)

    print(f"Indexed {len(chunk_dicts)} chunks from {len(df)} rows → {INDEX_PATH}")

if __name__ == "__main__":
    main()
