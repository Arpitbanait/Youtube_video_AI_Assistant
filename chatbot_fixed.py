"""Lightweight YouTube transcript → vector search → HuggingFace LLM example.

This file avoids depending on `langchain_community` so it runs without that package.
It uses:
- youtube_transcript_api to fetch transcripts
- sentence-transformers to create embeddings
- faiss (if available) for similarity search
- huggingface_hub.InferenceClient for chat completions

If any package is missing, install it into your venv (I can run installs if you want).
"""

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re

# Optional import: faiss may not be available on all systems; we try and fall back
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

load_dotenv()

# Configuration
MODEL_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_MODEL = "google/gemma-2-2b-it"  # chat model
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

print("HuggingFace token found:", bool(HUGGINGFACE_TOKEN))

# LLM client (Gemma is a chat model)
client = InferenceClient(model=HUGGINGFACE_MODEL, token=HUGGINGFACE_TOKEN)


def call_llm(prompt, temperature=0.9, max_tokens=300):
    resp = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message["content"]


def extract_video_id(url: str) -> str | None:
    patterns = [
        r'(?:v=|\\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\\.be\\/)([0-9A-Za-z_-]{11})',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def fetch_transcript(video_id: str) -> str | None:
    try:
        entries = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(e["text"] for e in entries)
    except Exception as e:
        print("Error fetching transcript:", e)
        return None


def chunk_text(text: str, chunk_size=1000, overlap=200):
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_size, n)
        chunk = text[i:end]
        chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks


def build_embeddings(chunks: list[str], model_name=MODEL_EMBED):
    model = SentenceTransformer(model_name)
    emb = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    # normalize for cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    return emb


def build_faiss_index(embeddings: np.ndarray):
    if not _HAS_FAISS:
        raise RuntimeError("faiss is not available in this environment")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))
    return index


def topk_search(index, embeddings, query_emb, k=4):
    # query_emb should be normalized
    if _HAS_FAISS:
        D, I = index.search(query_emb.astype('float32'), k)
        return I[0], D[0]
    else:
        # numpy fallback
        sims = (embeddings @ query_emb.T).squeeze()
        idx = np.argsort(-sims)[:k]
        return idx.tolist(), sims[idx].tolist()


def process_url(url: str):
    vid = extract_video_id(url)
    if not vid:
        raise ValueError("Could not extract video id from URL")
    text = fetch_transcript(vid)
    if not text:
        raise RuntimeError("No transcript available for this video")
    chunks = chunk_text(text)
    embeddings = build_embeddings(chunks)
    index = None
    if _HAS_FAISS:
        index = build_faiss_index(embeddings)
    return {
        "video_id": vid,
        "chunks": chunks,
        "embeddings": embeddings,
        "index": index,
    }


PROMPT_TMPL = """Answer the question using ONLY the transcript context below.
If answer isn't in context, say "I don't know."

Context:
{context}

Question: {question}"""


def answer_question(state, question: str):
    model = SentenceTransformer(MODEL_EMBED)
    q_emb = model.encode([question], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    ids, scores = topk_search(state["index"], state["embeddings"], q_emb, k=4)
    selected = [state["chunks"][i] for i in ids]
    context = "\n\n".join(selected)
    prompt = PROMPT_TMPL.format(context=context, question=question)
    return call_llm(prompt)


if __name__ == "__main__":
    # Example usage
    url = "https://www.youtube.com/watch?v=o6wx_UwB4AA"
    print("Processing:", url)
    state = process_url(url)
    q = "Summarize the paragraph in 10 points"
    print("Asking:", q)
    ans = answer_question(state, q)
    print("\nAnswer:\n", ans)
