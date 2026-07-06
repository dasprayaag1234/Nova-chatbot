# ============================================================
# RAG.PY — Nova's Knowledge Search Engine
# RAG = Retrieval Augmented Generation
# This file reads knowledge.txt, breaks it into chunks,
# and finds the most relevant pieces for any question asked
# ============================================================


# --- SECTION: IMPORTS ---
# sentence_transformers = understands meaning of text (not just keywords)
# numpy = helps us do math to find similarity between texts

from sentence_transformers import SentenceTransformer
import numpy as np


# --- SECTION: LOAD THE MODEL ---
# This model converts text into numbers (called embeddings)
# Similar meaning = similar numbers = easy to compare
# We use a small, fast, free model that runs locally on your PC

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


# --- SECTION: LOAD AND CHUNK KNOWLEDGE ---
# We read knowledge.txt and split it into chunks
# Each chunk = one paragraph or idea
# Nova searches across all chunks to find the most relevant ones

def load_knowledge(filepath="knowledge.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by double newline = one chunk per paragraph
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return chunks


# --- SECTION: BUILD KNOWLEDGE INDEX ---
# We convert all chunks into embeddings (numbers) once at startup
# This lets us quickly compare any question against all chunks

def build_index(chunks):
    embeddings = EMBEDDING_MODEL.encode(chunks, show_progress_bar=False)
    return embeddings


# --- SECTION: SEARCH KNOWLEDGE ---
# Given a question, find the top N most relevant chunks
# How it works:
# 1. Convert the question into an embedding
# 2. Compare it against all chunk embeddings using cosine similarity
# 3. Return the top N most similar chunks

def search_knowledge(question, chunks, embeddings, top_n=3):

    # Convert question to embedding
    question_embedding = EMBEDDING_MODEL.encode([question])

    # Calculate cosine similarity between question and all chunks
    # Cosine similarity = how similar two vectors (number lists) are
    # Score of 1.0 = identical, 0.0 = completely different
    similarities = np.dot(embeddings, question_embedding.T).flatten()
    similarities = similarities / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding) + 1e-10
    )

    # Get the top N most similar chunk indices
    top_indices = np.argsort(similarities)[::-1][:top_n]

    # Return the actual text of those chunks
    results = [chunks[i] for i in top_indices]
    return results


# --- SECTION: GET CONTEXT FOR NOVA ---
# This is the main function nova.py will call
# It returns a formatted string of relevant knowledge
# that gets added to Nova's prompt before answering

def get_relevant_context(question, chunks, embeddings, top_n=3):
    relevant_chunks = search_knowledge(question, chunks, embeddings, top_n)

    # Format the chunks into a clean context block
    context = "\n\n---\n\n".join(relevant_chunks)
    return context