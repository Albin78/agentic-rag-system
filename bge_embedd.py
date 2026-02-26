import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import time
from rank_bm25 import BM25Okapi
import re

# -------------------------------
# 1. Load BGE model
# -------------------------------
MODEL_NAME = "BAAI/bge-base-en-v1.5"
print("Loading model...")
model = SentenceTransformer(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("On device:", device)
model.to(device)
N = 10000
D = 768
M = 32

# -------------------------------
# 2. Load PDF
# -------------------------------
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# -------------------------------
# 3. Chunk text
# -------------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

# -------------------------------
# 4. Embed document chunks
# -------------------------------
def embed_chunks(chunks):
    return model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

# -------------------------------
# 5. Query search
# -------------------------------
# def search(query, doc_embeddings, chunks, top_k=3):

#     # IMPORTANT for BGE
    

#     query_embedding = model.encode(
#         [instruction + query],
#         normalize_embeddings=True
#     )[0]
#     # query_embedding_reshaped = query_embedding.reshape(-1, 1)
    # print("Shape of query embedding:", query_embedding.shape)
    # print("Type of query_embedding:", type(query_embedding))
    # print("Single Query embedding shape:", query_embedding[0].shape)
    # print("Random Query vector:", query_embedding[0][767])
    # print("Reshaped matrix:", query_embedding_reshaped.shape)
    # exit()
  
    # cosine similarity (because normalized)
    # scores = np.dot(doc_embeddings, query_embedding)
    # print("Shape of scores", scores.shape)

    # top_indices = np.argsort(scores)[-top_k:][::-1]

    # print("\nTop Results:\n")
    # for idx in top_indices:
    #     print(f"Score: {scores[idx]:.4f}")
    #     print(chunks[idx])
    #     print("-" * 60)

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()
    

def hybrid_search(query, chunks, model, index, bm25, top_k=5, alpha=0.6, candidate_pool=50):
    
    print("\n================ HYBRID SEARCH DEBUG ================\n")
    
    instruction = "Represent this sentence for searching relevant passages: "
    
    # ---------------------------
    # 1️⃣ Dense Retrieval
    # ---------------------------
    query_embedding = model.encode(
        [instruction + query],
        normalize_embeddings=True
    ).astype(np.float32, copy=False)

    dense_dist, dense_ind = index.search(query_embedding, candidate_pool)

    dense_scores = dense_dist[0]
    dense_indices = dense_ind[0]

    print("Dense indices:", dense_indices[:10])
    print("Dense scores (raw):", dense_scores[:10])

    # ---------------------------
    # 2️⃣ BM25 Retrieval
    # ---------------------------
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_pool]
    bm25_top_scores = bm25_scores[bm25_indices]

    print("BM25 indices:", bm25_indices[:10])
    print("BM25 scores (raw):", bm25_top_scores[:10])

    # ---------------------------
    # 3️⃣ Normalize Scores
    # ---------------------------
    dense_min, dense_max = dense_scores.min(), dense_scores.max()
    dense_norm = (dense_scores - dense_min) / (dense_max - dense_min + 1e-8)

    bm25_min, bm25_max = bm25_top_scores.min(), bm25_top_scores.max()
    bm25_norm = (bm25_top_scores - bm25_min) / (bm25_max - bm25_min + 1e-8)

    print("Dense normalized (first 5):", dense_norm[:5])
    print("BM25 normalized (first 5):", bm25_norm[:5])

    # ---------------------------
    # 4️⃣ Fusion
    # ---------------------------
    combined_scores = {}

    for idx, score in zip(dense_indices, dense_norm):
        combined_scores[idx] = alpha * score

    for idx, score in zip(bm25_indices, bm25_norm):
        if idx in combined_scores:
            combined_scores[idx] += (1 - alpha) * score
        else:
            combined_scores[idx] = (1 - alpha) * score

    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    print("\nTop Combined Scores:")
    for i in range(min(top_k, len(ranked))):
        print(f"Doc {ranked[i][0]} → {ranked[i][1]:.4f}")

    print("\n================ FINAL RESULTS ================\n")

    results = ranked[:top_k]

    for i, (idx, score) in enumerate(results):
        print(f"Rank {i+1}")
        print("Hybrid Score:", score)
        print(chunks[idx][:500])
        print("-"*60)

    return results, query_embedding


def evaluate(query_embedding, index_exact,index, k=3):


    # ANN search
    dist_ann, ind_ann = index.search(query_embedding, k)

    # Exact search
    dist_exact, ind_exact = index_exact.search(query_embedding, k)

    # Compute ANN recall
    ann_overlap = len(
        set(ind_ann[0]) & set(ind_exact[0])
    ) / k

    return ann_overlap
    # -------------------------------
# 6. Main
# -------------------------------
if __name__ == "__main__":

    pdf_path = "Anatomy_and_Physiology.pdf"
    
    start_time = time.time()

    text = extract_text(pdf_path)
    print("Length of text extracted", len(text))
    chunks = chunk_text(text)

    print("Building BM25 index...")
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    print(f"Total chunks: {len(chunks)}")

    doc_embeddings = embed_chunks(chunks)
    doc_embeddings = doc_embeddings.astype(np.float32)
    doc_embeddings = doc_embeddings.copy()

    print("Shape of embeddings", doc_embeddings.shape)
    print("Shape of each doc:", doc_embeddings.shape[1])

    query = "What are the six levels of organization in the human body?"
    instruction = "Represent this sentence for searching relevant passages: "

    # search(query, doc_embeddings, chunks)

    print("Embedding shape:", doc_embeddings.shape)
    print("Embedding dtype:", doc_embeddings.dtype)
    # print("Bytes for embedding:", doc_embeddings.nbytes)
    # # print("Random doc vector:", doc_embeddings[8192])

    # print("Shape of query embedding:", query_embedding.shape)
    # print("Type of query_embedding:", type(query_embedding))
    # # print("Bytes for query embedding:", query_embedding.nbytes) 
    # # print("Query vector:", query_embedding)
    # # exit()

    D = doc_embeddings.shape[1]   
    M = 32                        # graph neighbors

    # ---------------------------
    # Create HNSW index
    # ---------------------------
    index = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = 128
    print("Adding embeddings to HNSW index...")
    index.add(doc_embeddings)

    # Build exact index once
    index_exact = faiss.IndexFlatIP(D)
    print("Adding embeddings to exact index...")
    index_exact.add(doc_embeddings)

    print("Total vectors inside FAISS:", index.ntotal)
    
    top_k = 5
    search, query_embedding = hybrid_search(query, chunks, model, index, bm25, top_k=5, alpha=0.6, candidate_pool=50)

    # distances, indices = index.search(query_embedding.astype("float32", copy=False), k)
    # print("Distances:", distances)
    # print("Indices:", indices)
    # # print("Type of distance and indices:", type(distances), type(indices))
    print(search)

    # for i, idx in enumerate(indices[0]):
    #     print(f"Rank {i+1}")
    #     print("Score:", distances[0][i])
    #     print(chunks[idx])
    #     print("-"*60)
    
    metric = evaluate(query_embedding, index_exact, index, top_k)
    print("ANN Recall@{} vs Exact: {:.4f}".format(top_k, metric))
    end_time = time.time()
    print(f"Total time taken for the process: {end_time - start_time:.2f}s")
