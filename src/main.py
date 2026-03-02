import torch
import numpy as np
import re
from rank_bm25 import BM25Okapi
from config import Config
from embedder import Embedder
from indexer import HNSWIndexer
from retriever import HybridRetriever
from evaluation import Evaluator
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



def load_and_chunk(pdf_path, chunk_size, overlap):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunks = splitter.split_documents(documents)
    return chunks



def tokenize(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.split()

if __name__ == "__main__":

    config = Config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunks = load_and_chunk("Anatomy_and_Physiology.pdf", config.chunk_size, config.overlap)
    texts = [doc.page_content for doc in chunks]
    tokenized_docs = [tokenize(doc) for doc in texts]
    bm25 = BM25Okapi(tokenized_docs)

    # -------------------
    # 2. Embedding
    # -------------------
    embedder = Embedder(config.model_name, device)

    doc_embeddings = embedder.encode_docs(texts)
    dim = doc_embeddings.shape[1]

    # -------------------
    # 3. Indexing
    # -------------------
    indexer = HNSWIndexer(
        dim,
        M=config.hnsw_m,
        ef_search=config.ef_search,
        ef_construction=config.ef_construction
    )

    indexer.add(doc_embeddings)

    # -------------------
    # 4. Retrieval
    # -------------------
    retriever = HybridRetriever(indexer.index, bm25)

    query = "What are the six levels of organization?"
    tokenized_query = tokenize(query)

    query_vec = embedder.encode_query(query)

    results = retriever.search(
        query_vec,
        tokenized_query,
        candidate_pool=50,
        top_k=config.top_k
    )

    print("\nTop Results:")
    for idx, score in results:
        print(score, "→", documents[idx])