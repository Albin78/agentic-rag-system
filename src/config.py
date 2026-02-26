from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "BAAI/bge-base-en-v1.5"
    chunk_size: int = 500
    overlap: int = 50
    hnsw_m: int = 32
    ef_search: int = 128
    ef_construction: int = 200
    metric: str = "ip"
    top_k: int = 5