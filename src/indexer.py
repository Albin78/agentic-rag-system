import faiss

class HNSWIndexer:
    def __init__(self, dim, M=32, ef_search=128,
                 ef_construction=200, metric="ip"):

        if metric == "ip":
            metric_type = faiss.METRIC_INNER_PRODUCT
        elif metric == "l2":
            metric_type = faiss.METRIC_L2
        else:
            raise ValueError("Unsupported metric")

        self.index = faiss.IndexHNSWFlat(dim, M, metric_type)
        self.index.hnsw.efSearch = ef_search
        self.index.hnsw.efConstruction = ef_construction
        
    def add(self, vectors):
        self.index.add(vectors)

    def search(self, query_vec, k):
        return self.index.search(query_vec, k)