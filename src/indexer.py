import faiss

class HNSWIndexer:
    def __init__(self, dim, M=32, ef_search=128, ef_construction=200):
        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = ef_search
        self.index.hnsw.efConstruction = ef_construction

    def add(self, vectors):
        self.index.add(vectors)

    def search(self, query_vec, k):
        return self.index.search(query_vec, k)