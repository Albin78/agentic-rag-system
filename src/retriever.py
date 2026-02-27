import numpy as np

class HybridRetriever:
    def __init__(self, dense_index, bm25, alpha=0.6):
        self.dense_index = dense_index
        self.bm25 = bm25
        self.alpha = alpha

    def _min_max_normalize(self, scores):
        min_s = np.min(scores)
        max_s = np.max(scores)

        if max_s - min_s == 0:
            return np.zeros_like(scores)

        return (scores - min_s) / (max_s - min_s)

    def search(self, query_vec, tokenized_query,
               candidate_pool=50, top_k=5):

        # -------------------------
        # 1️⃣ Dense retrieval
        # -------------------------
        dense_dist, dense_ind = self.dense_index.search(
            query_vec, candidate_pool
        )

        dense_scores = dense_dist[0]
        dense_indices = dense_ind[0]

        # Normalize dense scores
        dense_scores = self._min_max_normalize(dense_scores)

        # -------------------------
        # 2️⃣ BM25 retrieval
        # -------------------------
        bm25_scores = self.bm25.get_scores(tokenized_query)

        bm25_top_indices = np.argsort(bm25_scores)[::-1][:candidate_pool]
        bm25_top_scores = bm25_scores[bm25_top_indices]

        # Normalize BM25 scores
        bm25_top_scores = self._min_max_normalize(bm25_top_scores)

        # -------------------------
        # 3️⃣ Fusion
        # -------------------------
        combined = {}

        # Add dense scores
        for idx, score in zip(dense_indices, dense_scores):
            combined[idx] = self.alpha * score

        # Add BM25 scores
        for idx, score in zip(bm25_top_indices, bm25_top_scores):
            combined[idx] = combined.get(idx, 0) + \
                            (1 - self.alpha) * score

        # -------------------------
        # 4️⃣ Ranking
        # -------------------------
        ranked = sorted(
            combined.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]