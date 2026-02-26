class HybridRetriever:
    def __init__(self, dense_index, bm25, alpha=0.6):
        self.dense_index = dense_index
        self.bm25 = bm25
        self.alpha = alpha

    def search(self, query_vec, tokenized_query, candidate_pool=50, top_k=5):

        dense_dist, dense_ind = self.dense_index.search(query_vec, candidate_pool)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        combined = {}

        for idx, score in zip(dense_ind[0], dense_dist[0]):
            combined[idx] = self.alpha * score

        for idx in sorted(range(len(bm25_scores)),
                          key=lambda x: bm25_scores[x],
                          reverse=True)[:candidate_pool]:

            combined[idx] = combined.get(idx, 0) + (1 - self.alpha) * bm25_scores[idx]

        ranked = sorted(combined.items(),
                        key=lambda x: x[1],
                        reverse=True)

        return ranked[:top_k]