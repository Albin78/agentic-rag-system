import numpy as np

class Evaluator:

    @staticmethod
    def recall_at_k(predicted, relevant, k):
        return len(set(predicted[:k]) & set(relevant)) / len(relevant)

    @staticmethod
    def mrr(predicted, relevant):
        for rank, idx in enumerate(predicted):
            if idx in relevant:
                return 1.0 / (rank + 1)
        return 0.0

    @staticmethod
    def ndcg(predicted, relevant, k):
        dcg = 0
        for i in range(k):
            if predicted[i] in relevant:
                dcg += 1 / np.log2(i + 2)

        ideal_dcg = sum(
            1 / np.log2(i + 2)
            for i in range(min(len(relevant), k))
        )

        return dcg / ideal_dcg if ideal_dcg > 0 else 0