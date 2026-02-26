from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name, device):
        self.model = SentenceTransformer(model_name)
        self.model.to(device)

    def encode_docs(self, texts):
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query):
        instruction = "Represent this sentence for searching relevant passages: "
        emb = self.model.encode(
            [instruction + query],
            normalize_embeddings=True
        )
        return emb.astype(np.float32)