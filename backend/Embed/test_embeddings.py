import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

def load_embeddings(root: Path):
    embeddings_path = root / "embeddings.npy"
    keys_path = root / "keys.txt"

    if not embeddings_path.exists() or not keys_path.exists():
        raise FileNotFoundError("Please run embed_data.py before testing embeddings.")

    embeddings = np.load(embeddings_path)
    keys = keys_path.read_text(encoding="utf-8").splitlines()
    return embeddings, keys

def main():
    root = Path(__file__).resolve().parent
    embeddings, keys = load_embeddings(root)

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding matrix loaded. Type a query (blank to exit).\n")
    while True:
        query = input("Query: ").strip()
        if not query:
            break

        query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        # Cosine similarity because vectors are normalized
        scores = embeddings @ query_vec
        top_k = np.argsort(-scores)[:5]

        print("Top 5 most similar entries:\n")
        for rank, idx in enumerate(top_k, 1):
            print(f"{rank}. {keys[idx]} (score={scores[idx]:.4f})")
        print()

if __name__ == "__main__":
    main() 