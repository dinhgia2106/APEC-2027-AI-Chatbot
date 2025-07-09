import json
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


def load_json(path: Path) -> Any:
    """Read a JSON file using UTF-8 encoding."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten(data: Any, parent_key: str = "") -> List[Tuple[str, str]]:
    """Flatten a nested JSON structure into a list of (key_path, text) tuples."""
    items: List[Tuple[str, str]] = []

    if isinstance(data, dict):
        # Xử lý đặc biệt cho meeting objects
        if "event_title" in data and "date" in data:
            # Tạo chunk kết hợp cho meeting
            combined_text = f"{data['event_title']}"
            if "date" in data:
                combined_text += f" diễn ra ngày {data['date']}"
            if "venue" in data:
                combined_text += f" tại {data['venue']}"
            
            items.append((f"{parent_key}.combined", combined_text))
        
        # Tiếp tục flatten bình thường
        for k, v in data.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            items.extend(flatten(v, new_key))
            
    elif isinstance(data, list):
        for idx, v in enumerate(data):
            new_key = f"{parent_key}[{idx}]"
            items.extend(flatten(v, new_key))
    else:
        # Record non-empty scalar values only (str/int/float/bool)
        text = str(data).strip()
        if text:
            items.append((parent_key, text))

    return items


def main() -> None:
    root = Path(__file__).resolve().parent
    json_path = root.parent / "data" / "data.json"

    if not json_path.exists():
        raise FileNotFoundError("data.json not found in the project directory.")

    data = load_json(json_path)
    entries = flatten(data)

    # Tách key và text
    keys = [k for k, _ in entries]
    texts = [t for _, t in entries]

    print(f"Đã tạo {len(texts)} chunks để embedding")
    
    # Debug: In ra một vài chunk sample
    print("\nMột vài chunk mẫu:")
    for i, (key, text) in enumerate(entries[:10]):
        print(f"{i+1}. {key}: {text}")

    # Initialize the multilingual model
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model = SentenceTransformer(model_name)

    # Generate embeddings
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # Save results
    np.save(root / "embeddings.npy", embeddings)
    (root / "keys.txt").write_text("\n".join(keys), encoding="utf-8")

    print(f"Saved {len(texts)} embeddings to embeddings.npy and keys.txt")


if __name__ == "__main__":
    main() 