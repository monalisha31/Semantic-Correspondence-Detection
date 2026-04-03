import json
import os
import time
import argparse
import numpy as np
from typing import List, Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    from sentence_transformers import SentenceTransformer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from data_loader import load_structured_models, flatten_to_elements, stringify_class


def build_index(embeddings: np.ndarray, use_faiss: bool = True) -> object:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-8)

    if use_faiss:
        try:
            import faiss
            dim = embeddings_norm.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings_norm.astype(np.float32))
            return {'type': 'faiss', 'index': index, 'embeddings': embeddings_norm}
        except ImportError:
            pass

    return {'type': 'numpy', 'embeddings': embeddings_norm}


def search_index(index_obj: dict, query_emb: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)

    if index_obj['type'] == 'faiss':
        sims, indices = index_obj['index'].search(query_norm.astype(np.float32), k)
        return indices[0], sims[0]
    else:
        sims = np.dot(index_obj['embeddings'], query_norm.T).flatten()
        top_k = np.argsort(sims)[-k:][::-1]
        return top_k, sims[top_k]


class RetrievalPipeline:

    def __init__(self, config: dict, text_model_path: str = None,
                 corpus_path: str = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for RetrievalPipeline")

        self.config = config
        tc = config.get('text_simclr', {})

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = SentenceTransformer(tc.get('encoder_name', 'all-MiniLM-L6-v2'))

        self.projector = None
        if text_model_path and os.path.exists(text_model_path):
            from models import ProjectionHead
            self.projector = ProjectionHead(
                tc.get('embedding_dim', 384),
                tc.get('hidden_dim', 512),
                tc.get('projection_dim', 256),
            ).to(self.device)
            ckpt = torch.load(text_model_path, map_location=self.device, weights_only=False)
            state = ckpt.get('model_state_dict', ckpt)
            self.projector.load_state_dict(state)
            self.projector.eval()

        self.corpus = []
        self.index = None

        if corpus_path:
            self.load_corpus(corpus_path)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        base_embs = self.encoder.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            convert_to_tensor=True
        )

        if self.projector is None:
            return base_embs.cpu().numpy()

        projected = []
        with torch.no_grad():
            for i in range(0, len(base_embs), batch_size):
                batch = base_embs[i:i+batch_size].to(self.device)
                proj = self.projector(batch)
                projected.append(proj.cpu().numpy())

        return np.concatenate(projected, axis=0)

    def load_corpus(self, data_path: str):
        models = load_structured_models(data_path)
        self.corpus = flatten_to_elements(models)

        texts = [e['text'] for e in self.corpus]
        embeddings = self.encode(texts)

        self.index = build_index(embeddings)
        print(f"Indexed {len(self.corpus)} elements")

    def retrieve(self, query_text: str, k: int = 10,
                 exclude_self: bool = True) -> List[Dict]:
        if self.index is None:
            raise ValueError("Corpus not loaded. Call load_corpus() first.")

        query_emb = self.encode([query_text])
        indices, sims = search_index(self.index, query_emb, k + (1 if exclude_self else 0))

        results = []
        for idx, sim in zip(indices, sims):
            if idx < 0 or idx >= len(self.corpus):
                continue
            elem = self.corpus[idx]
            if exclude_self and elem['text'] == query_text:
                continue
            results.append({
                'model_id': elem['model_id'],
                'class_name': elem['class_name'],
                'text': elem['text'],
                'similarity': float(sim),
            })
            if len(results) >= k:
                break

        return results

    def retrieve_batch(self, query_texts: List[str], k: int = 10) -> List[List[Dict]]:
        return [self.retrieve(qt, k) for qt in query_texts]


def main():
    parser = argparse.ArgumentParser(description="Retrieval Pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/structured_ecore.json")
    parser.add_argument("--text-model", default="checkpoints/text_simclr_head.pth")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit-queries", type=int, default=50)
    parser.add_argument("--output", default="results/retrieval_results.json")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, 'src')
    from config import get_config

    config = get_config(args.config)

    pipeline = RetrievalPipeline(
        config, text_model_path=args.text_model, corpus_path=args.data
    )

    elements = flatten_to_elements(load_structured_models(args.data))
    query_elements = elements[:args.limit_queries]

    results = []
    for i, elem in enumerate(query_elements):
        candidates = pipeline.retrieve(elem['text'], k=args.top_k)
        results.append({
            'query': {
                'model_id': elem['model_id'],
                'class_name': elem['class_name'],
                'text': elem['text'],
            },
            'candidates': candidates,
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(query_elements)} queries")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")
    print(f"Sample result:")
    print(f"  Query: {results[0]['query']['class_name']}")
    for c in results[0]['candidates'][:3]:
        print(f"  -> {c['class_name']} (sim={c['similarity']:.4f})")


if __name__ == "__main__":
    main()
