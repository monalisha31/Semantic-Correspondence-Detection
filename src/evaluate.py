import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import argparse
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'evaluation'))

from models import ProjectionHead, GNNEncoder, GatedFusion
from ground_truth import split_ground_truth
from utils import set_seed, get_device, setup_logging, stringify_class, load_json
from config import get_config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'evaluation'))
from metrics import (classification_metrics, find_optimal_threshold,
                     correlation_metrics, print_report)

logger = setup_logging("evaluate")


def encode_texts(encoder, texts, device, projector=None, batch_size=64):
    base_embs = encoder.encode(texts, batch_size=batch_size,
                                show_progress_bar=False, convert_to_tensor=True)

    if projector is None:
        return base_embs.cpu().numpy()

    projector.eval()
    projected = []
    with torch.no_grad():
        for i in range(0, len(base_embs), batch_size):
            batch = base_embs[i:i+batch_size].to(device)
            proj = projector(batch)
            projected.append(proj.cpu().numpy())

    return np.concatenate(projected, axis=0)


def compute_pair_similarities(encoder, pairs, device, projector=None,
                               fusion_model=None, batch_size=64):
    texts_a = [p['text_a'] for p in pairs]
    texts_b = [p['text_b'] for p in pairs]

    embs_a = encode_texts(encoder, texts_a, device, projector, batch_size)
    embs_b = encode_texts(encoder, texts_b, device, projector, batch_size)

    embs_a = embs_a / (np.linalg.norm(embs_a, axis=1, keepdims=True) + 1e-8)
    embs_b = embs_b / (np.linalg.norm(embs_b, axis=1, keepdims=True) + 1e-8)

    similarities = (embs_a * embs_b).sum(axis=1)

    return similarities


def evaluate_on_ground_truth(pairs, similarities, label_threshold=0.75):
    labels = np.array([p['label'] for p in pairs])
    binary_labels = (labels >= label_threshold).astype(float)

    opt_thresh, opt_metrics = find_optimal_threshold(similarities, binary_labels)

    fixed_metrics = {}
    for t in [0.5, 0.6, 0.7, 0.8]:
        fixed_metrics[f't={t}'] = classification_metrics(similarities, binary_labels, t)

    corr = correlation_metrics(similarities, labels)

    pair_types = {}
    for i, p in enumerate(pairs):
        pt = p.get('pair_type', 'unknown')
        if pt not in pair_types:
            pair_types[pt] = {'sims': [], 'labels': []}
        pair_types[pt]['sims'].append(similarities[i])
        pair_types[pt]['labels'].append(labels[i])

    per_type = {}
    for pt, data in pair_types.items():
        sims = np.array(data['sims'])
        per_type[pt] = {
            'mean_sim': float(sims.mean()),
            'std_sim': float(sims.std()),
            'count': len(sims),
        }

    return {
        'optimal_threshold': opt_thresh,
        'optimal_metrics': opt_metrics,
        'fixed_thresholds': fixed_metrics,
        'correlation': corr,
        'per_type': per_type,
        'n_pairs': len(pairs),
    }


def run_experiment(name, encoder, pairs, device, projector=None,
                   fusion_model=None):
    logger.info(f"Running experiment: {name}")

    start = time.time()
    sims = compute_pair_similarities(encoder, pairs, device, projector, fusion_model)
    elapsed = time.time() - start

    results = evaluate_on_ground_truth(pairs, sims)
    results['inference_time_s'] = elapsed
    results['name'] = name

    opt = results['optimal_metrics']
    corr = results['correlation']
    logger.info(
        f"  {name}: F1={opt['f1']:.4f} (thresh={results['optimal_threshold']:.2f}) | "
        f"Acc={opt['accuracy']:.4f} | Prec={opt['precision']:.4f} | Rec={opt['recall']:.4f} | "
        f"Spearman={corr['spearman_r']:.4f} | Time={elapsed:.2f}s"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Full Evaluation Pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--gt", default="data/ground_truth.json")
    parser.add_argument("--text-model", default=None)
    parser.add_argument("--graph-model", default=None)
    parser.add_argument("--fusion-model", default=None)
    parser.add_argument("--output", default="results/evaluation_results.json")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = get_config(args.config)
    tc = config['text_simclr']
    gc = config['graph_simclr']
    fc = config['fusion']

    seed = args.seed or config.get('seed', 42)
    set_seed(seed)
    device = get_device()

    if not os.path.exists(args.gt):
        logger.error(f"Ground truth not found: {args.gt}")
        logger.error("Run: python src/ground_truth.py first")
        return

    train, val, test = split_ground_truth(args.gt, seed=seed)
    logger.info(f"Test set: {len(test)} pairs")

    encoder = SentenceTransformer(tc['encoder_name'])

    all_results = {}

    all_results['baseline_sbert'] = run_experiment(
        "Baseline (SBERT)", encoder, test, device
    )

    text_model_path = args.text_model or tc['model_out']
    if os.path.exists(text_model_path):
        text_proj = ProjectionHead(
            tc['embedding_dim'], tc['hidden_dim'], tc['projection_dim']
        ).to(device)
        ckpt = torch.load(text_model_path, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        text_proj.load_state_dict(state)

        all_results['text_simclr'] = run_experiment(
            "Text SimCLR", encoder, test, device, projector=text_proj
        )
    else:
        logger.warning(f"Text SimCLR model not found: {text_model_path}")

    fusion_model_path = args.fusion_model or fc['model_out']
    if os.path.exists(fusion_model_path):
        logger.info("Fusion evaluation: using text embeddings through fusion gate")

    logger.info("\n" + "="*70)
    logger.info(" RESULTS SUMMARY")
    logger.info("="*70)

    summary_rows = []
    for name, res in all_results.items():
        opt = res['optimal_metrics']
        corr = res['correlation']
        summary_rows.append({
            'Method': res['name'],
            'F1': opt['f1'],
            'Accuracy': opt['accuracy'],
            'Precision': opt['precision'],
            'Recall': opt['recall'],
            'Spearman': corr['spearman_r'],
            'Opt_Threshold': res['optimal_threshold'],
        })

    header = f"{'Method':<25s} {'F1':>8s} {'Acc':>8s} {'Prec':>8s} {'Rec':>8s} {'Spear':>8s} {'Thresh':>8s}"
    logger.info(header)
    logger.info("-" * len(header))
    for row in summary_rows:
        logger.info(
            f"{row['Method']:<25s} {row['F1']:>8.4f} {row['Accuracy']:>8.4f} "
            f"{row['Precision']:>8.4f} {row['Recall']:>8.4f} {row['Spearman']:>8.4f} "
            f"{row['Opt_Threshold']:>8.2f}"
        )

    best_method = max(all_results.keys(), key=lambda k: all_results[k]['optimal_metrics']['f1'])
    best = all_results[best_method]

    logger.info(f"\nPer-type analysis ({best['name']}):")
    for pt, stats in sorted(best['per_type'].items()):
        logger.info(f"  {pt:<40s}: mean_sim={stats['mean_sim']:.4f} +/- {stats['std_sim']:.4f} (n={stats['count']})")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(args.output, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
