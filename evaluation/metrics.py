import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def recall_at_k(similarities: np.ndarray, labels: np.ndarray, k: int) -> float:
    n = similarities.shape[0]
    hits = 0

    for i in range(n):
        sims = similarities[i].copy()
        sims[i] = -np.inf

        top_k = np.argsort(sims)[-k:][::-1]
        if labels[i, top_k].any():
            hits += 1

    return hits / n


def precision_at_k(similarities: np.ndarray, labels: np.ndarray, k: int) -> float:
    n = similarities.shape[0]
    total_precision = 0.0

    for i in range(n):
        sims = similarities[i].copy()
        sims[i] = -np.inf

        top_k = np.argsort(sims)[-k:][::-1]
        n_relevant = labels[i, top_k].sum()
        total_precision += n_relevant / k

    return total_precision / n


def mean_reciprocal_rank(similarities: np.ndarray, labels: np.ndarray) -> float:
    n = similarities.shape[0]
    total_rr = 0.0

    for i in range(n):
        sims = similarities[i].copy()
        sims[i] = -np.inf

        ranked = np.argsort(sims)[::-1]
        for rank, idx in enumerate(ranked, 1):
            if labels[i, idx]:
                total_rr += 1.0 / rank
                break

    return total_rr / n


def mean_average_precision(similarities: np.ndarray, labels: np.ndarray) -> float:
    n = similarities.shape[0]
    total_ap = 0.0

    for i in range(n):
        sims = similarities[i].copy()
        sims[i] = -np.inf

        ranked = np.argsort(sims)[::-1]

        n_relevant = 0
        sum_precision = 0.0

        for rank, idx in enumerate(ranked, 1):
            if labels[i, idx]:
                n_relevant += 1
                sum_precision += n_relevant / rank

        if n_relevant > 0:
            total_ap += sum_precision / n_relevant

    return total_ap / n


def ndcg_at_k(similarities: np.ndarray, relevance: np.ndarray, k: int) -> float:
    n = similarities.shape[0]
    total_ndcg = 0.0

    for i in range(n):
        sims = similarities[i].copy()
        sims[i] = -np.inf

        top_k = np.argsort(sims)[-k:][::-1]

        dcg = 0.0
        for rank, idx in enumerate(top_k):
            dcg += relevance[i, idx] / np.log2(rank + 2)

        ideal_rels = np.sort(relevance[i])[::-1][:k]
        idcg = 0.0
        for rank, rel in enumerate(ideal_rels):
            idcg += rel / np.log2(rank + 2)

        if idcg > 0:
            total_ndcg += dcg / idcg

    return total_ndcg / n


def classification_metrics(predictions: np.ndarray, labels: np.ndarray,
                          threshold: float = 0.5) -> Dict[str, float]:
    pred_binary = (predictions >= threshold).astype(int)
    labels_binary = labels.astype(int)

    tp = ((pred_binary == 1) & (labels_binary == 1)).sum()
    fp = ((pred_binary == 1) & (labels_binary == 0)).sum()
    fn = ((pred_binary == 0) & (labels_binary == 1)).sum()
    tn = ((pred_binary == 0) & (labels_binary == 0)).sum()

    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'threshold': threshold,
    }


def find_optimal_threshold(predictions: np.ndarray, labels: np.ndarray,
                          thresholds: np.ndarray = None) -> Tuple[float, Dict]:
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    best_f1 = 0.0
    best_threshold = 0.5
    best_metrics = {}

    for t in thresholds:
        metrics = classification_metrics(predictions, labels, threshold=t)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = t
            best_metrics = metrics

    return best_threshold, best_metrics


def correlation_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    from scipy import stats

    spearman_r, spearman_p = stats.spearmanr(predictions, labels)
    pearson_r, pearson_p = stats.pearsonr(predictions, labels)

    return {
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
    }


def full_evaluation_report(
    similarities: np.ndarray,
    labels: np.ndarray,
    k_values: List[int] = [1, 3, 5, 10],
    graded_relevance: np.ndarray = None,
) -> Dict:
    report = {}

    for k in k_values:
        report[f'recall@{k}'] = recall_at_k(similarities, labels, k)
        report[f'precision@{k}'] = precision_at_k(similarities, labels, k)
        if graded_relevance is not None:
            report[f'ndcg@{k}'] = ndcg_at_k(similarities, graded_relevance, k)

    report['mrr'] = mean_reciprocal_rank(similarities, labels)
    report['map'] = mean_average_precision(similarities, labels)

    return report


def print_report(report: Dict, title: str = "Evaluation Report"):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

    retrieval = {k: v for k, v in report.items()
                 if any(k.startswith(p) for p in ['recall', 'precision', 'ndcg', 'mrr', 'map'])}
    classification = {k: v for k, v in report.items()
                     if k in ['accuracy', 'f1', 'threshold']}

    if retrieval:
        print("\n Retrieval Metrics:")
        for k, v in sorted(retrieval.items()):
            print(f"   {k:>15s}: {v:.4f}")

    if classification:
        print("\n Classification Metrics:")
        for k, v in sorted(classification.items()):
            print(f"   {k:>15s}: {v:.4f}")

    other = {k: v for k, v in report.items() if k not in retrieval and k not in classification}
    if other:
        print("\n Other:")
        for k, v in sorted(other.items()):
            if isinstance(v, float):
                print(f"   {k:>15s}: {v:.4f}")
            else:
                print(f"   {k:>15s}: {v}")

    print(f"{'='*60}\n")
