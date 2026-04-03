import json
import random
import os
from typing import List, Dict, Tuple
from collections import defaultdict

from augmentations import transform_class_struct
from utils import stringify_class, save_json, load_json, setup_logging

logger = setup_logging("ground_truth")


def load_structured_data(path: str) -> List[dict]:
    with open(path, 'r') as f:
        return json.load(f)


def group_by_category(models_data: List[dict]) -> Dict[str, List[dict]]:
    all_classes = []
    for model in models_data:
        m_id = model.get('id', 'unknown')
        for cls in model.get('classes', []):
            all_classes.append({
                'model_id': m_id,
                'class': cls,
                'text': stringify_class(cls),
            })
    return all_classes


def construct_ground_truth(
    data_path: str,
    output_path: str,
    num_positive: int = 2000,
    num_hard_neg: int = 2000,
    num_easy_neg: int = 1000,
    transform_types: List[str] = None,
    seed: int = 42,
) -> Dict:
    random.seed(seed)

    if transform_types is None:
        transform_types = ["rename", "reorder", "type_abstract", "add_optional",
                          "flatten_inherit", "case_switch", "combined"]

    logger.info(f"Loading data from {data_path}...")
    models_data = load_structured_data(data_path)

    all_classes = []
    model_to_classes = defaultdict(list)

    for model in models_data:
        m_id = model.get('id', 'unknown')
        for cls in model.get('classes', []):
            entry = {
                'model_id': m_id,
                'class': cls,
                'text': stringify_class(cls),
            }
            all_classes.append(entry)
            model_to_classes[m_id].append(entry)

    logger.info(f"Total classes: {len(all_classes)} from {len(model_to_classes)} models")

    rich_classes = [c for c in all_classes
                    if len(c['class'].get('attributes', [])) >= 1 or
                       len(c['class'].get('references', [])) >= 1]

    logger.info(f"Rich classes (>=1 attr or ref): {len(rich_classes)}")

    pairs = []

    logger.info(f"Generating {num_positive} positive pairs...")

    sample_size = min(num_positive, len(rich_classes))
    sampled = random.sample(rich_classes, sample_size)

    for entry in sampled:
        transform = random.choice(transform_types)
        transformed = transform_class_struct(entry['class'], transform)

        pairs.append({
            'class_a': entry['class'],
            'text_a': entry['text'],
            'class_b': transformed,
            'text_b': stringify_class(transformed),
            'label': 1.0,
            'pair_type': f"positive_{transform}",
            'model_id_a': entry['model_id'],
            'model_id_b': entry['model_id'],
        })

    logger.info(f"Generating {num_hard_neg} hard negative pairs...")

    multi_class_models = {k: v for k, v in model_to_classes.items() if len(v) >= 2}
    model_ids = list(multi_class_models.keys())

    hard_neg_count = 0
    attempts = 0
    max_attempts = num_hard_neg * 10

    while hard_neg_count < num_hard_neg and attempts < max_attempts:
        attempts += 1
        if not model_ids:
            break

        m_id = random.choice(model_ids)
        classes = multi_class_models[m_id]

        if len(classes) < 2:
            continue

        c1, c2 = random.sample(classes, 2)

        if c1['class'].get('name') == c2['class'].get('name'):
            continue

        pairs.append({
            'class_a': c1['class'],
            'text_a': c1['text'],
            'class_b': c2['class'],
            'text_b': c2['text'],
            'label': 0.5,
            'pair_type': 'hard_negative_same_model',
            'model_id_a': m_id,
            'model_id_b': m_id,
        })
        hard_neg_count += 1

    logger.info(f"Generated {hard_neg_count} hard negatives")

    logger.info(f"Generating {num_easy_neg} easy negative pairs...")

    easy_neg_count = 0
    attempts = 0
    all_model_ids = list(model_to_classes.keys())

    while easy_neg_count < num_easy_neg and attempts < num_easy_neg * 5:
        attempts += 1

        if len(all_model_ids) < 2:
            break

        m1, m2 = random.sample(all_model_ids, 2)

        c1 = random.choice(model_to_classes[m1])
        c2 = random.choice(model_to_classes[m2])

        pairs.append({
            'class_a': c1['class'],
            'text_a': c1['text'],
            'class_b': c2['class'],
            'text_b': c2['text'],
            'label': 0.0,
            'pair_type': 'easy_negative_cross_model',
            'model_id_a': m1,
            'model_id_b': m2,
        })
        easy_neg_count += 1

    logger.info(f"Generated {easy_neg_count} easy negatives")

    random.shuffle(pairs)

    stats = {
        'total_pairs': len(pairs),
        'positive_pairs': sum(1 for p in pairs if p['label'] == 1.0),
        'hard_negative_pairs': sum(1 for p in pairs if p['label'] == 0.5),
        'easy_negative_pairs': sum(1 for p in pairs if p['label'] == 0.0),
        'total_classes_available': len(all_classes),
        'total_models': len(model_to_classes),
        'transform_types': transform_types,
    }

    result = {'pairs': pairs, 'stats': stats}

    logger.info(f"Ground truth stats: {json.dumps(stats, indent=2)}")

    save_json(result, output_path)
    logger.info(f"Saved to {output_path}")

    return result


def split_ground_truth(
    gt_path: str,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    random.seed(seed)

    data = load_json(gt_path)
    pairs = data['pairs']

    by_label = defaultdict(list)
    for p in pairs:
        by_label[p['label']].append(p)

    train, val, test = [], [], []

    for label, group in by_label.items():
        random.shuffle(group)
        n = len(group)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        test.extend(group[:n_test])
        val.extend(group[n_test:n_test + n_val])
        train.extend(group[n_test + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    return train, val, test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/structured_ecore.json")
    parser.add_argument("--output", default="data/ground_truth.json")
    parser.add_argument("--num-positive", type=int, default=2000)
    parser.add_argument("--num-hard-neg", type=int, default=2000)
    parser.add_argument("--num-easy-neg", type=int, default=1000)
    args = parser.parse_args()

    construct_ground_truth(
        args.data, args.output,
        num_positive=args.num_positive,
        num_hard_neg=args.num_hard_neg,
        num_easy_neg=args.num_easy_neg,
    )
