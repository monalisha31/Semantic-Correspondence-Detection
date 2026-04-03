import json
import os
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


def stringify_class(cls_obj: dict) -> str:
    name = cls_obj.get('name', 'UnnamedClass')
    supertypes = ', '.join(cls_obj.get('supertypes', [])) or 'None'

    attrs = []
    for a in cls_obj.get('attributes', []):
        attrs.append(f"{a['name']} ({a['type']})")
    attrs_str = ', '.join(attrs) or 'None'

    refs = []
    for r in cls_obj.get('references', []):
        c = 'Composition' if r.get('containment') else 'Association'
        refs.append(f"{r['name']} -> {r['type']} [{c}]")
    refs_str = ', '.join(refs) or 'None'

    return (f"Class: {name}. Supertypes: {supertypes}. "
            f"Attributes: {attrs_str}. References: {refs_str}.")


def load_structured_models(path: str) -> List[dict]:
    with open(path, 'r') as f:
        return json.load(f)


def flatten_to_elements(models_data: List[dict]) -> List[dict]:
    elements = []
    for model in models_data:
        m_id = model.get('id', 'unknown')
        for cls in model.get('classes', []):
            elements.append({
                'model_id': m_id,
                'class': cls,
                'text': stringify_class(cls),
                'class_name': cls.get('name', 'Unnamed'),
            })
    return elements


def load_ground_truth(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def split_ground_truth(
    gt_path: str,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    random.seed(seed)
    data = load_ground_truth(gt_path)
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

    return train, val, test


def get_category_mapping(models_data: List[dict]) -> Dict[str, str]:
    mapping = {}
    for model in models_data:
        m_id = model.get('id', '')
        parts = m_id.split('/')
        if len(parts) >= 3:
            category = parts[2]
        else:
            category = 'unknown'
        mapping[m_id] = category
    return mapping


def compute_dataset_statistics(models_data: List[dict]) -> Dict:
    n_models = len(models_data)
    classes_per_model = [len(m.get('classes', [])) for m in models_data]

    all_classes = flatten_to_elements(models_data)
    attrs_per_class = [len(e['class'].get('attributes', [])) for e in all_classes]
    refs_per_class = [len(e['class'].get('references', [])) for e in all_classes]
    supers_per_class = [len(e['class'].get('supertypes', [])) for e in all_classes]

    type_counts = defaultdict(int)
    for e in all_classes:
        for a in e['class'].get('attributes', []):
            type_counts[a['type']] += 1

    return {
        'n_models': n_models,
        'n_classes': len(all_classes),
        'classes_per_model': {
            'mean': np.mean(classes_per_model),
            'std': np.std(classes_per_model),
            'min': min(classes_per_model),
            'max': max(classes_per_model),
            'median': np.median(classes_per_model),
        },
        'attributes_per_class': {
            'mean': np.mean(attrs_per_class),
            'std': np.std(attrs_per_class),
        },
        'references_per_class': {
            'mean': np.mean(refs_per_class),
            'std': np.std(refs_per_class),
        },
        'supertypes_per_class': {
            'mean': np.mean(supers_per_class),
        },
        'top_10_types': dict(sorted(type_counts.items(), key=lambda x: -x[1])[:10]),
    }
