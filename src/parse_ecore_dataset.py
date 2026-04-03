import xml.etree.ElementTree as ET
import os
import sys
import json
import argparse
import logging
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger("ecore_parser")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

ECORE_NS = {
    'ecore': 'http://www.eclipse.org/emf/2002/Ecore',
    'xmi': 'http://www.omg.org/XMI',
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
}

ECORE_TYPE_MAP = {
    'ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EString': 'EString',
    'ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EInt': 'EInt',
    'ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EBoolean': 'EBoolean',
    'ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EFloat': 'EFloat',
    'ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EDouble': 'EDouble',
    'ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//ELong': 'ELong',
    'ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EDate': 'EDate',
    'ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EByte': 'EByte',
    'ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EChar': 'EChar',
    'ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EShort': 'EShort',
    'ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EBigDecimal': 'EBigDecimal',
    'ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EBigInteger': 'EBigInteger',
    'ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EByteArray': 'EByteArray',
    'ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EJavaObject': 'EJavaObject',
    'ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EJavaClass': 'EJavaClass',
    'ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EFeatureMapEntry': 'EFeatureMapEntry',
}


def resolve_type_name(raw_type: str) -> str:
    if not raw_type:
        return "EObject"

    if raw_type in ECORE_TYPE_MAP:
        return ECORE_TYPE_MAP[raw_type]

    if '#//' in raw_type:
        fragment = raw_type.split('#//')[-1]
        return fragment.split('/')[-1]

    if raw_type.startswith('ecore:'):
        parts = raw_type.split()
        if len(parts) >= 2 and '#//' in parts[-1]:
            return parts[-1].split('#//')[-1].split('/')[-1]
        return raw_type.split(':')[-1].split()[-1]

    return raw_type


def resolve_supertypes(raw: str) -> List[str]:
    if not raw:
        return []

    parts = raw.strip().split()
    result = []
    for p in parts:
        name = resolve_type_name(p)
        if name and name != "EObject":
            result.append(name)
    return result


def parse_ecore_file(filepath: str) -> Optional[Dict]:
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except (ET.ParseError, Exception):
        return None

    tag = root.tag
    ns_prefix = ''
    if '}' in tag:
        ns_prefix = tag[:tag.index('}') + 1]

    classes = []

    _extract_classes(root, ns_prefix, classes)

    if not classes:
        return None

    return {"classes": classes}


def _extract_classes(element, ns_prefix: str, classes: list):
    xsi_type_key = '{http://www.w3.org/2001/XMLSchema-instance}type'

    for child in element:
        child_tag = child.tag.replace(ns_prefix, '')

        if child_tag == 'eClassifiers':
            xsi_type = child.get(xsi_type_key, '')

            if 'EClass' not in xsi_type and xsi_type != '':
                continue
            if 'EEnum' in xsi_type or 'EDataType' in xsi_type:
                continue

            name = child.get('name', '')
            if not name:
                continue

            cls_data = _parse_eclass(child, ns_prefix)
            if cls_data:
                classes.append(cls_data)

        elif child_tag == 'eSubpackages' or child_tag.endswith('EPackage'):
            _extract_classes(child, ns_prefix, classes)


def _parse_eclass(element, ns_prefix: str) -> Optional[Dict]:
    name = element.get('name', '')
    if not name:
        return None

    raw_supers = element.get('eSuperTypes', '')
    supertypes = resolve_supertypes(raw_supers)

    attributes = []
    references = []

    xsi_type_key = '{http://www.w3.org/2001/XMLSchema-instance}type'

    for feat in element:
        feat_tag = feat.tag
        if '}' in feat_tag:
            feat_tag = feat_tag.split('}')[-1]

        if feat_tag != 'eStructuralFeatures':
            continue

        feat_name = feat.get('name', '')
        if not feat_name:
            continue

        xsi_type = feat.get(xsi_type_key, '')
        raw_etype = feat.get('eType', '')
        type_name = resolve_type_name(raw_etype)

        if 'EAttribute' in xsi_type:
            attributes.append({
                'name': feat_name,
                'type': type_name,
            })
        elif 'EReference' in xsi_type:
            containment = feat.get('containment', 'false').lower() == 'true'
            references.append({
                'name': feat_name,
                'type': type_name,
                'containment': containment,
            })
        else:
            if type_name.startswith('E') and type_name in (
                'EString', 'EInt', 'EBoolean', 'EFloat', 'EDouble',
                'ELong', 'EDate', 'EByte', 'EChar', 'EShort',
                'EBigDecimal', 'EBigInteger'
            ):
                attributes.append({'name': feat_name, 'type': type_name})
            else:
                containment = feat.get('containment', 'false').lower() == 'true'
                references.append({
                    'name': feat_name,
                    'type': type_name,
                    'containment': containment,
                })

    return {
        'name': name,
        'supertypes': supertypes,
        'attributes': attributes,
        'references': references,
    }


def find_ecore_files(root_dir: str) -> List[str]:
    ecore_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.ecore'):
                ecore_files.append(os.path.join(dirpath, f))
    return ecore_files


def process_single_file(args: Tuple[str, str]) -> Optional[Dict]:
    filepath, base_dir = args
    try:
        result = parse_ecore_file(filepath)
        if result is None:
            return None

        rel_path = os.path.relpath(filepath, os.path.dirname(base_dir))
        rel_path = rel_path.replace('\\', '/')

        result['id'] = rel_path
        return result
    except Exception:
        return None


def parse_dataset(
    input_dir: str,
    output_path: str,
    min_classes: int = 1,
    max_classes: int = 500,
    max_workers: int = 8,
    deduplicate: bool = True,
) -> List[Dict]:
    logger.info(f"Scanning for .ecore files in {input_dir}...")
    ecore_files = find_ecore_files(input_dir)
    logger.info(f"Found {len(ecore_files)} .ecore files")

    if not ecore_files:
        logger.error("No .ecore files found!")
        return []

    logger.info(f"Parsing with {max_workers} workers...")

    args_list = [(f, input_dir) for f in ecore_files]
    models = []
    errors = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, a): a for a in args_list}

        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            if done_count % 1000 == 0:
                logger.info(f"  Parsed {done_count}/{len(ecore_files)} files...")

            try:
                result = future.result()
                if result is not None:
                    models.append(result)
                else:
                    errors += 1
            except Exception:
                errors += 1

    logger.info(f"Successfully parsed: {len(models)} models ({errors} errors/skipped)")

    before = len(models)
    models = [m for m in models if min_classes <= len(m['classes']) <= max_classes]
    logger.info(f"After filtering ({min_classes} <= classes <= {max_classes}): "
                f"{len(models)} models (removed {before - len(models)})")

    if deduplicate:
        before = len(models)
        seen_signatures = set()
        unique_models = []

        for m in models:
            sig_parts = []
            for cls in sorted(m['classes'], key=lambda c: c['name']):
                attrs = sorted([a['name'] for a in cls.get('attributes', [])])
                refs = sorted([r['name'] for r in cls.get('references', [])])
                sig_parts.append(f"{cls['name']}:{','.join(attrs)}:{','.join(refs)}")

            sig = '|'.join(sig_parts)

            if sig not in seen_signatures:
                seen_signatures.add(sig)
                unique_models.append(m)

        models = unique_models
        logger.info(f"After deduplication: {len(models)} models (removed {before - len(models)} duplicates)")

    models.sort(key=lambda m: m['id'])

    total_classes = sum(len(m['classes']) for m in models)
    classes_per_model = [len(m['classes']) for m in models]

    logger.info(f"\n{'='*60}")
    logger.info(f" DATASET SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Models:           {len(models)}")
    logger.info(f"  Total EClasses:   {total_classes}")
    logger.info(f"  Classes/model:    {sum(classes_per_model)/len(classes_per_model):.1f} avg, "
                f"{min(classes_per_model)}-{max(classes_per_model)} range")

    total_attrs = sum(
        len(cls.get('attributes', []))
        for m in models for cls in m['classes']
    )
    total_refs = sum(
        len(cls.get('references', []))
        for m in models for cls in m['classes']
    )
    logger.info(f"  Total attributes: {total_attrs}")
    logger.info(f"  Total references: {total_refs}")
    logger.info(f"{'='*60}\n")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(models, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved to {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")

    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse raw .ecore files into structured JSON for training"
    )
    parser.add_argument("--input", "-i", default="../modelset/raw-data/repo-ecore-all")
    parser.add_argument("--output", "-o", default="data/structured_ecore_full.json")
    parser.add_argument("--min-classes", type=int, default=2)
    parser.add_argument("--max-classes", type=int, default=500)
    parser.add_argument("--workers", "-w", type=int, default=8)
    parser.add_argument("--no-dedup", action="store_true")
    args = parser.parse_args()

    parse_dataset(
        input_dir=args.input,
        output_path=args.output,
        min_classes=args.min_classes,
        max_classes=args.max_classes,
        max_workers=args.workers,
        deduplicate=not args.no_dedup,
    )
