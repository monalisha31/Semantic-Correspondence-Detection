import json
import os
import argparse
import re
import time
import numpy as np
from typing import List, Dict, Optional

from utils import setup_logging, load_json, save_json
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logger = setup_logging("llm_rerank")

FEW_SHOT_EXAMPLES = [
    {
        "class_a": "Class: Family. Supertypes: None. Attributes: address (EString). References: members -> Person [Composition].",
        "class_b": "Class: Household. Supertypes: None. Attributes: location (EString). References: residents -> Individual [Composition].",
        "result": '{"similarity_score": 0.92, "reasoning": "Both represent a domestic unit with a location attribute and a composition reference to people. Naming differences (Family/Household, address/location, members/residents) are synonymous."}'
    },
    {
        "class_a": "Class: Person. Supertypes: None. Attributes: name (EString), age (EInt). References: None.",
        "class_b": "Class: Employee. Supertypes: None. Attributes: loginID (EString). References: manager -> Employee [Association].",
        "result": '{"similarity_score": 0.25, "reasoning": "Person is a general entity with personal attributes (name, age). Employee is a domain-specific role with corporate attributes (loginID) and organizational hierarchy (manager). Different modeling concepts despite both representing people."}'
    },
    {
        "class_a": "Class: Node. Supertypes: Element. Attributes: id (EInt), label (EString). References: edges -> Edge [Association].",
        "class_b": "Class: Vertex. Supertypes: GraphElement. Attributes: identifier (EInt), name (EString). References: connections -> Link [Association].",
        "result": '{"similarity_score": 0.95, "reasoning": "Both represent graph nodes with an integer ID, a string label/name, and references to edges/connections. Node/Vertex and Edge/Link are standard synonyms in graph theory."}'
    },
]


SYSTEM_MSG = (
    "You are a Model-Driven Engineering expert specialized in Ecore meta-model analysis. "
    "Your task is to assess the degree of semantic correspondence between two EClass descriptions — "
    "that is, whether they represent the SAME or an overlapping modeling concept, "
    "even if they use different names, types, or structural conventions.\n\n"
    "Consider:\n"
    "- Synonym naming (Person/Individual, address/location)\n"
    "- Type equivalence (EString/String, EInt/Integer)\n"
    "- Structural role similarity (same attributes/references pattern)\n"
    "- Domain context (Family domain vs Industrial domain)\n\n"
    "Respond ONLY with a valid JSON object:\n"
    '{"similarity_score": float(0-1), "reasoning": "brief explanation of why the elements correspond or differ"}'
)


def build_prompt(class_a: str, class_b: str, few_shot: bool = True) -> str:
    user_msg = ""

    if few_shot:
        user_msg += "Here are some examples:\n\n"
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
            user_msg += f"Example {i}:\n"
            user_msg += f"Class A: {ex['class_a']}\n"
            user_msg += f"Class B: {ex['class_b']}\n"
            user_msg += f"Answer: {ex['result']}\n\n"
        user_msg += "Now analyze this pair:\n"

    user_msg += f"Class A: {class_a}\nClass B: {class_b}"

    return user_msg


def parse_llm_output(raw_text: str) -> Dict:
    cleaned = raw_text.strip()
    cleaned = re.sub(r'```json\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'```\s*', '', cleaned)

    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        json_str = cleaned[start:end + 1]
    else:
        json_str = cleaned

    json_str = json_str.replace(': True', ': true').replace(': False', ': false')
    json_str = json_str.replace("'", '"')
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)

    try:
        result = json.loads(json_str)
        result.setdefault('similarity_score', 0.5)
        result.setdefault('reasoning', 'No reasoning provided')

        score = result['similarity_score']
        if isinstance(score, str):
            score = float(score)
        result['similarity_score'] = max(0.0, min(1.0, float(score)))

        return result

    except (json.JSONDecodeError, ValueError) as e:
        score_match = re.search(r'"similarity_score"\s*:\s*([\d.]+)', cleaned)
        score = float(score_match.group(1)) if score_match else 0.0
        score = max(0.0, min(1.0, score))
        return {
            'similarity_score': score,
            'reasoning': f'Parse fallback: {cleaned[:200]}',
            'parse_error': str(e),
        }


def _load_api_key(env_var: str, env_path: str) -> Optional[str]:
    api_key = os.environ.get(env_var)
    if not api_key and os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{env_var}="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    return api_key


def initialize_llm(model_id: str = "gpt-4o-mini"):
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package required. pip install openai")
        return None

    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    api_key = _load_api_key("OPENAI_API_KEY", env_path)
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment or .env file.")
        return None

    client = OpenAI(api_key=api_key)
    logger.info(f"OpenAI client initialized, model: {model_id}")
    return {"client": client, "model_id": model_id}


def llm_verify_pair(llm, class_a: str, class_b: str,
                    few_shot: bool = True) -> Dict:
    prompt = build_prompt(class_a, class_b, few_shot)

    response = llm["client"].chat.completions.create(
        model=llm["model_id"],
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()

    return parse_llm_output(raw)


def evaluate_llm_on_ground_truth(
    gt_path: str,
    output_path: str = "results/llm_results.json",
    model_id: str = "gpt-4o-mini",
    limit: int = 100,
    few_shot: bool = True,
):
    from ground_truth import split_ground_truth

    _, _, test = split_ground_truth(gt_path)

    if limit:
        test = test[:limit]

    pipe = initialize_llm(model_id)
    if pipe is None:
        logger.error("Failed to initialize LLM")
        return

    results = []
    correct = 0
    total = 0

    logger.info(f"Verifying {len(test)} pairs...")

    for i, pair in enumerate(test):
        start = time.time()
        llm_result = llm_verify_pair(pipe, pair['text_a'], pair['text_b'], few_shot)
        elapsed = time.time() - start

        gt_label = pair['label'] >= 0.75
        sim_score = llm_result['similarity_score']
        predicted = sim_score >= 0.5

        is_correct = (predicted == gt_label)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            'pair_type': pair.get('pair_type', 'unknown'),
            'gt_label': pair['label'],
            'gt_binary': gt_label,
            'predicted': predicted,
            'similarity_score': sim_score,
            'reasoning': llm_result['reasoning'],
            'correct': is_correct,
            'time_s': elapsed,
        })

        if (i + 1) % 10 == 0:
            acc = correct / total
            logger.info(f"  Progress: {i+1}/{len(test)} | Running Acc: {acc:.4f}")

    predictions = np.array([1.0 if r['predicted'] else 0.0 for r in results])
    labels = np.array([1.0 if r['gt_binary'] else 0.0 for r in results])

    from evaluation.metrics import classification_metrics
    metrics = classification_metrics(predictions, labels, threshold=0.5)

    sim_scores = np.array([r['similarity_score'] for r in results])
    conf_metrics = classification_metrics(sim_scores, labels, threshold=0.5)

    output = {
        'metrics': metrics,
        'confidence_metrics': conf_metrics,
        'results': results,
        'n_pairs': len(test),
        'model_id': model_id,
        'few_shot': few_shot,
    }

    save_json(output, output_path)
    logger.info(f"LLM results saved to {output_path}")
    logger.info(f"LLM Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Re-ranking Evaluation")
    parser.add_argument("--gt", default="data/ground_truth.json")
    parser.add_argument("--output", default="results/llm_results.json")
    parser.add_argument("--model-id", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--no-few-shot", action="store_true")
    args = parser.parse_args()

    evaluate_llm_on_ground_truth(
        args.gt, args.output, args.model_id, args.limit,
        few_shot=not args.no_few_shot,
    )
