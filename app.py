from flask import Flask, render_template, request, jsonify
import os
import sys
import json
import re
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import ProjectionHead, GatedFusion
from data_loader import load_structured_models, flatten_to_elements, stringify_class
from llm_rerank import build_prompt, parse_llm_output, FEW_SHOT_EXAMPLES, SYSTEM_MSG
from config import get_config

app = Flask(__name__)

state = {
    'encoder': None,
    'projector': None,
    'fusion': None,
    'llm_pipe': None,
    'corpus': [],
    'corpus_embeddings': None,
    'device': None,
    'config': None,
    'has_llm': False,
}


def init_app(config_path="configs/default.yaml", skip_llm=False):
    config = get_config(config_path)
    tc = config['text_simclr']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state['device'] = device
    state['config'] = config

    print(f"[INIT] Device: {device}")

    print(f"[INIT] Loading sentence encoder: {tc['encoder_name']}...")
    state['encoder'] = SentenceTransformer(tc['encoder_name'])

    model_path = tc.get('model_out', 'checkpoints/text_simclr_head.pth')
    projector = ProjectionHead(
        tc['embedding_dim'], tc['hidden_dim'], tc['projection_dim'], dropout=0.1
    ).to(device)

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        proj_state = ckpt.get('model_state_dict', ckpt)
        projector.load_state_dict(proj_state)
        print(f"[INIT] Loaded SimCLR head from {model_path}")
    else:
        print(f"[INIT] Warning: No trained model at {model_path}, using random init")

    projector.eval()
    state['projector'] = projector

    data_path = config.get('data', {}).get('structured_json', 'data/structured_ecore.json')
    if os.path.exists(data_path):
        print(f"[INIT] Loading corpus from {data_path}...")
        models_data = load_structured_models(data_path)
        state['corpus'] = flatten_to_elements(models_data)

        print(f"[INIT] Pre-computing embeddings for {len(state['corpus'])} elements...")
        texts = [e['text'] for e in state['corpus']]
        state['corpus_embeddings'] = encode_texts(texts)
        print(f"[INIT] Corpus indexed ({state['corpus_embeddings'].shape})")
    else:
        print(f"[INIT] Warning: Corpus not found at {data_path}")

    if not skip_llm:
        try:
            from openai import OpenAI
            from llm_rerank import _load_api_key

            env_path = os.path.join(os.path.dirname(__file__), '.env')
            api_key = _load_api_key("OPENAI_API_KEY", env_path)
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment or .env file")

            llm_id = config.get('llm', {}).get('model_id', 'gpt-4o-mini')
            print(f"[INIT] Initializing OpenAI model: {llm_id}...")

            state['llm_pipe'] = {"client": OpenAI(api_key=api_key), "model_id": llm_id}
            state['has_llm'] = True
            print(f"[INIT] OpenAI GPT initialized successfully")
        except Exception as e:
            print(f"[INIT] LLM not available: {e}")
            print(f"[INIT] Running in SimCLR-only mode (no semantic verification)")
    else:
        print(f"[INIT] LLM skipped (--no-llm flag)")


def encode_texts(texts, batch_size=64):
    encoder = state['encoder']
    projector = state['projector']
    device = state['device']

    base = encoder.encode(texts, batch_size=batch_size, show_progress_bar=False,
                          convert_to_tensor=True)

    projected = []
    with torch.no_grad():
        for i in range(0, len(base), batch_size):
            batch = base[i:i+batch_size].to(device)
            proj = projector(batch)
            projected.append(proj.cpu().numpy())

    embs = np.concatenate(projected, axis=0)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / (norms + 1e-8)


def compute_similarity(text_a, text_b):
    embs = encode_texts([text_a, text_b])
    return float(np.dot(embs[0], embs[1]))


def retrieve_similar(query_text, top_k=5):
    if state['corpus_embeddings'] is None or len(state['corpus']) == 0:
        return []

    query_emb = encode_texts([query_text])
    sims = np.dot(state['corpus_embeddings'], query_emb.T).flatten()

    top_indices = np.argsort(sims)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        elem = state['corpus'][idx]
        results.append({
            'model_id': elem['model_id'],
            'class_name': elem['class_name'],
            'text': elem['text'],
            'similarity': float(sims[idx]),
        })

    return results


def llm_verify(text_a, text_b):
    if not state['has_llm'] or state['llm_pipe'] is None:
        return {
            'similarity_score': None,
            'reasoning': 'LLM not available. Showing SimCLR structural similarity only.',
        }

    prompt = build_prompt(text_a, text_b, few_shot=True)

    try:
        llm = state['llm_pipe']
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
    except Exception as e:
        return {
            'similarity_score': None,
            'reasoning': f'LLM inference error: {str(e)}',
        }


@app.route('/')
def index():
    return render_template('index.html',
                           has_llm=state['has_llm'],
                           corpus_size=len(state['corpus']))


@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    text_a = data.get('textA', '').strip()
    text_b = data.get('textB', '').strip()

    if not text_a or not text_b:
        return jsonify({'error': 'Both Class A and Class B are required.'}), 400

    t0 = time.time()

    similarity = compute_similarity(text_a, text_b)

    llm_result = llm_verify(text_a, text_b)

    elapsed = time.time() - t0

    llm_score = llm_result.get('similarity_score')
    if llm_score is not None:
        combined_score = 0.4 * similarity + 0.6 * llm_score
    else:
        combined_score = similarity

    return jsonify({
        'structural_similarity': similarity,
        'semantic_similarity': llm_score,
        'combined_score': round(combined_score, 4),
        'reasoning': llm_result.get('reasoning', 'Structural similarity computed.'),
        'inference_time': round(elapsed, 2),
        'mode': 'simclr+llm' if state['has_llm'] else 'simclr_only',
    })


@app.route('/retrieve', methods=['POST'])
def retrieve():
    data = request.json
    query = data.get('query', '').strip()
    top_k = min(data.get('top_k', 5), 20)

    if not query:
        return jsonify({'error': 'Query text is required.'}), 400

    t0 = time.time()
    results = retrieve_similar(query, top_k=top_k)
    elapsed = time.time() - t0

    return jsonify({
        'results': results,
        'query': query,
        'inference_time': round(elapsed, 3),
        'corpus_size': len(state['corpus']),
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'has_llm': state['has_llm'],
        'corpus_size': len(state['corpus']),
        'device': str(state['device']),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ModelEquiv Web Interface")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM loading (SimCLR-only mode)")
    args = parser.parse_args()

    init_app(config_path=args.config, skip_llm=args.no_llm)

    print(f"\n{'='*50}")
    print(f" ModelEquiv running at http://{args.host}:{args.port}")
    print(f" Mode: {'SimCLR + LLM' if state['has_llm'] else 'SimCLR only'}")
    print(f" Corpus: {len(state['corpus'])} elements")
    print(f"{'='*50}\n")

    app.run(debug=False, port=args.port, host=args.host)
