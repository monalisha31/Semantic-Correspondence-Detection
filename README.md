# ModelEquiv: Detecting Semantic Correspondences in Ecore Models via Contrastive Learning and LLM-based Approach

This repository contains the implementation for the paper:

> **Detecting Semantic Correspondences in Ecore Models via Contrastive Learning and LLM-based Approach**

## Overview

ModelEquiv is a hybrid pipeline for detecting semantic correspondences between elements of independently developed Ecore metamodels. It combines contrastive representation learning (SimCLR) with LLM-based semantic verification (GPT-4o-mini).

**Pipeline stages:**

1. **Text SimCLR** - Contrastive learning on serialized EClass descriptions using domain-specific augmentations
2. **Graph SimCLR** - GCN-based contrastive learning on Ecore inheritance/reference graphs
3. **Gated Fusion** - Learned gating mechanism to combine text and graph embeddings
4. **LLM Verification** - GPT-4o-mini performs pairwise semantic reasoning on top-k candidates

## Project Structure

```
paper_code/
├── app.py                          # Flask web interface
├── configs/
│   └── default.yaml                # All hyperparameters
├── src/
│   ├── models.py                   # ProjectionHead, GNNEncoder, GatedFusion
│   ├── losses.py                   # NT-Xent, HardNegativeNTXent, SupConLoss
│   ├── augmentations.py            # Domain-specific text & structural augmentations
│   ├── train_text_simclr.py        # Stage 1: Text contrastive training
│   ├── train_graph_simclr.py       # Stage 2: Graph contrastive training
│   ├── train_fusion.py             # Stage 3: Multi-modal fusion training
│   ├── ground_truth.py             # Automated ground truth construction
│   ├── evaluate.py                 # Full evaluation pipeline
│   ├── retrieve.py                 # Retrieval pipeline with FAISS indexing
│   ├── llm_rerank.py               # LLM verification stage (GPT-4o-mini)
│   ├── parse_ecore_dataset.py      # Raw .ecore XML to structured JSON parser
│   ├── data_loader.py              # Data loading and splitting utilities
│   ├── config.py                   # Configuration loader
│   └── utils.py                    # Shared utilities (seeding, logging, etc.)
├── evaluation/
│   └── metrics.py                  # Recall@k, Precision@k, MRR, MAP, NDCG, F1
├── templates/
│   └── index.html                  # Web UI
├── data/                           # Dataset files (not included, see below)
├── checkpoints/                    # Trained model weights (generated during training)
|__ requirements.txt

```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For graph training (optional), install PyTorch Geometric separately:

```bash
pip install torch_geometric torch_scatter torch_sparse
```

### 2. Dataset

This project uses the [ModelSet](https://github.com/modelset/modelset-dataset) repository (v0.9.2).

1. Download or clone ModelSet:
   ```bash
   git clone https://github.com/modelset/modelset-dataset.git modelset
   ```

2. Parse the raw `.ecore` files into structured JSON:
   ```bash
   cd src
   python parse_ecore_dataset.py \
       --input ../modelset/raw-data/repo-ecore-all \
       --output ../data/structured_ecore.json \
       --min-classes 2 --max-classes 500
   ```

### 3. API Key (for LLM verification)

Add your OpenAI API key:

```bash

# Create .env and set: OPENAI_API_KEY=sk-your-key-here
```

The LLM stage is optional. The pipeline works without it in SimCLR-only mode.

## Usage

### Full Pipeline

Run all stages sequentially from the project root:

```bash
# Step 1: Generate ground truth pairs
cd src
python ground_truth.py --data ../data/structured_ecore.json --output ../data/ground_truth.json

# Step 2: Train Text SimCLR (Stage 1)
python train_text_simclr.py --config ../configs/default.yaml

# Step 3: Train Graph SimCLR (Stage 2, optional)
python train_graph_simclr.py --config ../configs/default.yaml

# Step 4: Train Fusion (Stage 3)
python train_fusion.py --config ../configs/default.yaml

# Step 5: Evaluate
python evaluate.py --config ../configs/default.yaml --gt ../data/ground_truth.json

# Step 6: LLM Re-ranking evaluation (requires API key)
python llm_rerank.py --gt ../data/ground_truth.json --limit 500
```

### Web Interface

Launch the interactive demo:

```bash
python app.py --port 5000
```

With LLM disabled:

```bash
python app.py --port 5000 --no-llm
```

Open `http://localhost:5000` to:
- **Compare Pair**: Enter two EClass descriptions and get structural + semantic similarity scores
- **Search Corpus**: Query against the full indexed corpus

### Retrieval

Run standalone retrieval:

```bash
cd src
python retrieve.py --data ../data/structured_ecore.json --text-model ../checkpoints/text_simclr_head.pth --top-k 10
```

## Configuration

All hyperparameters are in `configs/default.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sentence encoder | all-MiniLM-L6-v2 | Frozen base encoder (384-dim) |
| Projection head | 384 → 512 → 512 → 256 | 3-layer MLP with BN + ReLU |
| GNN encoder | 3-layer GCN, hidden=128 | With dual pooling (mean+max) |
| Temperature | 0.1 | NT-Xent contrastive loss |
| LLM | GPT-4o-mini | Few-shot semantic verification |

## Results

On the full ModelSet benchmark (89,128 EClasses from 3,589 metamodels):

| Method | Precision | Recall | F1 | Spearman |
|--------|-----------|--------|----|----------|
| SimCLR-only | 0.944 | 0.949 | 0.946 | 0.868 |
| LLM-only | 0.990 | 0.950 | 0.969 | 0.873 |
| **Hybrid** | **0.961** | **0.990** | **0.975** | **0.886** |



