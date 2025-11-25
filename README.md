# HIPS (Hierarchical Inference of Progression Stage)

HIPS is a hierarchical transformer framework designed to infer Alzheimer’s disease pathology staging from single-nucleus transcriptomic data. It leverages the brain’s cellular hierarchy to identify molecular changes associated with disease progression.

## Project Overview

This repository contains the pipeline for embedding single-cell data and training the stage prediction model.

### 1. Data Embedding (`scgpt_embeddings.py`)
This script processes raw transcriptomic data to generate cell-level embeddings using a pre-trained **scGPT** model.

### 2. Model Training (`scgpt_transformer.py`)
This script contains the `TransformerNet` architecture and training loop. It takes the embeddings generated in step 1 and trains the model to predict donor-level Alzheimer's stages.

## Usage

**Step 1: Generate Embeddings**
Update the `model_dir` and data paths in `scgpt_embeddings.py`, then run:
```bash
python scgpt_embeddings.py
```

**Step 2: Train Model Once embeddings are ready, run the training pipeline:**
```bash
python scgpt_transformer.py
```
**Requirements**
```bash
Python 3.9+
scgpt (and its dependencies)
torch
scikit-learn
scanpy
```


**Note: Project is currently in progress**
