# CriteriaMatrix: Extracting Hidden Moderation Criteria from Reddit Communities

This repository contains the implementation for the paper "Decoding the Rule Book: Extracting Hidden Moderation Criteria from Reddit Communities" which introduces a novel approach to identify and extract implicit moderation criteria from historical Reddit moderation data.

**Accepted as a Main Conference paper at EMNLP 2025.**

## Overview

The CriteriaMatrix approach uses Partial Attention Transformer (PAT) to extract interpretable moderation criteria from Reddit communities. This allows for systematic comparison of moderation patterns across different subreddits.

## Quick Links

**Just want to explore the data?**
- 📊 [Released Artifacts](#artifacts-released) - Pre-computed CriteriaMatrix, and datasets
- 📈 [Analysis Examples](#key-outputs-and-analysis) - See what insights you can extract

**Want to reproduce or extend?**
- 🔧 [Setup Instructions](#setup)
- 📋 [Pipeline Overview](#pipeline-steps)

## Artifacts Released

To facilitate reproduction and further research, we release the following artifacts that you can **download and use directly without running training**:

1. **Subreddit list**: `subreddit_splits/train.txt`
   - 60 subreddit names used in the study
   
2. **Preprocessed training data**: `datasets/train_data2/`
   - Balanced moderation datasets for each subreddit
   - Format: CSV with [text, label] columns
   - **Use these to explore the data or train your own models**
   - 
3. **Vocabulary lists**: `top_10k_voca/`
   - 10,000 top terms for each n-gram size (1-9)
   - Ranked by Llama-3.1 language model probability
   - **Use these to understand what terms are being scored**

4**Term scores (CriteriaMatrix)**: `sb_term_scores/`
   - Moderation scores for all terms across all subreddits
   - **This is the main output - start your analysis here!**
   - Can be combined to form the complete CriteriaMatrix

### Download Artifacts
```bash
# Install huggingface_hub if not already installed
pip install huggingface-hub

# Download dataset using HuggingFace Hub
from huggingface_hub import snapshot_download

# Download all artifacts to output/ directory
snapshot_download(
    repo_id="youngwoo-umass/CriteriaMatrix",
    repo_type="dataset",
    local_dir="output"
)
```

## Key Outputs and Analysis

## Setup

All commands should be run from the repository root with `PYTHONPATH=src`:
```bash
# Clone repository
git clone https://github.com/youngwoo-umass/criteria-matrix.git
cd criteria-matrix

# Install dependencies
pip install -r requirements.txt

# Run commands with PYTHONPATH=src prefix
PYTHONPATH=src python src/rule_gen/reddit/bert_pat/train_pat.py --subreddit politics
```
## Pipeline Steps

The analysis pipeline consists of four main steps:

### Step 1: Dataset Preparation (Reference Implementation)
**File:** `src/rule_gen/reddit/dataset_helper/build_dataset2.py`

**Purpose:** Shows how balanced datasets are prepared from Reddit moderation data for training PAT models.

**What it does:**
- Creates balanced datasets with equal proportions of moderated and non-moderated content
- Splits data across different subreddits for training individual PAT models
- Demonstrates the data preprocessing pipeline used in the paper

**Inputs:** : 
- Positive samples (moderated comments): `{data_root_path}/reddit/reddit-removal-log.csv` (from Chandrasekharan et al., 2018)
  - Contains moderated (deleted) Reddit comments collected from May 2016 to March 2017
- Negative samples (non-moderated comments): `{output_root_path}/reddit/subreddit_samples/{subreddit}.jsonl` (not included in repo)
  - To build these samples, we downloaded Reddit Pushshift dumps from the same time period. We sampled the desired number of comments from each subreddit, as the complete dumps were too large to fit in memory.
  - Pushshift dumps torrent: https://academictorrents.com/details/1614740ac8c94505e4ecb9d88be8bed7b6afddd4
  
**Outputs:**
- Training datasets: `{output_root_path}/datasets/train_data2/{subreddit}/train.csv`
- Validation datasets: `{output_root_path}/datasets/train_data2/{subreddit}/val.csv`
- Format: CSV with columns `[text, label]` where label ∈ {0,1}

**Note:** This script is provided for reference and transparency but is not directly runnable without access to large Reddit data dumps. The code shows:
- How the original Reddit moderation dataset was processed
- Data balancing and filtering procedures
- Subreddit-specific dataset creation methodology

**For researchers:** If you have access to Reddit data dumps, you can adapt this code for your own dataset preparation.

---

### Step 2: PAT Model Training
**File:** `src/rule_gen/reddit/bert_pat/train_pat.py`

**Purpose:** Trains Partial Attention Transformer (PAT) models for each subreddit to learn moderation patterns.

**What it does:**
- Trains a separate PAT model for each subreddit using its specific moderation data
- PAT learns to predict moderation decisions based on text spans
- Models are trained to assign well-calibrated probability scores to lexical expressions
- Saves trained models for each subreddit

**Inputs:**
- Training data: `{output_root_path}/datasets/train_data2/{subreddit}/train.csv`
- Validation data: `{output_root_path}/datasets/train_data2/{subreddit}/val.csv`
- Base BERT model: `bert-base-uncased` (from HuggingFace)

**Outputs:**
- Trained PAT models: `{model_save_path}/bert_ts_{subreddit}/`
  - Model files: `pytorch_model.bin`, `config.json`, `tokenizer_config.json`, etc.
  - One model per subreddit (e.g., `bert_ts_politics/`, `bert_ts_Games/`)

**Usage:**
```bash
python src/rule_gen/reddit/bert_pat/train_pat.py --subreddit [SUBREDDIT_NAME]
```

**Example:**
```bash
python src/rule_gen/reddit/bert_pat/train_pat.py --subreddit politics
```

---

### Step 3: Vocabulary Scoring
**File:** `src/rule_gen/reddit/term_scoring/pat_inf_filter.py`

**Purpose:** Uses trained PAT models to score vocabulary terms and build the CriteriaMatrix.

**What it does:**
- Takes a shared vocabulary of n-grams (1-9 tokens) across subreddits
- Applies each subreddit's PAT model to score all terms in the vocabulary
- Generates moderation probability scores for each term in each subreddit
- Builds the CriteriaMatrix M where M[i,j] = score for term j in subreddit i

**Inputs:**
- Trained PAT models: `{model_save_path}/bert_ts_{subreddit}/`
- Vocabulary terms: `{output_root_path}/reddit/top_10k_voca/{n}.pkl`
  - Contains: `[(term_key, term_text, lm_probability), ...]`
  - 10,000 most frequent n-grams per n value (1-9)
  - Term candidates initially built from subreddit corpora, then ranked by Llama-3 language model probability
- Subreddit list: `{output_root_path}/reddit/subreddit_splits/train.txt`

**Outputs:**
- Score files per subreddit: `{output_root_path}/reddit/sb_term_scores/{subreddit}.{n}.pkl`
  - Contains: `list[float]` - 10,000 scores corresponding to vocabulary terms in order
  - Each score represents P(moderated | term, subreddit)
  - These files collectively form the rows of the CriteriaMatrix

**Usage:**
```bash
python src/rule_gen/reddit/term_scoring/pat_inf_filter.py --n [NGRAM_SIZE]
```

**Parameters:**
- `--n`: N-gram size (1 for unigrams, 2 for bigrams, etc., up to 9)

**Example:**
```bash
# Score all unigrams for all subreddits
python src/rule_gen/reddit/term_scoring/pat_inf_filter.py --n 1

# Score all bigrams for all subreddits
python src/rule_gen/reddit/term_scoring/pat_inf_filter.py --n 2
```

---

### Step 4: Clustering Analysis
**File:** `src/rule_gen/reddit/term_scoring/score_analysis/run_kmeans.py`

**Purpose:** Performs clustering analysis on the CriteriaMatrix to identify patterns in moderation criteria.

**What it does:**
- Clusters terms based on their score similarity across subreddits using K-means (k=100)
- Uses Pearson correlation coefficients as similarity metric
- Identifies groups of terms that have similar moderation patterns
- Analyzes clusters to understand different types of violations (e.g., personal attacks, hate speech)
- Calculates silhouette scores to evaluate clustering quality

**Inputs:**
- Score matrices: `{output_root_path}/reddit/sb_term_scores/{subreddit}.{n}.pkl`
  - Loads scores for multiple n-gram sizes (typically n=1-4)
- Term lists: `{output_root_path}/reddit/top_10k_voca/{n}.pkl`
- Subreddit list: `{output_root_path}/reddit/subreddit_splits/train.txt`

**Outputs:**
- Console output showing:
  - Cluster assignments for vocabulary terms
  - Top distinguishing words per cluster
  - Cluster sizes and characteristics
  - Silhouette scores for cluster quality evaluation
- Optional save location: `{output_root_path}/reddit/clustering_results/`
  - Can be modified in the script to save cluster assignments

**Usage:**
```bash
python src/rule_gen/reddit/term_scoring/score_analysis/run_kmeans.py
```

**Output example:**
```
Cluster 0:
Avg val: 0.784, Size: 156
Most influential features (in order):
politics: 0.891 Games: 0.823 askscience: 0.756 ...
Top terms: idiot, stupid, moron, dumb, ...
```

---

## Key File Locations

### Configuration Paths:
- `output_root_path`: Defined in `src/rule_gen/cpath.py`
- `model_save_path`: From `desk_util.path_helper.get_model_save_path()`

### Directory Structure:
```
{output_root_path}/reddit/
├── subreddit_samples/           # Raw Reddit data (not included)
│   └── {subreddit}.jsonl
├── reddit-removal-log.csv       # Moderation logs (from Chandrasekharan et al.)
├── datasets/
│   └── train_data2/
│       └── {subreddit}/
│           ├── train.csv        # Balanced training data
│           └── val.csv          # Validation data
├── subreddit_splits/
│   └── train.txt                # List of 60 subreddit names
├── top_10k_voca/               # Vocabulary files
│   ├── 1.pkl                    # 10k unigrams
│   ├── 2.pkl                    # 10k bigrams
│   └── ...                      # Up to 9.pkl
└── sb_term_scores/             # CriteriaMatrix rows
    ├── politics.1.pkl           # Unigram scores for r/politics
    ├── politics.2.pkl           # Bigram scores for r/politics
    ├── Games.1.pkl              # Unigram scores for r/Games
    └── ...

{model_save_path}/
├── bert_ts_politics/            # PAT model for r/politics
│   ├── pytorch_model.bin
│   ├── config.json
│   └── ...
├── bert_ts_Games/               # PAT model for r/Games
└── ...
```

## Runnable vs. Reference Code

### Runnable Steps (2-4)
Steps 2-4 can be executed if you have:
- Balanced moderation datasets (Step 2)
- Trained PAT models (Steps 3-4)
- CriteriaMatrix data (Step 4)

### Reference Implementation (Step 1)
Step 1 is provided as a reference implementation to show:
- **Methodology transparency**: Exactly how the paper's datasets were created
- **Reproducibility**: Complete pipeline documentation for future research
- **Adaptability**: Template for researchers with access to Reddit data

## Getting Started

1. **With preprocessed data**: Start directly from Step 2 if you have balanced moderation datasets
2. **Analysis only**: Use Step 4 if you have the CriteriaMatrix (term scores)

## Citation

If you use this code, please cite:
```bibtex
@article{kim2025decoding,
  title={Decoding the Rule Book: Extracting Hidden Moderation Criteria from Reddit Communities},
  author={Kim, Youngwoo and Beniwal, Himanshu and Johnson, Steven L and Hartvigsen, Thomas},
  journal={arXiv preprint arXiv:2509.02926},
  year={2025}
}
```

## Data

The original Reddit moderation dataset is from Chandrasekharan et al. (2018). Please refer to the paper for data access and preprocessing details.

## Notes

- Steps 2-4 depend on outputs of previous steps
- Training PAT models requires significant computational resources (GPU recommended)
- The vocabulary scoring step (Step 3) can be parallelized across subreddits for efficiency
- CriteriaMatrix is stored as separate files rather than a single large matrix for storage efficiency
- Clustering analysis provides interpretable insights into moderation patterns
- Step 1 code serves as documentation of the complete methodology