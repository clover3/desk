# CriteriaMatrix-Resources: Reddit Content Moderation Dataset

This dataset accompanies the paper **"Decoding the Rule Book: Extracting Hidden Moderation Criteria from Reddit Communities"** (EMNLP 2025).

## Overview

CriteriaMatrix-Resources contains all artifacts needed to build and analyze **CriteriaMatrix** - term score tables that enable systematic comparison of content moderation patterns across 60+ Reddit communities. This dataset includes balanced training data, vocabularies for interpretable analysis, pre-computed CriteriaMatrix score tables, and supporting metadata for reproducibility.

### What's Included

1. **Training Data**: Balanced datasets of removed and kept comments from 60+ subreddits
2. **Vocabulary**: 10,000 most frequent n-grams (1-9 tokens) ranked by language model probability
3. **CriteriaMatrix Score Tables**: Pre-computed moderation probability scores for each vocabulary term in each subreddit
4. **Metadata**: Subreddit lists and splits for reproducible research

## Dataset Structure

```
CriteriaMatrix-Resources/
├── reddit/
│   ├── datasets/
│   │   └── train_data2/              # Training datasets by subreddit
│   │       ├── politics/
│   │       │   ├── train.csv
│   │       │   └── val.csv
│   │       └── ... (60+ subreddits)
│   │
│   ├── subreddit_splits/
│   │   ├── train.txt                 # Training subreddits
│   │   ├── val.txt                   # Validation subreddits
│   │   └── test.txt                  # Test subreddits
│   │
│   ├── top_10k_voca/                 # Vocabulary files
│   │   ├── 1.pkl                     # 10k unigrams
│   │   ├── 2.pkl                     # 10k bigrams
│   │   └── ... (up to 9.pkl)
│   │
│   ├── sb_term_scores/               # CriteriaMatrix score tables
│   │   ├── politics.1.pkl            # Unigram scores for r/politics
│   │   ├── askscience.1.pkl          # Unigram scores for r/askscience
│   │   └── ... (60+ subreddits × 9 n-gram sizes)
│   │
│   └── reddit-removal-log.csv        # Original moderation data
```

## Data Formats

### Training CSV Files

Each CSV contains two columns:
- **text**: The comment/post text
- **label**: Binary label (0 = kept, 1 = removed by moderators)

```csv
text,labeld
"This is a helpful comment",0
"This violates community guidelines",1
```

### Vocabulary Files (`top_10k_voca/{n}.pkl`)

Python pickle files containing lists of tuples:
```python
[(term_key, term_text, lm_probability), ...]

term_key: list[str]  # List of tokens that constitute the term
term_text: str  # surface form of term
lm_probabilty: float  # Probability of term generated based on Llama 3 
```


### Term Score Files (`sb_term_scores/{subreddit}.{n}.pkl`)

Python pickle files containing lists of 10,000 floats, where each score represents `P(moderated | term, subreddit)`. These files contain the **CriteriaMatrix score tables** that enable systematic comparison of moderation patterns across communities.

### Subreddit List Files

Plain text files with one subreddit name per line.

## Usage

For detailed instructions on downloading and using this dataset, including code examples and integration with the analysis pipeline, please refer to the [official code repository](https://github.com/youngwoo-umass/criteria-matrix).

The repository provides:
- Data loading utilities with HuggingFace integration
- Training scripts for PAT models
- Analysis tools for exploring CriteriaMatrix score tables
- Reproduction scripts for paper experiments

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Subreddits** | ~60 communities |
| **Vocabulary Size** | 90K = 10K terms per n-gram size (1-9) |
| **Term Scores** | ~5.4M scores (60 subreddits × 90K terms) |
| **Time Period** | May 2016 - March 2017 |
| **Languages** | English |

## Data Collection

### Source Data

- **Moderated content**: Reddit Removal Log from [Chandrasekharan et al. (2018)](https://zenodo.org/records/3338698)
- **Non-moderated content**: Random samples from Reddit Pushshift dumps via [Academic Torrents](https://academictorrents.com/details/1614740ac8c94505e4ecb9d88be8bed7b6afddd4)

### Processing Pipeline

The dataset was created through a five-stage pipeline:

1. **Dataset Preparation**: Balance moderated and non-moderated samples per subreddit
2. **Model Training**: Train Partial Attention Transformer (PAT) models for each subreddit
3. **Vocabulary Generation**: Extract frequent n-grams, rank by LM probability
4. **Term Scoring**: Apply PAT models to compute moderation probabilities for each term, creating CriteriaMatrix score tables
5. **Analysis**: Cluster terms and analyze moderation patterns

For detailed pipeline documentation, see the [code repository](https://github.com/youngwoo-umass/criteria-matrix).

## Use Cases

This dataset enables research on:

- **Content Moderation**: Understanding what makes content likely to be removed
- **Community Norms**: Comparing moderation patterns across communities using CriteriaMatrix
- **Interpretable AI**: Extracting human-readable moderation criteria
- **Social Computing**: Analyzing governance in online communities
- **Toxicity Detection**: Building context-aware content classifiers

### Example Research Questions

- What terms are consistently moderated across all communities?
- How do political subreddits differ in their moderation criteria?
- Can we identify clusters of communities with similar moderation patterns?
- Which violations are universal vs. community-specific?

## Pre-trained Models

Trained PAT models for each subreddit are available separately at the [model collection](https://huggingface.co/youngwoo-umass).

## Limitations and Ethical Considerations

### Ethical Considerations

**Intended Use**:
- Academic research on content moderation
- Understanding community governance
- Building interpretable moderation tools
- Comparative analysis of online communities

**Not Intended For**:
- Automated content moderation without human oversight
- Censorship or suppression of legitimate speech
- Identifying or targeting specific users
- Creating adversarial attacks against moderation systems

## Citation

```bibtex
@article{kim2025decoding,
  title={Decoding the Rule Book: Extracting Hidden Moderation Criteria from Reddit Communities},
  author={Kim, Youngwoo and Beniwal, Himanshu and Johnson, Steven L and Hartvigsen, Thomas},
  journal={arXiv preprint arXiv:2509.02926},
  year={2025}
}
```

## License

This dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) License, consistent with the original Reddit moderation data from Chandrasekharan et al. (2018).

## Contact

For questions, issues, or feedback:

- **Paper**: [arXiv:2509.02926](https://arxiv.org/abs/2509.02926)
- **Code Repository**: [github.com/youngwoo-umass/criteria-matrix](https://github.com/youngwoo-umass/criteria-matrix)

## Updates and Versions

