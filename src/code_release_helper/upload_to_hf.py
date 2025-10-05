"""
Script to upload your datasets to Hugging Face.
Provides flexible upload options for different parts of the dataset.
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import fire

from rule_gen.cpath import data_root_path, output_root_path


class HFUploader:
    """Handles uploading datasets to HuggingFace."""


    def __init__(self, repo_id: str = "youngwoo-umass/CriteriaMatrix", token: str = None):
        """
        Initialize uploader.

        Args:
            repo_id: HF repository ID
            token: HF token (or set HF_TOKEN environment variable)
        """
        self.repo_id = repo_id
        self.api = HfApi(token=token)


        # Check if repo exists, create if not
        print(self.api.repo_info(repo_id=self.repo_id, repo_type="dataset"))


    def upload_training_data(self, subreddit: str = None):
        """
        Upload training data for one or all subreddits.

        Args:
            subreddit: Specific subreddit name, or None for all
        """
        train_data_root = os.path.join(output_root_path, "reddit", "train_data2")

        if not os.path.exists(train_data_root):
            print(f"❌ Training data not found at: {train_data_root}")
            return

        if subreddit:
            # Upload single subreddit
            subreddit_path = os.path.join(train_data_root, subreddit)
            if not os.path.exists(subreddit_path):
                print(f"❌ Subreddit not found: {subreddit}")
                return

            print(f"Uploading {subreddit}...")
            self.api.upload_folder(
                folder_path=subreddit_path,
                path_in_repo=f"reddit/train_data2/{subreddit}",
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            print(f"✓ Uploaded {subreddit}")
        else:
            # Upload all subreddits
            print("Uploading all training data...")
            self.api.upload_folder(
                folder_path=train_data_root,
                path_in_repo="reddit/train_data2",
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            print("✓ Uploaded all training data")

    def upload_subreddit_lists(self):
        """Upload subreddit split lists."""
        print("Uploading subreddit lists...")

        for split in ["train", "val", "test"]:
            list_path = os.path.join(data_root_path, "reddit", f"subreddits_{split}.csv")
            if os.path.exists(list_path):
                self.api.upload_file(
                    path_or_fileobj=list_path,
                    path_in_repo=f"reddit/subreddits_{split}.csv",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                )
                print(f"✓ Uploaded subreddits_{split}.csv")
            else:
                print(f"⚠ Not found: subreddits_{split}.csv")

        self.api.upload_file(
            path_or_fileobj=list_path,
            path_in_repo=f"reddit/study-subreddits.csv",
            repo_id=self.repo_id,
            repo_type="dataset",
        )
    def upload_removal_log(self):
        """Upload reddit removal log."""
        removal_log = os.path.join(data_root_path, "reddit", "reddit-removal-log.csv")

        if not os.path.exists(removal_log):
            print(f"⚠ Removal log not found at: {removal_log}")
            return

        print("Uploading removal log...")
        self.api.upload_file(
            path_or_fileobj=removal_log,
            path_in_repo="reddit/reddit-removal-log.csv",
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        print("✓ Uploaded reddit-removal-log.csv")

    def upload_intermediate_outputs(self, subdir: str = None):
        """
        Upload intermediate processing outputs (vocab, scores, etc.).

        Args:
            subdir: Specific subdirectory (e.g., 'top_10k_voca'), or None for all
        """
        processing_root = os.path.join(output_root_path, "reddit", "rule_processing")

        if not os.path.exists(processing_root):
            print(f"⚠ Processing outputs not found at: {processing_root}")
            return

        if subdir:
            # Upload specific subdirectory
            subdir_path = os.path.join(processing_root, subdir)
            if not os.path.exists(subdir_path):
                print(f"❌ Subdirectory not found: {subdir}")
                return

            print(f"Uploading {subdir}...")
            self.api.upload_folder(
                folder_path=subdir_path,
                path_in_repo=f"reddit/rule_processing/{subdir}",
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            print(f"✓ Uploaded {subdir}")
        else:
            # Upload all intermediate outputs
            print("Uploading all intermediate outputs...")
            self.api.upload_folder(
                folder_path=processing_root,
                path_in_repo="reddit/rule_processing",
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            print("✓ Uploaded all intermediate outputs")

    def upload_models(self, model_name: str = None):
        """
        Upload trained models (optional - consider separate model repos).

        Args:
            model_name: Specific model name, or None for all
        """
        models_root = os.path.join(output_root_path, "models")

        if not os.path.exists(models_root):
            print(f"⚠ Models not found at: {models_root}")
            return

        if model_name:
            # Upload specific model
            model_path = os.path.join(models_root, model_name)
            if not os.path.exists(model_path):
                print(f"❌ Model not found: {model_name}")
                return

            print(f"Uploading model {model_name}...")
            self.api.upload_folder(
                folder_path=model_path,
                path_in_repo=f"models/{model_name}",
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            print(f"✓ Uploaded {model_name}")
        else:
            print("⚠ Uploading all models - consider using separate model repositories")
            print("Use upload_models --model_name=<name> to upload specific models")

    def upload_all(self):
        """Upload everything except models."""
        print("=" * 60)
        print("Starting full dataset upload")
        print("=" * 60)

        self.upload_subreddit_lists()
        print()

        self.upload_removal_log()
        print()

        self.upload_intermediate_outputs()
        print()

        print("=" * 60)
        print(f"✅ Upload complete!")
        print(f"View at: https://huggingface.co/datasets/{self.repo_id}")
        print("=" * 60)

    def create_readme(self):
        """Create or update README.md for the dataset."""
        readme_content = f"""---
license: mit
task_categories:
- text-classification
language:
- en
tags:
- reddit
- content-moderation
- toxicity-detection
size_categories:
- 10K<n<100K
---

# CriteriaMatrix: Reddit Content Moderation Dataset

This dataset contains training data for Reddit content moderation across multiple subreddits, 
including binary classification labels for removed vs. kept content.

## Dataset Structure

```
CriteriaMatrix/
├── reddit/
│   ├── train_data2/              # Training datasets by subreddit
│   │   ├── askscience/
│   │   │   ├── train.csv
│   │   │   ├── val.csv
│   │   │   └── test.csv
│   │   ├── TwoXChromosomes/
│   │   │   └── ...
│   │   └── ... (multiple subreddits)
│   ├── subreddits_train.csv      # List of training subreddits
│   ├── subreddits_val.csv        # List of validation subreddits
│   ├── reddit-removal-log.csv    # Raw removal log data
│   └── rule_processing/          # Intermediate outputs
│       ├── top_10k_voca/         # Top vocabulary n-grams
│       └── sb_term_scores/       # Term scores per subreddit
└── models/                       # (Optional) Pre-trained models
```

## Quick Start

### Installation

```bash
pip install transformers datasets huggingface-hub fire
```

### Usage

```python
from rule_gen.hf_dataset_loader import init_hf_manager

# Initialize HF dataset manager
init_hf_manager("youngwoo-umass/CriteriaMatrix")

# Train a classifier (automatically downloads data)
from rule_gen.reddit.base_bert.train2 import train_subreddit_classifier
train_subreddit_classifier(sb="askscience")
```

The code will automatically download and cache required files from HuggingFace.

## Data Format

### Training CSVs
Each CSV file contains two columns (no header):
- **Column 0 (text)**: The comment/post text
- **Column 1 (label)**: Binary label (0=keep, 1=remove)

### Subreddit Lists
CSV files listing subreddit names for train/val splits.

## Dataset Statistics

- **Subreddits**: Multiple communities with varying moderation policies
- **Split**: Train/Val/Test splits provided per subreddit
- **Task**: Binary classification (content should be removed vs. kept)

## Code Repository

The complete code for training models on this dataset is available at:
[Your GitHub Repository URL]

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{{criteriamatrix2024,
  author = {{Youngwoo Kim}},
  title = {{CriteriaMatrix: Reddit Content Moderation Dataset}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/datasets/youngwoo-umass/CriteriaMatrix}}}}
}}
```

## License

This dataset is released under the MIT License.

## Contact

For questions or issues, please open an issue on the dataset repository or contact:
- Email: [Your Email]
- GitHub: [Your GitHub]
"""

        print("Creating README...")
        self.api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        print("✓ Created/updated README.md")


def main():
    """Main CLI interface."""
    token_path = os.path.join(data_root_path, "hf_token.txt")
    token = open(token_path, "r").read().strip()
    uploader = HFUploader(token=token)
    fire.Fire(uploader)

#
if __name__ == "__main__":
    """
    Usage examples:
    
    # Upload everything
    python upload_to_hf.py upload_all
    
    # Upload specific components
    python upload_to_hf.py upload_training_data
    python upload_to_hf.py upload_training_data --subreddit=askscience
    python upload_to_hf.py upload_subreddit_lists
    python upload_to_hf.py upload_removal_log
    python upload_to_hf.py upload_intermediate_outputs
    python upload_to_hf.py upload_intermediate_outputs --subdir=top_10k_voca
    
    # Upload models (use sparingly, consider separate repos)
    python upload_to_hf.py upload_models --model_name=bert2_askscience
    
    # Create/update README
    python upload_to_hf.py create_readme
    
    # Use custom repo
    python upload_to_hf.py --repo_id="username/custom-repo" upload_all
    """
    main()