"""
Hugging Face dataset loader with local file caching.
This module downloads datasets from HF and caches them locally,
maintaining compatibility with existing file-based code.
"""
import os

from huggingface_hub import hf_hub_download

from rule_gen.cpath import data_root_path, output_root_path

HF_REPO_ID = "youngwoo-umass/CriteriaMatrix"

class HFDatasetManager:
    """
    Manages downloading and caching of Hugging Face datasets.
    Downloads occur only once; subsequent calls use cached files.
    """

    def __init__(self, repo_id: str, cache_root: str = None):
        """
        Args:
            repo_id: Your HF repo ID, e.g., "username/reddit-moderation-data"
            cache_root: Root directory for caching. Defaults to project data_root_path
        """
        self.repo_id = repo_id
        self.cache_root = cache_root or data_root_path

    def get_csv_path(self, file_path: str) -> str:
        """
        Download a CSV file from HF and return local path.

        Args:
            file_path: Path within the HF repo, e.g., "reddit/train_data2/askscience/train.csv"

        Returns:
            Local file path to the downloaded CSV
        """
        local_path = os.path.join(self.cache_root, file_path)

        # Return if already cached
        if os.path.exists(local_path):
            return local_path

        # Download from HF
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        downloaded_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=file_path,
            repo_type="dataset",
            cache_dir=None,  # Uses default HF cache
        )

        # Copy to expected location (or create symlink)
        import shutil
        shutil.copy2(downloaded_path, local_path)

        return local_path

    def get_pickle_path(self, file_path: str) -> str:
        """
        Download a pickle file from HF and return local path.

        Args:
            file_path: Path within the HF repo

        Returns:
            Local file path to the downloaded pickle file
        """
        return self.get_csv_path(file_path)  # Same logic

    def get_model_path(self, model_name: str) -> str:
        """
        Download a model from HF and return local path.

        Args:
            model_name: Model identifier in HF repo

        Returns:
            Local directory path containing the model files
        """
        local_path = os.path.join(output_root_path, "models", model_name)

        if os.path.exists(os.path.join(local_path, "config.json")):
            return local_path

        # Download all model files
        os.makedirs(local_path, exist_ok=True)

        # Common model files
        model_files = [
            "config.json",
            "model.safetensors",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
        ]

        for filename in model_files:
            try:
                file_path = f"models/{model_name}/{filename}"
                downloaded = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=file_path,
                    repo_type="dataset",
                )
                import shutil
                shutil.copy2(downloaded, os.path.join(local_path, filename))
            except Exception as e:
                print(f"Warning: Could not download {filename}: {e}")

        return local_path


# Global instance - initialize once in your main script
_hf_manager = None



def get_hf_manager() -> HFDatasetManager:
    """Get the global HF dataset manager."""
    global _hf_manager
    if _hf_manager is None:
        _hf_manager = HFDatasetManager(HF_REPO_ID)
    return _hf_manager