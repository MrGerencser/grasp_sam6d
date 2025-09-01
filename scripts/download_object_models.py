#!/usr/bin/env python3
"""
Script to download object models dataset from Hugging Face Hub
"""

from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    """Download object models dataset"""
    
    # Set target directory relative to script location
    script_dir = Path(__file__).parent
    target_dir = script_dir.parent / "Data" / "models"
    
    print(f"Downloading dataset to: {target_dir}")
    
    try:
        # Download dataset directly to target directory
        snapshot_download(
            repo_id="Gerencser/object_dataset",
            repo_type="dataset",
            local_dir=str(target_dir)
        )
        
        print("✅ Download completed successfully!")
        
        # Count downloaded files
        total_files = sum(1 for _ in target_dir.rglob("*") if _.is_file())
        print(f"📁 Total files downloaded: {total_files}")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")


if __name__ == "__main__":
    main()