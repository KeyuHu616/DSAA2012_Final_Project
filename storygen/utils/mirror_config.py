"""
Enhanced HF Mirror Configuration - Complete China Mirror Solution
- Multiple mirror sources for reliability
- Incomplete download cleanup
- Model integrity verification
"""

import os
from pathlib import Path


# Project-level model cache directory (unified for all pretrained models)
PROJECT_BASE_DIR = Path(__file__).parent.parent.parent  # Project root
MODELS_CACHE_DIR = PROJECT_BASE_DIR / "models"  # ./models directory

# Set environment variables for all ML libraries to use project cache
def configure_all_cache_dirs():
    """Configure cache directories for all ML libraries"""
    cache_path = str(MODELS_CACHE_DIR)

    # HuggingFace
    os.environ.setdefault("HF_HOME", cache_path)

    # Torch (for some models)
    os.environ.setdefault("TORCH_HOME", cache_path)

    # OpenCLIP (uses HF_HOME)
    os.environ.setdefault("HF_HUB_CACHE", cache_path)

    # Ensure directory exists
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_models_cache_dir() -> Path:
    """
    Get unified project model cache directory.
    Auto-creates the directory if it doesn't exist.

    Returns:
        Path: Absolute path to ./models directory
    """
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_CACHE_DIR


def setup_china_mirrors():
    """
    Configure model sources for optimal download in China
    
    Strategy:
    1. Use original HuggingFace Hub (most reliable)
    2. Mirror fallback if needed
    """
    # Use original HuggingFace - most stable
    os.environ.pop("HF_ENDPOINT", None)  # Remove mirror setting
    
    # Enable hf-transfer for faster downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    # Set reasonable timeout
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes per file
    
    # Keep progress bars for visibility
    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    
    print("[Mirror] HuggingFace Hub configured (direct connection)")
    print(f"   • Timeout: {os.environ['HF_HUB_DOWNLOAD_TIMEOUT']}s")
    print(f"   • hf-transfer: enabled")


def cleanup_incomplete_downloads(
    cache_dir: str = None,
    dry_run: bool = False
) -> dict:
    """
    Find and clean up incomplete/in corrupted downloads

    Args:
        cache_dir: HuggingFace cache directory (default: ./models)
        dry_run: If True, only show what would be deleted

    Returns:
        dict with statistics about cleaned files
    """
    if cache_dir is None:
        cache_dir = get_models_cache_dir()  # Use project ./models directory
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        return {"deleted": [], "freed_space_mb": 0}

    results = {"deleted": [], "freed_space_mb": 0}

    for model_dir in cache_dir.glob("models--*"):
        blobs_dir = model_dir / "blobs"
        if not blobs_dir.exists():
            continue

        for blob_file in blobs_dir.iterdir():
            try:
                if blob_file.suffix == '.incomplete' or blob_file.stat().st_size == 0:
                    size_mb = blob_file.stat().st_size / (1024 * 1024)
                    if not dry_run:
                        blob_file.unlink()
                    results["deleted"].append(str(blob_file))
                    results["freed_space_mb"] += size_mb
            except FileNotFoundError:
                pass

    return results


def verify_model_integrity(
    model_name: str,
    cache_dir: str = None
) -> bool:
    """
    Verify if a model is completely downloaded without incomplete files

    Args:
        model_name: Model name (e.g., "stabilityai/stable-diffusion-xl-base-1.0")
        cache_dir: Custom cache directory path (default: ./models)

    Returns:
        bool: True if model is complete and usable
    """
    if cache_dir is None:
        cache_dir = get_models_cache_dir()  # Use project ./models directory
    cache_dir = Path(cache_dir)

    model_cache_path = cache_dir / f"models--{model_name.replace('/', '--')}"

    if not model_cache_path.exists():
        return False

    # Check for incomplete files
    blobs_dir = model_cache_path / "blobs"
    if blobs_dir.exists():
        for f in blobs_dir.iterdir():
            try:
                if f.suffix == '.incomplete' or f.stat().st_size == 0:
                    return False
            except FileNotFoundError:
                return False

    # Check snapshots exist
    snapshots_dir = model_cache_path / "snapshots"
    if not snapshots_dir.exists():
        return False

    snapshot_dirs = list(snapshots_dir.iterdir())
    if len(snapshot_dirs) == 0:
        return False

    return True


def get_all_cached_models_status(cache_dir: str = None) -> dict:
    """Get status of all cached models"""
    if cache_dir is None:
        cache_dir = get_models_cache_dir()  # Use project ./models directory
    cache_dir = Path(cache_dir)

    status = {}

    for model_dir in sorted(cache_dir.glob("models--*")):
        model_name = model_dir.name.replace("models--", "").replace("--", "/")
        is_complete = verify_model_integrity(model_name, cache_dir)

        total_size = sum(
            f.stat().st_size
            for f in model_dir.rglob('*')
            if f.is_file()
        ) / (1024**3)

        status[model_name] = {
            'complete': is_complete,
            'size_gb': round(total_size, 2),
            'path': str(model_dir)
        }

    return status


def print_model_status_report():
    """Print formatted report of all cached models"""
    status = get_all_cached_models_status()

    print("\n" + "=" * 80)
    print("📊 MODEL CACHE STATUS REPORT")
    print("=" * 80)

    complete_count = sum(1 for v in status.values() if v['complete'])
    incomplete_count = len(status) - complete_count

    print(f"\nTotal models: {len(status)} | ✅ Complete: {complete_count} | ⚠️ Incomplete: {incomplete_count}\n")

    print(f"{'Model Name':<55} {'Status':<12} {'Size':<8}")
    print("-" * 80)

    for model_name, info in sorted(status.items()):
        status_icon = "✅" if info['complete'] else "⚠️ "
        status_text = "COMPLETE" if info['complete'] else "INCOMPLETE"
        print(f"{model_name:<55} {status_icon:<2} {status_text:<10} {info['size_gb']:>6.1f} GB")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    setup_china_mirrors()

    parser = argparse.ArgumentParser(description="HuggingFace Cache Manager")
    parser.add_argument('--status', action='store_true', help='Show model status report')
    parser.add_argument('--cleanup', action='store_true', help='Clean incomplete downloads')

    args = parser.parse_args()

    if args.status:
        print_model_status_report()

    if args.cleanup:
        cleanup_incomplete_downloads()
