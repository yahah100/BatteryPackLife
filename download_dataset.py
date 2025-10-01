from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError
import os
import time
import glob
import shutil

# Set comprehensive timeout and retry settings
os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120" 
os.environ["HF_HUB_DOWNLOAD_RETRIES"] = "5"
os.environ["REQUESTS_TIMEOUT"] = "120"

def clean_incomplete_files():
    """Remove corrupted incomplete files"""
    incomplete_files = glob.glob(".cache/datasets--Battery-Life--BatteryLife_Processed/blobs/*.incomplete")
    if incomplete_files:
        print(f"Removing {len(incomplete_files)} incomplete files...")
        for file in incomplete_files:
            try:
                os.remove(file)
                print(f"Removed: {os.path.basename(file)}")
            except OSError:
                pass

def download_with_retry(max_retries=5):
    hf_token = os.getenv("HF_TOKEN", False)
    if not hf_token:
        raise ValueError("Warning: HF_TOKEN environment variable is not set. Proceeding without authentication may lead to rate limiting.")

    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt + 1}/{max_retries}")
            snapshot_download(
                repo_id="Battery-Life/BatteryLife_Processed",
                repo_type="dataset", 
                cache_dir=".cache/",
                token=hf_token,
                force_download=False,
                max_workers=1  # Use single worker to avoid race conditions
            )
            print("Download completed successfully!")
            return
        except HfHubHTTPError as e:
            if "416" in str(e):
                print(f"Range error detected (416): {e}")
                print("Cleaning incomplete files and retrying...")
                clean_incomplete_files()
                # Don't count 416 errors against retry limit
                continue
            else:
                print(f"HTTP error: {e}")
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print("All retry attempts failed!")
                    raise
        except Exception as e:
            error_str = str(e)
            print(f"Attempt {attempt + 1} failed: {e}")
            
            # Handle consistency check failures
            if "Consistency check failed" in error_str:
                print("Corrupted file detected. Cleaning cache and using force download...")
                clean_incomplete_files()
                # Force download on consistency failures
                try:
                    print("Attempting force download for corrupted files...")
                    snapshot_download(
                        repo_id="Battery-Life/BatteryLife_Processed",
                        repo_type="dataset", 
                        cache_dir=".cache/",
                        token=hf_token,
                        force_download=True,  # Force re-download corrupted files
                        max_workers=1
                    )
                    print("Force download completed successfully!")
                    return
                except Exception as force_e:
                    print(f"Force download also failed: {force_e}")
                    
            if attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print("All retry attempts failed!")
                raise

if __name__ == "__main__":
    # Clean any existing incomplete files first
    clean_incomplete_files()
    download_with_retry()