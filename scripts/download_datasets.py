#!/usr/bin/env python3
"""
Dataset download scripts for Traffic-Eye AI.

Downloads datasets from public sources (Kaggle CLI, Roboflow API).
Kaggle requires ~/.kaggle/kaggle.json credentials.
Roboflow requires a ROBOFLOW_API_KEY environment variable.
If neither is available, prints manual instructions.

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --helmet
    python scripts/download_datasets.py --seatbelt
    python scripts/download_datasets.py --plates
    python scripts/download_datasets.py --demo-video
"""

import os
import sys
import argparse
import subprocess
import zipfile
import tarfile
import shutil
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'


def download_file(url, dest_path, desc=""):
    """Download a file with progress reporting."""
    print(f"\n📥 Downloading {desc or url}...")
    print(f"   → {dest_path}")

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r   Progress: {pct:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
        print(f"\n   ✅ Downloaded successfully!")
        return True
    except Exception as e:
        print(f"\n   ❌ Download failed: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"📦 Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
        print(f"   ✅ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        return False


def download_helmet_dataset():
    """
    Download helmet detection dataset.
    
    Primary: Kaggle Safety Helmet Detection dataset
    Fallback: Manual instructions for Roboflow download
    """
    dest_dir = DATA_DIR / 'helmet'
    os.makedirs(dest_dir, exist_ok=True)

    print("\n" + "="*60)
    print("🪖 HELMET DETECTION DATASET")
    print("="*60)

    # Try Kaggle CLI first
    print("\nAttempting Kaggle download...")
    kaggle_dataset = "andrewmvd/helmet-detection"
    zip_path = dest_dir / 'helmet_dataset.zip'

    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', kaggle_dataset,
             '-p', str(dest_dir), '--force'],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            # Find and extract the downloaded zip
            for f in dest_dir.glob('*.zip'):
                extract_zip(str(f), str(dest_dir))
                os.remove(f)
            print("   ✅ Helmet dataset downloaded via Kaggle!")
            return True
        else:
            print(f"   Kaggle CLI failed: {result.stderr[:200]}")
    except FileNotFoundError:
        print("   Kaggle CLI not found.")
    except Exception as e:
        print(f"   Kaggle download error: {e}")

    # ── Attempt 2: Roboflow API ──────────────────────────────────
    # Dataset: "Motorcycle Helmet Detection" by data-science-173-msp9r
    # URL: https://universe.roboflow.com/data-science-173-msp9r/motorcycle-helmet-detection
    roboflow_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if roboflow_key:
        print("\nAttempting Roboflow download...")
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=roboflow_key)
            project = rf.workspace("data-science-173-msp9r").project("motorcycle-helmet-detection")
            version = project.version(1)
            version.download("yolov8", location=str(dest_dir))
            print("   ✅ Helmet dataset downloaded via Roboflow!")
            return True
        except ImportError:
            print("   'roboflow' package not installed. Run: pip install roboflow")
        except Exception as e:
            print(f"   Roboflow download failed: {e}")
    else:
        print("\n   ROBOFLOW_API_KEY not set — skipping automated Roboflow download. See manual instructions below.")

    # ── Fallback: Manual instructions ────────────────────────────
    print("\n" + "-"*40)
    print("📋 MANUAL DOWNLOAD INSTRUCTIONS:")
    print("-"*40)
    print("1. Go to: https://universe.roboflow.com/")
    print("   Search: 'helmet detection motorcycle'")
    print("2. Pick a dataset with >3000 images")
    print("3. Click 'Download Dataset' → Format: 'YOLOv8' → 'download zip'")
    print(f"4. Extract to: {dest_dir}/")
    print("   Expected structure:")
    print("   data/helmet/train/images/  +  data/helmet/train/labels/")
    print("   data/helmet/valid/images/  +  data/helmet/valid/labels/")
    print("")
    print("   OR use Kaggle:")
    print("   kaggle datasets download -d andrewmvd/helmet-detection")
    print("-"*40)

    # Create a placeholder data.yaml for when data is downloaded
    yaml_content = """# Helmet Detection Dataset
# Update paths after downloading

train: train/images
val: valid/images
test: test/images

nc: 2
names: ['helmet', 'no-helmet']
"""
    yaml_path = dest_dir / 'data.yaml'
    if not yaml_path.exists():
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"   Created template data.yaml at {yaml_path}")

    return False


def download_seatbelt_dataset():
    """
    Download seatbelt detection dataset.
    
    Note: Seatbelt datasets are small (~250-544 images).
    We'll create a template and provide download instructions.
    """
    dest_dir = DATA_DIR / 'seatbelt'
    os.makedirs(dest_dir, exist_ok=True)

    print("\n" + "="*60)
    print("🔒 SEATBELT DETECTION DATASET")
    print("="*60)

    # Try Kaggle
    print("\nAttempting Kaggle download...")
    kaggle_dataset = "shantanuss/seatbelt-detection"

    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', kaggle_dataset,
             '-p', str(dest_dir), '--force'],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            for f in dest_dir.glob('*.zip'):
                extract_zip(str(f), str(dest_dir))
                os.remove(f)
            print("   ✅ Seatbelt dataset downloaded via Kaggle!")
            return True
        else:
            print(f"   Kaggle CLI: {result.stderr[:200]}")
    except FileNotFoundError:
        print("   Kaggle CLI not found.")
    except Exception as e:
        print(f"   Error: {e}")

    # ── Attempt 2: Roboflow API ──────────────────────────────────
    # Dataset: "Seatbelt Detection" by traffic-violations
    # URL: https://universe.roboflow.com/traffic-violations/seatbelt-detection-esut6
    roboflow_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if roboflow_key:
        print("\nAttempting Roboflow download...")
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=roboflow_key)
            project = rf.workspace("traffic-violations").project("seatbelt-detection-esut6")
            version = project.version(1)
            version.download("yolov8", location=str(dest_dir))
            print("   ✅ Seatbelt dataset downloaded via Roboflow!")
            return True
        except ImportError:
            print("   'roboflow' package not installed. Run: pip install roboflow")
        except Exception as e:
            print(f"   Roboflow download failed: {e}")
    else:
        print("\n   ROBOFLOW_API_KEY not set — skipping automated Roboflow download. See manual instructions below.")

    # ── Fallback: Manual instructions ────────────────────────────
    print("\n" + "-"*40)
    print("📋 MANUAL DOWNLOAD INSTRUCTIONS:")
    print("-"*40)
    print("1. Go to: https://universe.roboflow.com/")
    print("   Search: 'seatbelt detection'")
    print("2. Pick the 'Seatbelt Detection' dataset by Traffic Violations")
    print("3. Download in YOLOv8 format")
    print(f"4. Extract to: {dest_dir}/")
    print("")
    print("   ⚠️  NOTE: Seatbelt datasets are small (~500 images).")
    print("   Use Roboflow augmentation during export for best results.")
    print("-"*40)

    yaml_content = """# Seatbelt Detection Dataset
# Update paths after downloading

train: train/images
val: valid/images
test: test/images

nc: 2
names: ['seatbelt', 'no-seatbelt']
"""
    yaml_path = dest_dir / 'data.yaml'
    if not yaml_path.exists():
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

    return False


def download_plates_dataset():
    """
    Download license plate detection dataset.
    
    Primary: Kaggle Large License Plate Detection Dataset (~27,900 images)
    """
    dest_dir = DATA_DIR / 'plates'
    os.makedirs(dest_dir, exist_ok=True)

    print("\n" + "="*60)
    print("🔢 LICENSE PLATE DETECTION DATASET")
    print("="*60)

    # Try Kaggle - large dataset
    print("\nAttempting Kaggle download...")
    kaggle_dataset = "fareselmenshawii/large-license-plate-dataset"

    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', kaggle_dataset,
             '-p', str(dest_dir), '--force'],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            for f in dest_dir.glob('*.zip'):
                extract_zip(str(f), str(dest_dir))
                os.remove(f)
            print("   ✅ License plate dataset downloaded via Kaggle!")
            return True
        else:
            print(f"   Kaggle CLI: {result.stderr[:200]}")
    except FileNotFoundError:
        print("   Kaggle CLI not found.")
    except Exception as e:
        print(f"   Error: {e}")

    # ── Attempt 2: Roboflow API ──────────────────────────────────
    # Dataset: "License Plates" by samrat-sahoo
    # URL: https://universe.roboflow.com/samrat-sahoo/license-plates-f8vsn
    roboflow_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if roboflow_key:
        print("\nAttempting Roboflow download...")
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=roboflow_key)
            project = rf.workspace("samrat-sahoo").project("license-plates-f8vsn")
            version = project.version(1)
            version.download("yolov8", location=str(dest_dir))
            print("   ✅ License plate dataset downloaded via Roboflow!")
            return True
        except ImportError:
            print("   'roboflow' package not installed. Run: pip install roboflow")
        except Exception as e:
            print(f"   Roboflow download failed: {e}")
    else:
        print("\n   ROBOFLOW_API_KEY not set — skipping automated Roboflow download. See manual instructions below.")

    # ── Fallback: Manual instructions ────────────────────────────
    print("\n" + "-"*40)
    print("📋 MANUAL DOWNLOAD INSTRUCTIONS:")
    print("-"*40)
    print("1. Go to: https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset")
    print("2. Download the dataset (requires Kaggle account)")
    print(f"3. Extract to: {dest_dir}/")
    print("")
    print("   Alternative: Search Roboflow for 'license plate detection'")
    print("   and download in YOLOv8 format.")
    print("-"*40)

    yaml_content = """# License Plate Detection Dataset
# Update paths after downloading

train: train/images
val: valid/images
test: test/images

nc: 1
names: ['license-plate']
"""
    yaml_path = dest_dir / 'data.yaml'
    if not yaml_path.exists():
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

    return False


def download_demo_video():
    """
    Download demo traffic video from Pexels (free license).
    """
    dest_dir = DATA_DIR / 'test_videos'
    os.makedirs(dest_dir, exist_ok=True)

    print("\n" + "="*60)
    print("🎬 DEMO TRAFFIC VIDEO")
    print("="*60)

    # Pexels free traffic videos (direct download links)
    videos = [
        {
            'url': 'https://videos.pexels.com/video-files/2053100/2053100-sd_640_360_30fps.mp4',
            'filename': 'traffic_demo_1.mp4',
            'description': 'Urban traffic intersection (Pexels, free license)',
        },
        {
            'url': 'https://videos.pexels.com/video-files/1448735/1448735-sd_640_360_24fps.mp4',
            'filename': 'traffic_demo_2.mp4',
            'description': 'Busy road traffic (Pexels, free license)',
        },
    ]

    success = False
    for video in videos:
        dest_path = dest_dir / video['filename']
        if dest_path.exists():
            print(f"   ✅ {video['filename']} already exists, skipping.")
            success = True
            continue

        result = download_file(
            video['url'],
            str(dest_path),
            desc=video['description']
        )
        if result:
            success = True

    if not success:
        print("\n   ⚠️  Could not auto-download demo videos.")
        print("   Please download manually from:")
        print("   https://www.pexels.com/search/videos/traffic/")
        print(f"   Save to: {dest_dir}/")

    return success


def main():
    parser = argparse.ArgumentParser(description="Download datasets for Traffic-Eye AI")
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    parser.add_argument('--helmet', action='store_true', help='Download helmet dataset')
    parser.add_argument('--seatbelt', action='store_true', help='Download seatbelt dataset')
    parser.add_argument('--plates', action='store_true', help='Download license plate dataset')
    parser.add_argument('--demo-video', action='store_true', help='Download demo video')
    args = parser.parse_args()

    if not any([args.all, args.helmet, args.seatbelt, args.plates, args.demo_video]):
        args.all = True

    print("╔══════════════════════════════════════════╗")
    print("║   Traffic-Eye AI — Dataset Downloader    ║")
    print("╚══════════════════════════════════════════╝")

    results = {}

    if args.all or args.helmet:
        results['helmet'] = download_helmet_dataset()

    if args.all or args.seatbelt:
        results['seatbelt'] = download_seatbelt_dataset()

    if args.all or args.plates:
        results['plates'] = download_plates_dataset()

    if args.all or args.demo_video:
        results['demo_video'] = download_demo_video()

    # Summary
    print("\n" + "="*60)
    print("📊 DOWNLOAD SUMMARY")
    print("="*60)
    for name, success in results.items():
        status = "✅ Downloaded" if success else "⚠️  Manual download needed"
        print(f"   {name:15s}: {status}")
    print("="*60)


if __name__ == '__main__':
    main()
