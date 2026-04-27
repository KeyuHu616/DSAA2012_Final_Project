#!/usr/bin/env python3
"""Cleanup incomplete downloads in HuggingFace cache"""

import os
import sys
from pathlib import Path

# Unset proxy
for var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(var, None)

# Add project to path for importing mirror_config
sys.path.insert(0, str(Path(__file__).parent))
from storygen.utils.mirror_config import get_models_cache_dir

# Use project ./models directory
cache_dir = get_models_cache_dir()
print(f"[Cleanup] Using cache directory: {cache_dir}")

if not cache_dir.exists():
    print('Cache directory not found')
    exit()

results = {'deleted': 0, 'freed_mb': 0, 'warnings': []}

print("=" * 70)
print("🧹 Scanning for incomplete downloads...")
print("=" * 70)

for model_dir in sorted(cache_dir.glob('models--*')):
    blobs_dir = model_dir / 'blobs'
    if not blobs_dir.exists():
        continue
    
    model_name = model_dir.name.replace('models--', '').replace('--', '/')
    
    # Check for .incomplete files
    for f in blobs_dir.glob('*.incomplete'):
        try:
            size = f.stat().st_size
            freed = size / (1024 * 1024)
            status = "empty" if size == 0 else f"{freed:.1f} MB"
            print(f"   ⚠️  Deleting: {f.name[:30]}... [{status}]")
            f.unlink()
            results['deleted'] += 1
            results['freed_mb'] += freed
        except FileNotFoundError:
            pass  # Already deleted
        except Exception as e:
            results['warnings'].append(str(e))
    
    # Check for 0-byte files
    for f in blobs_dir.iterdir():
        try:
            if f.is_file() and f.stat().st_size == 0:
                print(f"   ⚠️  Deleting empty file: {f.name[:30]}...")
                f.unlink()
                results['deleted'] += 1
        except FileNotFoundError:
            pass
        except Exception as e:
            results['warnings'].append(str(e))

print("\n" + "-" * 70)
print(f"\n[Cleanup] Summary:")
print(f"   Files deleted: {results['deleted']}")
print(f"   Space freed: {results['freed_mb']:.1f} MB")
if results['warnings']:
    print(f"   Warnings: {len(results['warnings'])}")
