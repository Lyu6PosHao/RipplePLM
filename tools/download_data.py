#!/usr/bin/env python3
"""Download the RipplePLM dataset from Hugging Face."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data')
    args = parser.parse_args()
    snapshot_download(
        repo_id='GreatCaptainNemo/RipplePLM-data',
        repo_type='dataset',
        local_dir=str(args.output_dir),
        allow_patterns=['structural_split/*.csv', 'temporal_split/*.csv'],
    )


if __name__ == '__main__':
    main()
