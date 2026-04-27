"""
deploy_hf_space.py

Deploys hf_space/ to bishoygaloaa/gaslight-turing-test on HuggingFace Spaces.
Adds smoezzi as a collaborator with write access.

Prerequisites:
  1. Generate a HuggingFace token with WRITE access:
       https://huggingface.co/settings/tokens → New token → Role: Write
  2. Run this script:
       HF_TOKEN=hf_xxx python3 scripts/deploy_hf_space.py
     Or login first:
       huggingface-cli login --token hf_xxx
       python3 scripts/deploy_hf_space.py

Run from the repo root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, upload_folder, add_space_member
except ImportError:
    print("huggingface_hub not found. Install with: pip install huggingface_hub")
    sys.exit(1)

REPO_ID = "bishoygaloaa/gaslight-turing-test"
COLLABORATOR = "smoezzi"
HF_SPACE_DIR = Path(__file__).resolve().parent.parent / "hf_space"

token = os.environ.get("HF_TOKEN") or None  # falls back to cached login


def main() -> None:
    api = HfApi(token=token)

    # Verify auth
    me = api.whoami()
    role = me.get("auth", {}).get("accessToken", {}).get("role", "unknown")
    print(f"Logged in as: {me['name']}  (token role: {role})")
    if role == "read":
        print(
            "\nERROR: Your token is read-only.\n"
            "Generate a Write token at: https://huggingface.co/settings/tokens\n"
            "Then re-run: HF_TOKEN=hf_... python3 scripts/deploy_hf_space.py"
        )
        sys.exit(1)

    # Create or ensure Space exists
    print(f"\nCreating Space: {REPO_ID} …")
    try:
        url = api.create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True,
        )
        print(f"  Space URL: {url}")
    except Exception as e:
        print(f"  create_repo error (may already exist): {e}")

    # Upload files
    print(f"\nUploading {HF_SPACE_DIR} …")
    upload_folder(
        repo_id=REPO_ID,
        repo_type="space",
        folder_path=str(HF_SPACE_DIR),
        commit_message="Deploy Gaslight Turing Test leaderboard + run explorer",
        token=token,
        ignore_patterns=["__pycache__", "*.pyc", ".DS_Store"],
    )
    print("  Upload complete.")

    # Add collaborator
    print(f"\nAdding {COLLABORATOR} as collaborator …")
    try:
        api.add_space_member(
            repo_id=REPO_ID,
            user=COLLABORATOR,
            role="write",
            token=token,
        )
        print(f"  {COLLABORATOR} added with write access.")
    except Exception as e:
        print(f"  Could not add collaborator automatically: {e}")
        print(
            f"  Add manually at: https://huggingface.co/spaces/{REPO_ID}/settings"
            " → Members → Add member → smoezzi"
        )

    print(f"\nDone! View your Space at: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
