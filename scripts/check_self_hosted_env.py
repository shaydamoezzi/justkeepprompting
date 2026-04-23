#!/usr/bin/env python3
from __future__ import annotations

import sys

import torch


def main() -> int:
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device0:", torch.cuda.get_device_name(0))
        a = torch.tensor([1, 2, 3], device="cuda")
        b = a + 1
        print("cuda_smoke:", b.cpu().tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
