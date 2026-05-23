"""CLI: compute persistence diagrams for all volumes in a directory."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Directory of CT volumes")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for diagrams")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
