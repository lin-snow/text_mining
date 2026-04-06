#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令行：仅执行 RSS 采集并落盘（便于复现实验）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hw1_pipeline import collect_feeds  # noqa: E402

if __name__ == "__main__":
    collect_feeds()
