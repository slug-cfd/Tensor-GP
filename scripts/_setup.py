"""Puts the repository root on sys.path, so that the scripts run without installing gptensor, and
defines the output folder for files the scripts write."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, 'output')
os.makedirs(OUT, exist_ok=True)
