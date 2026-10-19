"""Small shared helpers: logging, output directory, command-line arguments."""
import argparse
import os
import time

_LOG = None
_T0 = time.time()


def parse_args(extra=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("CGMACROS_DIR", "data/cgmacros"),
                    help="folder with the CGMacros PhysioNet files (the folder that contains "
                         "CGMacros_dateshifted365/CGMacros, or the CGMacros folder itself)")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--cache", default="cache", help="cache directory (image embeddings)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="quick test: 12 participants, few iterations")
    if extra:
        extra(ap)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    return args


def open_log(out_dir, name):
    global _LOG, _T0
    _LOG = open(os.path.join(out_dir, name), "a", encoding="utf-8")
    _T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if _LOG is not None:
        _LOG.write(s + "\n")
        _LOG.flush()


def elapsed():
    return round(time.time() - _T0)
