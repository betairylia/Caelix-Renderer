"""Image metrics of a test image against a reference (8-bit RGB, same size).

    python tools/image_metrics.py ref.png test.png [--crop x0 y0 x1 y1] [--json]

PSNR (peak 255), RMSE, mean brightness ratio test/ref per channel and overall (bias), and the
firefly fraction: share of test pixels whose max channel exceeds 2x the 99th percentile of the
reference's max channel.
"""
import argparse
import json
import sys

import numpy as np
from PIL import Image


def load(path, crop=None):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float64)
    if crop:
        x0, y0, x1, y1 = crop
        a = a[y0:y1, x0:x1]
    return a


def compute(ref, test):
    """ref, test: float arrays (H, W, 3) in 0..255. Returns a dict of metrics."""
    if ref.shape != test.shape:
        raise ValueError(f"size mismatch: {ref.shape} vs {test.shape}")
    mse = float(np.mean((ref - test) ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
    ref_mean = ref.reshape(-1, 3).mean(axis=0)
    test_mean = test.reshape(-1, 3).mean(axis=0)
    ratio_c = [float(t / r) if r > 0 else float("nan") for t, r in zip(test_mean, ref_mean)]
    ratio = float(test.mean() / ref.mean()) if ref.mean() > 0 else float("nan")
    threshold = 2.0 * float(np.percentile(ref.max(axis=2), 99))
    firefly = float(np.mean(test.max(axis=2) > threshold))
    return {
        "psnr_db": psnr,
        "rmse": float(np.sqrt(mse)),
        "mean_ratio": ratio,
        "mean_ratio_rgb": ratio_c,
        "firefly_fraction": firefly,
        "firefly_threshold": threshold,
    }


def compute_files(ref_path, test_path, crop=None):
    return compute(load(ref_path, crop), load(test_path, crop))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ref")
    p.add_argument("test")
    p.add_argument("--crop", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    try:
        m = compute_files(args.ref, args.test, args.crop)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    r = m["mean_ratio_rgb"]
    print(f"PSNR {m['psnr_db']:.2f} dB, RMSE {m['rmse']:.3f}, mean ratio {m['mean_ratio']:.4f} "
          f"(R {r[0]:.4f} G {r[1]:.4f} B {r[2]:.4f}), fireflies {m['firefly_fraction'] * 100:.3f} %")
    if args.json:
        print(json.dumps(m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
