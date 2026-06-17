#!/usr/bin/env python3
"""Convert a hand-made RGBA cutout into a clean matting alpha aligned to the
original photo.  Used for the self-collected (campus) cognitive-challenge
samples.  The whole pipeline is classical / model-free on purpose: the ground
truth must NOT be produced by any benchmarked matting model, otherwise that
model would score ~0 error on its own GT (evaluation leakage).

Pipeline (cutout RGBA + original RGB -> alpha):
  1. take the cutout alpha channel;
  2. de-burr: keep only the main connected component (bridge hairline gaps with
     a small dilation, then drop isolated brush specks);
  3. solidify brush-opacity translucency with a levels curve (>=hi -> 255,
     <=lo -> 0), keeping a thin genuine soft edge in between;
  4. register the cutout to the original photo with SIFT + partial-affine RANSAC
     (the cutout is usually a crop/scale of the original), then warp the alpha
     into the original frame;
  5. snap/anti-alias the edge with a guided filter using the original gray as
     guide (classical, model-independent);
  6. force solid interior and clear far background, save 8-bit PNG.

Trimap for evaluation is generated separately by clean_and_gen_trimap.py
(dilate/erode the clean alpha, unknown band = 128).

Usage:
    python cutout_to_alpha.py --cutout cut.png --orig orig.jpg --out alpha.png
"""
import argparse
import cv2
import numpy as np
from scipy import ndimage


def _small(img, mx):
    h, w = img.shape[:2]
    s = mx / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s))), s


def register(cut_rgb, cut_alpha, orig, mx=2000):
    """Estimate cutout -> original affine (rotation+scale+translation)."""
    os_, sco = _small(orig, mx)
    cs_, scc = _small(cut_rgb, mx)
    m = cv2.resize(cut_alpha, (cs_.shape[1], cs_.shape[0]))
    mask = cv2.erode((m > 10).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
    sift = cv2.SIFT_create(30000, contrastThreshold=0.015, edgeThreshold=20)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(cs_, cv2.COLOR_BGR2GRAY), mask)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(os_, cv2.COLOR_BGR2GRAY), None)
    matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(d1, d2)
    src = np.float32([k1[x.queryIdx].pt for x in matches]).reshape(-1, 1, 2)
    dst = np.float32([k2[x.trainIdx].pt for x in matches]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                       ransacReprojThreshold=4, maxIters=8000,
                                       confidence=0.999, refineIters=50)
    M = M.copy()
    M[:, :2] *= (scc / sco)     # rescale to full resolution
    M[:, 2] /= sco
    return M


def guided_filter(guide, src, r, eps):
    bf = lambda x: cv2.boxFilter(x, -1, (r, r))
    mI, mp = bf(guide), bf(src)
    a = (bf(guide * src) - mI * mp) / (bf(guide * guide) - mI * mI + eps)
    b = mp - a * mI
    return bf(a) * guide + bf(b)


def cutout_to_alpha(cutout_path, orig_path, lo=10.0, hi=90.0):
    ca = cv2.imread(cutout_path, cv2.IMREAD_UNCHANGED)
    orig = cv2.imread(orig_path)
    H, W = orig.shape[:2]
    al = ca[..., 3].astype(np.float32)

    # (2) de-burr: keep bridged main connected component
    m = al > 10
    d = ndimage.binary_dilation(m, iterations=7)
    lab, n = ndimage.label(d)
    sums = ndimage.sum(m, lab, range(1, n + 1))
    keep = (lab == (np.argmax(sums) + 1)) & m
    al = np.where(keep, al, 0)

    # (3) solidify brush-opacity translucency, keep thin true edge
    al = np.clip((al - lo) / (hi - lo), 0, 1) * 255

    # (4) register to original frame
    M = register(ca[..., :3], ca[..., 3], orig)
    wal = cv2.warpAffine(al.astype(np.uint8), M, (W, H),
                         flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    # (5) edge refine with guided filter (classical, model-free)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    r = max(6, int(max(H, W) / 500))
    q = np.clip(guided_filter(gray, wal, r, 1e-4), 0, 1)

    # (6) solid interior / clear far background
    fg_core = cv2.erode((wal > 0.9).astype(np.uint8),
                        np.ones((r + 3, r + 3), np.uint8)).astype(bool)
    out = q.copy()
    out[fg_core] = 1.0
    out[wal < 0.02] = 0.0
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutout", required=True, help="hand-made RGBA cutout PNG")
    ap.add_argument("--orig", required=True, help="original photo (alpha is aligned to it)")
    ap.add_argument("--out", required=True, help="output 8-bit alpha PNG")
    a = ap.parse_args()
    cv2.imwrite(a.out, cutout_to_alpha(a.cutout, a.orig))
    print("wrote", a.out)
