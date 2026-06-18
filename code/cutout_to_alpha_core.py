# 手工抠图 -> 标准 alpha 真值（节选自 data_scripts/cutout_to_alpha.py）
# 用于校园自采的认知挑战样本标注。整条流程刻意全部为经典/无学习方法：
# 真值绝不能由任何被评测的抠图模型生成，否则该模型会在自己生成的真值上误差趋零
# （评测泄漏）。引导滤波等经典滤波不偏向任一学习型模型，故可用。
#
# 流程（手工抠图 RGBA + 原始照片 RGB -> alpha）：
#   (1) 取手工抠图的 alpha 通道；
#   (2) 去毛刺：只保留主连通域（先小幅膨胀桥接发丝级缝隙，再丢弃孤立画笔斑点）；
#   (3) 用 levels 曲线填实画笔不透明度造成的半透明（>=hi->255，<=lo->0），中间保留真实软边；
#   (4) 用 SIFT + 部分仿射 RANSAC 把抠图配准回原图坐标系（抠图通常是原图的裁剪/缩放），
#       再把 alpha 仿射变换进原图坐标系；
#   (5) 以原图灰度为引导做 guided filter，做边缘贴合与抗锯齿（经典、与模型无关）；
#   (6) 填实内部、清零远景背景，输出 8-bit PNG。
import cv2
import numpy as np
from scipy import ndimage


def _small(img, mx):
    """等比缩小到长边 mx，返回缩小图与缩放比（配准在小图上做以提速）。"""
    h, w = img.shape[:2]
    s = mx / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s))), s


def register(cut_rgb, cut_alpha, orig, mx=2000):
    """估计 抠图 -> 原图 的部分仿射变换（旋转+缩放+平移）。"""
    os_, sco = _small(orig, mx)
    cs_, scc = _small(cut_rgb, mx)
    m = cv2.resize(cut_alpha, (cs_.shape[1], cs_.shape[0]))
    # 只在抠图前景内部（腐蚀后）提取特征点，避免透明区噪声干扰
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
    M[:, :2] *= (scc / sco)     # 由小图比例换算回全分辨率
    M[:, 2] /= sco
    return M


def guided_filter(guide, src, r, eps):
    """He et al. 引导滤波：以 guide 为引导对 src 做保边平滑。"""
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

    # (2) 去毛刺：保留桥接后的主连通域
    m = al > 10
    d = ndimage.binary_dilation(m, iterations=7)
    lab, n = ndimage.label(d)
    sums = ndimage.sum(m, lab, range(1, n + 1))
    keep = (lab == (np.argmax(sums) + 1)) & m
    al = np.where(keep, al, 0)

    # (3) 填实画笔不透明度造成的半透明，保留薄软边
    al = np.clip((al - lo) / (hi - lo), 0, 1) * 255

    # (4) 配准回原图坐标系
    M = register(ca[..., :3], ca[..., 3], orig)
    wal = cv2.warpAffine(al.astype(np.uint8), M, (W, H),
                         flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    # (5) 以原图灰度为引导做边缘精修（经典、无学习模型参与）
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    r = max(6, int(max(H, W) / 500))
    q = np.clip(guided_filter(gray, wal, r, 1e-4), 0, 1)

    # (6) 填实内部 / 清零远景背景
    fg_core = cv2.erode((wal > 0.9).astype(np.uint8),
                        np.ones((r + 3, r + 3), np.uint8)).astype(bool)
    out = q.copy()
    out[fg_core] = 1.0
    out[wal < 0.02] = 0.0
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)

# 评测用 trimap 由 clean_and_gen_trimap.py 单独生成（对 clean alpha 膨胀/腐蚀，未知带=128）。
