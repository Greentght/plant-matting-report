# PM 数据集 alpha 清洗 + trimap 生成核心函数（节选自 data_scripts/clean_and_gen_trimap.py）
# 背景：PM 部分真值 alpha 的“背景”并非纯 0，而是 2/3/5/12 的近黑常数台地
#       （JPEG/色彩转换伪影）。trimap 生成器把非 0 非 255 的像素一律记为未知区，
#       这些图于是有 90%~97% 都成了未知区，ViTMatte 失去“确定背景”锚点 →
#       误差从 SAD 25 恶化到 140。故必须先清洗、再生成 trimap。
import cv2
import numpy as np


def clean_alpha(alpha, plateau_frac=0.02, low_search=20, high_search=20, margin=2):
    """把背景/前景的伪影“台地”吸附回 0/255。

    “台地”指占据图像超过 plateau_frac 比例的某个低值（或高值）——
    它是一块平坦常数区，对 matte 而言只可能是背景(0)或前景(255)，
    绝不会是真实的半透明软 alpha。台地（含 margin 容差）以内的像素被吸附。
    真实软边像素（发丝、薄叶）稀疏地散布在中间值，不会被误吸附。
    """
    a = alpha.copy()
    n = a.size

    # ---- 低值端：找出最高的“背景台地”值，并把它及以下吸附为 0 ----
    bg_cut = 0
    for v in range(1, low_search + 1):
        if (a == v).sum() / n > plateau_frac:
            bg_cut = v
    if bg_cut > 0:
        a[a <= bg_cut + margin] = 0

    # ---- 高值端：找出最低的“前景台地”值，并把它及以上吸附为 255 ----
    fg_cut = 255
    for v in range(254, 254 - high_search, -1):
        if (a == v).sum() / n > plateau_frac:
            fg_cut = v
    if fg_cut < 255:
        a[a >= fg_cut - margin] = 255

    return a


def alpha_to_trimap(alpha, kernel_size=15, iterations=1, fg_thr=254, bg_thr=1):
    """清洗后的 alpha(0-255) -> 三分图 trimap{0, 128, 255}。

    思路：以确定前景为种子做形态学膨胀/腐蚀，沿真值轮廓围出一条过渡带：
      - 腐蚀结果内部 = 确定前景(255)
      - 膨胀结果减去腐蚀 = 未知过渡带(128)
      - 其余 = 确定背景(0)
    再把 alpha 本身介于 (bg_thr, fg_thr) 的半透明像素并入未知带。
    kernel_size 直接决定未知带宽度，必须四个模型评测时统一固定。
    """
    fg = (alpha >= fg_thr).astype(np.uint8)
    semi = ((alpha > bg_thr) & (alpha < fg_thr)).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(fg, kernel, iterations=iterations)
    dilated = cv2.dilate(fg, kernel, iterations=iterations)

    trimap = np.zeros_like(alpha, dtype=np.uint8)
    trimap[dilated > 0] = 128   # 膨胀域：先全部置为未知
    trimap[eroded > 0] = 255    # 腐蚀域：内核确定前景
    trimap[semi > 0] = 128      # 半透明像素并入未知带
    return trimap
