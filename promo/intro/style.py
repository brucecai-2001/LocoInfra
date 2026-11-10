"""介绍视频的"印刷版画"风格：米色纸、墨色网点、朱红与青绿两个专色。

画面是 uint8 RGB 数组；矢量图形用 cv2（抗锯齿），文字用 PIL 排成灰度蒙版后按颜色盖印，
机器人由 stage.Shot（明度 + 分割）转成网点明暗 + 墨线。
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30

PAPER = np.array([238, 230, 212], dtype=np.float32)
PAPER_HI = np.array([251, 247, 238], dtype=np.float32)
INK = np.array([30, 28, 36], dtype=np.float32)
RED = np.array([219, 64, 37], dtype=np.float32)
TEAL = np.array([24, 108, 112], dtype=np.float32)
MUTED = np.array([124, 114, 98], dtype=np.float32)
FAINT = np.array([196, 186, 166], dtype=np.float32)

_SYS = "/System/Library/Fonts/"
_SUP = _SYS + "Supplemental/"
_FACES = {
    "song_black": (_SUP + "Songti.ttc", 0),
    "song_bold": (_SUP + "Songti.ttc", 1),
    "song": (_SUP + "Songti.ttc", 6),
    "hei": (_SYS + "Hiragino Sans GB.ttc", 0),
    "hei_bold": (_SYS + "Hiragino Sans GB.ttc", 2),
    "din_cond": (_SUP + "DIN Condensed Bold.ttf", 0),
    "din": (_SUP + "DIN Alternate Bold.ttf", 0),
    "mono": (_SYS + "Menlo.ttc", 0),
    "mono_bold": (_SYS + "Menlo.ttc", 1),
    "avenir": (_SYS + "Avenir Next.ttc", 5),
    "avenir_demi": (_SYS + "Avenir Next.ttc", 2),
}


# ---------------------------------------------------------------- 缓动
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def prog(t: float, t0: float, dur: float) -> float:
    """t 在 [t0, t0 + dur] 内的线性进度 0–1。"""
    return clamp((t - t0) / dur) if dur > 0 else float(t >= t0)


def ease_out(p: float) -> float:
    return 1 - (1 - clamp(p)) ** 3


def ease_in_out(p: float) -> float:
    p = clamp(p)
    return 4 * p * p * p if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def ease_back(p: float) -> float:
    """略微回弹的出场。"""
    p = clamp(p)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (p - 1) ** 3 + c1 * (p - 1) ** 2


# ---------------------------------------------------------------- 纸张与网屏
def _make_paper() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    low = cv2.GaussianBlur(rng.normal(0, 1, (H // 8, W // 8)).astype(np.float32), (0, 0), 6)
    low = cv2.resize(low / np.abs(low).max(), (W, H), interpolation=cv2.INTER_CUBIC)
    fibre = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=3.0, sigmaY=0.6)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = 1.0 - 0.075 * np.clip(r - 0.55, 0, 1) ** 1.5 / 0.45
    shade = (1.0 + 0.012 * low + 0.010 * fibre) * vig
    bg = np.clip(PAPER[None, None] * shade[..., None], 0, 255).astype(np.uint8)
    grain = np.clip(rng.normal(0, 3.2, (4, H, W)), -12, 12).astype(np.int16)  # 4 张细颗粒，逐"拍"轮换
    return bg, grain


PAPER_BG, GRAIN = _make_paper()


def _screens():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    def rot(angle):
        a = np.deg2rad(angle)
        return xx * np.cos(a) + yy * np.sin(a), -xx * np.sin(a) + yy * np.cos(a)

    def dots(pitch, angle):
        u, v = rot(angle)
        k = 2 * np.pi / pitch
        return (0.5 - 0.25 * (np.cos(u * k) + np.cos(v * k))).astype(np.float32)

    def lines(pitch, angle):
        u, _ = rot(angle)
        return (0.5 - 0.5 * np.cos(u * 2 * np.pi / pitch)).astype(np.float32)

    return dots(7.0, 45), lines(8.0, 45), lines(6.0, 90), lines(9.0, -45)


DOTS, HATCH, HLINES, HATCH2 = _screens()  # 网点（明暗）/ 斜线（参考影子）/ 横线（地面阴影）/ 反斜线（图表）


def new_frame() -> np.ndarray:
    return PAPER_BG.copy()


def finish(frame: np.ndarray, k: int) -> np.ndarray:
    """加纸张颗粒（每 3 帧换一张，像逐格动画的"抖动"）。"""
    out = frame.astype(np.int16) + GRAIN[(k // 3) % len(GRAIN)][..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def fade_to_paper(frame: np.ndarray, a: float, y1: int = H) -> None:
    """把 y < y1 的区域按 a（0 不变，1 全纸）淡向纸色。"""
    if a <= 0:
        return
    roi = frame[:y1]
    cv2.addWeighted(PAPER_BG[:y1], float(a), roi, 1 - float(a), 0, dst=roi)


# ---------------------------------------------------------------- 盖印（蒙版 × 颜色）
def stamp(frame: np.ndarray, mask: np.ndarray, x: int, y: int, color, alpha: float = 1.0) -> None:
    """把灰度蒙版（uint8 或 0–1 float）以 color 盖到 frame 的 (x, y) 处，自动裁边。"""
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    m = mask[y0 - y : y1 - y, x0 - x : x1 - x]
    m = m.astype(np.float32) * (alpha / 255.0 if mask.dtype == np.uint8 else alpha)
    roi = frame[y0:y1, x0:x1]
    c = np.asarray(color, dtype=np.float32)
    roi[:] = (roi * (1 - m[..., None]) + c * m[..., None] + 0.5).astype(np.uint8)


def _col(color) -> tuple[int, int, int]:
    return tuple(int(round(float(v))) for v in color)


def _blend_roi(frame, x0, y0, x1, y1, alpha, draw) -> None:
    """在 ROI 上画图形；alpha < 1 时先画到副本再混合。"""
    x0, y0, x1, y1 = int(max(x0, 0)), int(max(y0, 0)), int(min(x1, frame.shape[1])), int(min(y1, frame.shape[0]))
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    roi = frame[y0:y1, x0:x1]
    if alpha >= 0.999:
        draw(roi, x0, y0)
        return
    tmp = roi.copy()
    draw(tmp, x0, y0)
    cv2.addWeighted(tmp, float(alpha), roi, 1 - float(alpha), 0, dst=roi)


S = 4  # cv2 亚像素位数
_F = 1 << S


def _pt(p, ox=0, oy=0):
    return int(round((p[0] - ox) * _F)), int(round((p[1] - oy) * _F))


def line(frame, p0, p1, color, width: float = 2, alpha: float = 1.0) -> None:
    pad = width + 2
    x0, y0, x1, y1 = min(p0[0], p1[0]) - pad, min(p0[1], p1[1]) - pad, max(p0[0], p1[0]) + pad, max(p0[1], p1[1]) + pad
    _blend_roi(frame, x0, y0, x1 + 1, y1 + 1, alpha,
               lambda r, ox, oy: cv2.line(r, _pt(p0, ox, oy), _pt(p1, ox, oy), _col(color), max(1, int(round(width))), cv2.LINE_AA, S))


def polyline(frame, pts, color, width: float = 2, alpha: float = 1.0, closed: bool = False, fill: bool = False) -> None:
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return
    pad = width + 2
    x0, y0 = pts.min(0) - pad
    x1, y1 = pts.max(0) + pad + 1

    def draw(r, ox, oy):
        q = np.round((pts - (ox, oy)) * _F).astype(np.int32).reshape(-1, 1, 2)
        if fill:
            cv2.fillPoly(r, [q], _col(color), cv2.LINE_AA, S)
        else:
            cv2.polylines(r, [q], closed, _col(color), max(1, int(round(width))), cv2.LINE_AA, S)

    _blend_roi(frame, x0, y0, x1, y1, alpha, draw)


def dashed(frame, p0, p1, color, width: float = 2, dash: float = 10, gap: float = 8, alpha: float = 1.0, phase: float = 0.0) -> None:
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    n = float(np.linalg.norm(p1 - p0))
    if n < 1e-6:
        return
    d = (p1 - p0) / n
    s = -(phase % (dash + gap))
    while s < n:
        a, b = max(s, 0.0), min(s + dash, n)
        if b > a:
            line(frame, p0 + d * a, p0 + d * b, color, width, alpha)
        s += dash + gap


def circle(frame, c, r: float, color, fill: bool = True, width: float = 2, alpha: float = 1.0) -> None:
    pad = r + width + 2
    _blend_roi(frame, c[0] - pad, c[1] - pad, c[0] + pad + 1, c[1] + pad + 1, alpha,
               lambda roi, ox, oy: cv2.circle(roi, _pt(c, ox, oy), int(round(r * _F)), _col(color),
                                              -1 if fill else max(1, int(round(width))), cv2.LINE_AA, S))


def rect(frame, x0, y0, x1, y1, color, fill: bool = True, width: float = 2, alpha: float = 1.0) -> None:
    if fill:
        _blend_roi(frame, x0, y0, x1, y1, alpha,
                   lambda r, ox, oy: cv2.rectangle(r, (int(round(x0 - ox)), int(round(y0 - oy))),
                                                   (int(round(x1 - ox)) - 1, int(round(y1 - oy)) - 1), _col(color), -1))
    else:
        polyline(frame, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color, width, alpha, closed=True)


def arrow(frame, p0, p1, color, width: float = 2.5, head: float = 14, alpha: float = 1.0) -> None:
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    n = float(np.linalg.norm(p1 - p0))
    if n < 1.0:
        return
    d = (p1 - p0) / n
    nrm = np.array([-d[1], d[0]])
    head = min(head, n)
    line(frame, p0, p1 - d * head * 0.8, color, width, alpha)
    polyline(frame, [p1, p1 - d * head + nrm * head * 0.42, p1 - d * head - nrm * head * 0.42], color, 1, alpha, fill=True)


def screen_rect(frame, x0, y0, x1, y1, color, screen: np.ndarray, level: float, alpha: float = 1.0, sharp: float = 4.0) -> None:
    """用网屏（斜线 / 网点）按 level 0–1 的浓度填一个矩形。"""
    x0, y0, x1, y1 = int(max(x0, 0)), int(max(y0, 0)), int(min(x1, W)), int(min(y1, H))
    if x1 <= x0 or y1 <= y0:
        return
    m = np.clip((level - screen[y0:y1, x0:x1]) * sharp + 0.5, 0, 1)
    stamp(frame, m, x0, y0, color, alpha)


# ---------------------------------------------------------------- 文字
def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return 0x2E80 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF or 0x3000 <= o <= 0x303F or ch in "·—→←↑↓×≈≤≥°±"


class Font:
    """西文字体 + 中文字体配对；中文字符（及部分符号）用 cjk，其余用 latin。"""

    def __init__(self, latin: str, cjk: str, size: float, cjk_scale: float = 1.0, tracking: float = 0.0, cjk_dy: float = 0.0):
        self.key = (latin, cjk, size, cjk_scale, tracking, cjk_dy)
        lp, li = _FACES[latin]
        cp, ci = _FACES[cjk]
        self.latin = ImageFont.truetype(lp, int(round(size)), index=li)
        self.cjk = ImageFont.truetype(cp, int(round(size * cjk_scale)), index=ci)
        self.size, self.tracking, self.cjk_dy = size, tracking, cjk_dy

    def runs(self, s: str) -> list[tuple[str, ImageFont.FreeTypeFont, bool]]:
        out: list[tuple[str, ImageFont.FreeTypeFont, bool]] = []
        for ch in s:
            c = _is_cjk(ch)
            if out and out[-1][2] == c and not (self.tracking and not c):
                out[-1] = (out[-1][0] + ch, out[-1][1], c)
            else:
                out.append((ch, self.cjk if c else self.latin, c))
        return out


@lru_cache(maxsize=4096)
def _font(latin: str, cjk: str, size: float, cjk_scale: float, tracking: float, cjk_dy: float) -> Font:
    return Font(latin, cjk, size, cjk_scale, tracking, cjk_dy)


def font(latin: str, cjk: str, size: float, cjk_scale: float = 1.0, tracking: float = 0.0, cjk_dy: float = 0.0) -> Font:
    return _font(latin, cjk, size, cjk_scale, tracking, cjk_dy)


@lru_cache(maxsize=4096)
def _text_mask(s: str, key: tuple) -> tuple[np.ndarray, int, float]:
    """返回（蒙版，基线在蒙版里的行号，排版宽度）。"""
    f = _font(*key)
    runs = f.runs(s)
    width = sum(fo.getlength(r) + (f.tracking if not c else 0.0) for r, fo, c in runs)
    asc = max(fo.getmetrics()[0] for _, fo, _ in runs) if runs else int(f.size)
    desc = max(fo.getmetrics()[1] for _, fo, _ in runs) if runs else 0
    pad = 4
    img = Image.new("L", (int(np.ceil(width)) + 2 * pad + 2, asc + desc + 2 * pad + int(abs(f.cjk_dy)) + 2), 0)
    d = ImageDraw.Draw(img)
    x = float(pad)
    base = pad + asc
    for r, fo, c in runs:
        d.text((x, base + (f.cjk_dy if c else 0.0)), r, font=fo, fill=255, anchor="ls")
        x += fo.getlength(r) + (f.tracking if not c else 0.0)
    return np.asarray(img), base, float(width)


def text_width(s: str, f: Font) -> float:
    return _text_mask(s, f.key)[2] if s else 0.0


def text(frame, x: float, y: float, s: str, f: Font, color=INK, alpha: float = 1.0, anchor: str = "l",
         rise: float = 1.0, wipe: float = 1.0) -> float:
    """在基线 (x, y) 处排一行字。anchor：l / m / r。rise < 1 时从基线下方升起（带裁切），wipe < 1 时从左到右揭开。返回排版宽度。"""
    if not s or alpha <= 0:
        return 0.0
    mask, base, width = _text_mask(s, f.key)
    x0 = x - 4 - (width / 2 if anchor == "m" else width if anchor == "r" else 0)
    y0 = y - base
    if wipe < 1.0:
        mask = mask[:, : max(0, int(round(mask.shape[1] * clamp(wipe))))]
    if rise < 1.0:
        dy = int(round((1 - clamp(rise)) * mask.shape[0] * 0.85))
        if dy >= mask.shape[0]:
            return width
        m2 = np.zeros_like(mask)
        m2[dy:] = mask[: mask.shape[0] - dy]
        mask = m2
    if mask.shape[1] > 0:
        stamp(frame, mask, int(round(x0)), int(round(y0)), color, alpha)
    return width


def text_spans(frame, x: float, y: float, spans: list[tuple], alpha: float = 1.0, anchor: str = "l", gap: float = 0.0) -> float:
    """一行里多段不同字体 / 颜色：spans = [(文字, Font, 颜色), ...]。"""
    total = sum(text_width(s, f) for s, f, _ in spans) + gap * (len(spans) - 1)
    cx = x - (total / 2 if anchor == "m" else total if anchor == "r" else 0)
    for s, f, c in spans:
        text(frame, cx, y, s, f, c, alpha)
        cx += text_width(s, f) + gap
    return total


# ---------------------------------------------------------------- 机器人
def _bbox(mask: np.ndarray, pad: int = 6):
    ys = np.flatnonzero(mask.any(axis=1))
    if len(ys) == 0:
        return None
    xs = np.flatnonzero(mask.any(axis=0))
    return max(xs[0] - pad, 0), max(ys[0] - pad, 0), min(xs[-1] + pad + 1, mask.shape[1]), min(ys[-1] + pad + 1, mask.shape[0])


def _edges(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(外轮廓, 连杆之间的内线)，bool，各约 2 px 宽。"""
    dx = ids[:, 1:] != ids[:, :-1]
    dy = ids[1:] != ids[:-1]
    sil_x = dx & ((ids[:, 1:] < 0) | (ids[:, :-1] < 0))
    sil_y = dy & ((ids[1:] < 0) | (ids[:-1] < 0))
    sil, inner = np.zeros(ids.shape, bool), np.zeros(ids.shape, bool)
    for e, (a, b) in ((sil_x, (sil, 1)), (dx & ~sil_x, (inner, 1)), (sil_y, (sil, 0)), (dy & ~sil_y, (inner, 0))):
        if b:
            a[:, 1:] |= e
            a[:, :-1] |= e
        else:
            a[1:] |= e
            a[:-1] |= e
    return sil, inner


def robot_layers(shot, pad: int = 8):
    """从一次拍摄里取机器人的 (bbox, 柔边蒙版, 明度, 外轮廓线, 内线)；没拍到机器人返回 None。"""
    mask = shot.ids >= 0
    bb = _bbox(mask, pad)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    ids = shot.ids[y0:y1, x0:x1]
    soft = cv2.GaussianBlur((ids >= 0).astype(np.float32), (0, 0), 0.7)
    sil, inner = _edges(ids)
    sil = np.clip(cv2.GaussianBlur(cv2.dilate(sil.astype(np.float32), np.ones((3, 3), np.float32)), (0, 0), 0.8) * 1.5, 0, 1)
    inner = np.clip(cv2.GaussianBlur(inner.astype(np.float32), (0, 0), 0.7) * 1.5, 0, 1)
    return bb, soft, shot.lum[y0:y1, x0:x1], sil, inner


def draw_robot(frame, shot, alpha: float = 1.0, layers=None) -> None:
    """墨色网点明暗 + 墨线。"""
    L = layers or robot_layers(shot)
    if L is None or alpha <= 0:
        return
    (x0, y0, x1, y1), soft, lum, sil, inner = L
    cov = np.clip((0.90 - lum) / 0.62, 0, 1) ** 1.25 * 0.94
    ink = np.clip((cov - DOTS[y0:y1, x0:x1]) * 5.0 + 0.5, 0, 1)
    ink = np.maximum(ink, np.clip((0.30 - lum) / 0.08, 0, 1))  # 最暗处实地
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    col = PAPER_HI * (1 - ink[..., None]) + INK * ink[..., None]
    a = (soft * alpha)[..., None]
    roi = roi * (1 - a) + col * a
    ln = (np.maximum(sil, inner * 0.85) * alpha)[..., None]
    roi = roi * (1 - ln) + INK * ln
    frame[y0:y1, x0:x1] = (roi + 0.5).astype(np.uint8)


def draw_ghost(frame, shot, color=RED, alpha: float = 1.0, fill: float = 0.55, outline: float = 0.95, layers=None,
               inner_lines: bool = True, paper: float = 0.0) -> None:
    """专色斜线填充 + 专色轮廓（参考动作的"影子"、残影）。paper > 0 时先垫一层纸白，盖住后面的东西。"""
    L = layers or robot_layers(shot)
    if L is None or alpha <= 0:
        return
    (x0, y0, x1, y1), soft, _, sil, inner = L
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    c = np.asarray(color, dtype=np.float32)
    if paper > 0:
        a = (soft * paper * alpha)[..., None]
        roi = roi * (1 - a) + PAPER_HI * a
    if fill > 0:
        h = np.clip((0.40 - HATCH[y0:y1, x0:x1]) * 4.0 + 0.5, 0, 1) * soft
        a = (h * fill * alpha)[..., None]
        roi = roi * (1 - a) + c * a
    ln = np.maximum(sil, inner * 0.6) if inner_lines else sil
    a = (ln * outline * alpha)[..., None]
    roi = roi * (1 - a) + c * a
    frame[y0:y1, x0:x1] = (roi + 0.5).astype(np.uint8)


def draw_shadow(frame, shot, alpha: float = 1.0, lit: float = 0.976) -> None:
    """地面阴影：横线网屏的墨色。"""
    sh = np.where(shot.floor, np.clip((lit - shot.lum) / 0.30, 0, 1), 0).astype(np.float32)
    bb = _bbox(sh > 0.02, 4)
    if bb is None or alpha <= 0:
        return
    x0, y0, x1, y1 = bb
    s = cv2.GaussianBlur(sh[y0:y1, x0:x1], (0, 0), 1.2)
    m = np.clip((s * 0.62 - HLINES[y0:y1, x0:x1]) * 3.0 + 0.5, 0, 1) * np.clip(s * 3, 0, 1)
    stamp(frame, m, x0, y0, INK, 0.62 * alpha)


def draw_grid(frame, shot, center, radius: float = 2.6, step: float = 0.25, alpha: float = 1.0, y_max: int = H) -> None:
    """地面网格：以 center（世界 xy）为中心，离得越远越淡。"""
    if alpha <= 0:
        return
    layer = np.zeros((H, W), np.uint8)
    cx, cy = (np.round(np.asarray(center[:2], dtype=np.float64) / 1.0) * 1.0)
    ks = np.arange(-int(radius / step) - 2, int(radius / step) + 3)
    seg = step / 2
    ts = np.arange(-radius - 0.5, radius + 0.5 + 1e-6, seg)
    for axis in (0, 1):
        for k in ks:
            major = (k % int(round(1.0 / step))) == 0
            pts = np.zeros((len(ts), 3))
            pts[:, axis] = (cx if axis == 0 else cy) + k * step
            pts[:, 1 - axis] = (cy if axis == 0 else cx) + ts
            px, z = shot.project(pts)
            r = np.linalg.norm(pts[:, :2] - np.asarray(center[:2]), axis=1)
            a = np.clip(1 - r / radius, 0, 1) ** 1.3 * (1.0 if major else 0.5)
            for i in range(len(ts) - 1):
                v = 0.5 * (a[i] + a[i + 1])
                if v < 0.02 or z[i] < 0.15 or z[i + 1] < 0.15:
                    continue
                cv2.line(layer, _pt(px[i]), _pt(px[i + 1]), int(v * 255), 1, cv2.LINE_AA, S)
    layer[y_max:] = 0
    bb = _bbox(layer > 0, 0)
    if bb is None:
        return
    x0, y0, x1, y1 = bb
    stamp(frame, layer[y0:y1, x0:x1], x0, y0, MUTED, 0.62 * alpha)
