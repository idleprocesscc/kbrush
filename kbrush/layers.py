"""图层：Procreate 那样的一叠图层，建在一张纸上，从下往上合成。

    st = Stack(K.Paper(H, W, seed=3))
    bg = st.layer('底')                       # 不给 mode 就是 normal：画上去的东西不透明地盖住底下
    fill(bg, Shape(np.ones((H, W), bool)), '#1c1f2b')
    L = st.layer('笔')
    b = K.Brush('flat', 50); dip(b, (1.0, 0.55, 0.1), water=0.6)
    stroke(L, b, **K.gesture((100, 300), (700, 320), width=50, end='chu'), within=形)   # 亮橙压在暗底上，只落在形里
    fill(st.layer('影', mode='multiply'), cast, (0.55, 0.5, 0.68), edge=SOFT)          # 透明罩色：multiply 层，颜色自己挑
    fill(st.layer('光', mode='screen', mask=soft(beam, feather=60)), 全张, (1, .9, .6))  # 图层蒙版 1 露 0 藏
    look(st, 'x.png')

- Stack(纸, ground=None)：ground 给一个颜色就是平的底色。st.layer(名字, mode=, opacity=, mask=) 开一层放在最上面；st['名字'] 拿回来。
- 每层是一张 RGBA。mode 管它怎么跟底下合：normal 盖住，multiply 正片叠底，screen 滤色，overlay 叠加，
  soft_light 柔光，darken / lighten 变暗 / 变亮，add 相加（发光）。
- mask：图层蒙版（0..1，1 = 露出来；选区、bool 图，或 soft(选区, feather=) 给的羽化图）。opacity 不透明度，visible 显示 / 隐藏。
- fill(层, 选区, 颜色, edge=, grain=)：选区填一个直接挑的颜色。dip / dip_tip / stroke(层, 笔, **K.gesture(...), within=)：拿 kbrush 的笔刷画。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.join(_HERE, 'paint')):
    if _d not in sys.path:
        sys.path.append(_d)

import kbrush as K          # noqa: E402
import blocks as B          # noqa: E402
from blocks import HARD, SOFT, LOST   # noqa: E402,F401


# ---------- 混合模式（显示值上算，跟 Procreate 一样；b 底下，s 这一层的颜色） ----------
def _overlay(b, s):
    return np.where(b <= 0.5, 2 * b * s, 1 - 2 * (1 - b) * (1 - s))


def _soft_light(b, s):
    d = np.where(b <= 0.25, ((16 * b - 12) * b + 4) * b, np.sqrt(b))
    return np.where(s <= 0.5, b - (1 - 2 * s) * b * (1 - b), b + (2 * s - 1) * (d - b))


BLEND = {
    'normal': lambda b, s: s,
    'multiply': lambda b, s: b * s,
    'screen': lambda b, s: b + s - b * s,
    'overlay': _overlay,
    'soft_light': _soft_light,
    'darken': np.minimum,
    'lighten': np.maximum,
    'add': lambda b, s: np.minimum(b + s, 1.0),
}
MODES = tuple(BLEND)


def rgb(c):
    """颜色：(r, g, b) 0..1，或 0..255 的整数，或 '#rrggbb'。也可以是一张 (H, W, 3) 的图（每个像素一个颜色）。"""
    if isinstance(c, str):
        c = c.lstrip('#')
        return np.array([int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    a = np.asarray(c, float)
    if a.ndim == 1 and a.max() > 1.0:
        a = a / 255
    return a


def _as_map(m, H, W):
    """蒙版：Shape / bool 图 / 0..1 的图 → float32 (H, W)。None 照旧 None。"""
    if m is None:
        return None
    m = np.asarray(getattr(m, 'mask', m))
    return np.broadcast_to(np.clip(m.astype(np.float32), 0, 1), (H, W)).copy()


def _box(a, thr=0.0):
    """a > thr 的那个框 (y0, y1, x0, x1)；没有就 None。"""
    rows = np.nonzero((a > thr).any(1))[0]
    if not len(rows):
        return None
    cols = np.nonzero((a[rows[0]:rows[-1] + 1] > thr).any(0))[0]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])


def _sl(box):
    return slice(box[0], box[1]), slice(box[2], box[3])


class PaintLayer:
    """一张 RGBA：C 是乘过 alpha 的颜色 (3, H, W)，A 是 alpha (H, W)，都是 float32，第一次落色才分配。"""

    def __init__(self, paper, name, mode='normal', opacity=1.0, mask=None):
        if mode not in BLEND:
            raise ValueError(f'mode 只认 {MODES}，给的是 {mode!r}')
        self.p = paper
        self.name, self.mode, self.opacity, self.visible = name, mode, float(opacity), True
        self.lmask = mask
        self.C = None
        self.A = None
        self.box = None                       # 有东西的那个框（只增不减）

    @property
    def lmask(self):
        return self.__dict__.get('_lmask')

    @lmask.setter
    def lmask(self, m):
        self.__dict__['_lmask'] = _as_map(m, self.p.H, self.p.W)

    def alpha(self):
        return np.zeros((self.p.H, self.p.W), np.float32) if self.A is None else self.A

    def _paint(self, win, a, col):
        """在窗口 win 里按覆盖量 a（0..1）落颜色 col（(3,) 或 (3, h, w)），像 Procreate 的笔：新的盖在旧的上面。"""
        if self.A is None:
            self.C = np.zeros((3, self.p.H, self.p.W), np.float32)
            self.A = np.zeros((self.p.H, self.p.W), np.float32)
        ys, xs = _sl(win)
        a = np.clip(a, 0, 1).astype(np.float32)
        col = np.clip(np.asarray(col, np.float32), 0, 1)
        col = col[:, None, None] if col.ndim == 1 else col
        A = self.A[ys, xs]; C = self.C[:, ys, xs]
        C *= 1 - a; C += col * a
        A *= 1 - a; A += a
        bb = _box(a)
        if bb is not None:
            self.box = _union(self.box, (win[0] + bb[0], win[0] + bb[1], win[2] + bb[2], win[2] + bb[3]))

    def _composite(self, R):
        if self.box is None:
            return
        ys, xs = _sl(self.box)
        A = self.A[ys, xs]
        a = A * np.float32(self.opacity)
        if self.lmask is not None:
            a = a * self.lmask[ys, xs]
        if not a.any():
            return
        s = (self.C[:, ys, xs] / np.maximum(A, 1e-6)).transpose(1, 2, 0)
        b = R[ys, xs]
        R[ys, xs] = b + a[..., None] * (BLEND[self.mode](b, s) - b)


class Stack:
    """一叠图层，建在纸上。ground=None 底是纸（纸色乘纸纹）；给一个颜色就是一张平的底色。"""

    def __init__(self, paper, ground=None):
        self.p = paper
        self.layers = []
        self.ground = ground

    def layer(self, name, mode='normal', opacity=1.0, mask=None):
        """开一层放在最上面，返回它。mode：normal（默认）/ multiply / screen / overlay / soft_light / darken / lighten / add。
        mask：图层蒙版（1 = 露）。"""
        if any(L.name == name for L in self.layers):
            raise ValueError(f'已经有一层叫 {name!r}')
        L = PaintLayer(self.p, name, mode, opacity, mask)
        self.layers.append(L)
        return L

    def __getitem__(self, name):
        for L in self.layers:
            if L.name == name:
                return L
        raise KeyError(name)

    def _ground(self):
        if self.ground is None:
            g = (0.93 + 0.07 * self.p.h)[..., None]
            return np.broadcast_to(self.p.color * g, (self.p.H, self.p.W, 3)).copy()
        return np.broadcast_to(rgb(self.ground), (self.p.H, self.p.W, 3)).astype(float).copy()

    def render(self):
        R = self._ground()
        for L in self.layers:
            if L.visible:
                L._composite(R)
        return np.clip(R, 0, 1)


def render(stack):
    return stack.render()


# ---------- 在层上画 ----------
def _grained(p, win, rng, a, col, grain):
    """颗粒落在不透明的色上：颗粒谱长出来的一团团（几像素到三十来像素）让颜色一团深一团浅
    （乘一成多，暗色再加一点点，不然暗色上看不出团）；纸纹最深的那些凹处颜料够不着，底下透出来一点。
    只做"透"会像数码噪点（暗色铺在纸上是一层雪），所以主要靠深浅。返回 (a, 每像素的颜色 (3, h, w))。"""
    ys, xs = _sl(win)
    h, w = win[1] - win[0], win[3] - win[2]
    pg = B._pgrain(h, w, rng)
    low = -B._z(0.6 * B._z(p.grain[ys, xs]) + 0.4 * B._z(p.h[ys, xs]))      # 纸纹凹处为正
    a = a * np.clip(1 - grain * 0.4 * np.clip(B._z(0.5 * pg + 0.9 * low) - 1.4, 0, None), 0, 1)
    col = np.asarray(col, float)
    col = col[:, None, None] if col.ndim == 1 else col
    dark = 1 - col.mean(0, keepdims=True)
    return a, col * (1 + grain * 0.12 * pg)[None] + (grain * 0.035 * pg)[None] * dark


def fill(layer, sel, color, edge=HARD, opacity=1.0, water=0.5, rag=1.0, grain=0.0, seed=0):
    """选区填一个直接挑的颜色（不透明，按这层的 mode 跟底下合）。
    edge：HARD 锐（带一点纸的牙口）/ SOFT 往外淡出约 10px / LOST 约 25px，或 edges() 给的图。
    water 越多 SOFT / LOST 淡得越开；rag 边推拉的倍数；grain：颗粒（0 = 平的，1 = 明显的颗粒，底色从颗粒稀处透出来）。
    color 也可以是一张 (H, W, 3) 的图（渐变）。返回窗口 (y0, y1, x0, x1)。"""
    H, W = layer.p.H, layer.p.W
    sel = sel if hasattr(sel, 'mask') else B.Shape(np.asarray(sel, bool))
    rng = np.random.default_rng(seed)
    E = edge if isinstance(edge, np.ndarray) else None
    default = SOFT if E is not None else edge
    wmul = 0.6 + 0.8 * water
    reach = B._qreach(sel, E, default, wmul, rag)
    y0, y1, x0, x1 = sel.bbox
    win = (max(int(y0 - reach), 0), min(int(y1 + reach) + 1, H), max(int(x0 - reach), 0), min(int(x1 + reach) + 1, W))
    ys, xs = _sl(win)
    m = B._pinholes(sel.mask[ys, xs])
    fibz = B._z(np.log(layer.p.perm[ys, xs]))
    a = B._qcov(m, E[ys, xs] if E is not None else None, default, wmul, rag, fibz, rng)
    col = rgb(color)
    if col.ndim == 3:
        col = col[ys, xs].transpose(2, 0, 1)
    if grain:
        a, col = _grained(layer.p, win, rng, a, col, grain)
    layer._paint(win, a * opacity, col)
    return win


# 笔：kbrush 的鬃、压力、速度、干笔照样走，纸上落下的颜料量当覆盖量，颜色直接给。
INK_FULL = 0.1        # 落下这么多颜料，盖住六成三（1 - e^-1）；饱笔的中间一般 0.15–0.35，盖住八九成，鬃纹还在


def dip(brush, color, water=1.0, load=1.0, paste=None):
    """蘸：water 几成饱（0.25 上下是干笔），load 蘸多浓（越浓越盖得住）。"""
    brush.dip(load=load, water=water, paste=paste)
    c = rgb(color)
    brush.rgb = [c, c]


def dip_tip(brush, color, depth=0.35, part='tip', load=1.5):
    """笔尖（或平头笔一个角，part='corner'）再点一下另一个颜色：一笔两色，越画笔尖那色越淡。"""
    brush.dip_tip(load=load, depth=depth, part=part)
    brush.rgb[1] = rgb(color)


def stroke(layer, brush, pts, opacity=1.0, grain=0.0, within=None, **kw):
    """拿 K.Brush 在这层上拖一笔（先 dip(brush, 颜色)）。pts / pressure / speed / roll / side / hold / seed / streak 跟 K.stroke 一样，
    配 K.gesture 用：stroke(L, b, **K.gesture(...))。一笔落下的颜料量 → 覆盖量（不透明的色，鬃纹、飞白、干笔的碎都在）。
    opacity：这一笔的不透明度；grain：颗粒（跟 fill 一样）；within：只落在这个选区 / 0..1 的图里面。返回窗口，或 None（没落下东西）。"""
    if getattr(brush, 'rgb', None) is None:
        raise ValueError('笔先用 layers.dip(b, 颜色) 蘸')
    amt = K.stroke(layer.p, brush, pts, **kw)
    tot = np.maximum(amt.sum(0), 0)
    win = _box(tot, 1e-6)
    if win is None:
        return None
    ys, xs = _sl(win)
    t = tot[ys, xs]
    a = 1 - np.exp(-t / INK_FULL)
    if within is not None:
        a = a * np.asarray(getattr(within, 'mask', within), float)[ys, xs]
    c0, c1 = (np.asarray(c, float) for c in brush.rgb)
    if np.array_equal(c0, c1):
        col = c0
    else:
        f = np.clip(amt[1, ys, xs] / np.maximum(t, 1e-9), 0, 1)
        col = c0[:, None, None] * (1 - f) + c1[:, None, None] * f
    if grain:
        seed = kw.get('seed')
        rng = np.random.default_rng(K._auto_seed(None) if seed is None else [seed, 101])
        a, col = _grained(layer.p, win, rng, a, col, grain)
    layer._paint(win, a * opacity, col)
    return win
