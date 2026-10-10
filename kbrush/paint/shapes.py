"""形：选区是一张 bool 图（Shape），能 | 并、& 交、- 减；fbm 是推轮廓用的平滑噪声。"""
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt


def fbm(H, W, scale, seed=0, octaves=3):
    """平滑噪声，特征尺寸约 scale 像素，均值 0、方差约 1。"""
    rng = np.random.default_rng(seed)
    out = np.zeros((H, W)); tot = 0.0; amp = 1.0
    for o in range(octaves):
        s = scale / 2 ** o
        # 在小图上生成再放大，快
        f = max(1, int(s / 6))
        h, w = H // f + 2, W // f + 2
        n = gaussian_filter(rng.standard_normal((h, w)), s / f / 2.5, mode='wrap')
        n /= n.std() + 1e-9
        if f > 1:
            yi = np.linspace(0, h - 2, H); xi = np.linspace(0, w - 2, W)
            y0 = yi.astype(int); x0 = xi.astype(int); fy = (yi - y0)[:, None]; fx = (xi - x0)[None]
            n = (n[y0][:, x0] * (1 - fy) * (1 - fx) + n[y0 + 1][:, x0] * fy * (1 - fx)
                 + n[y0][:, x0 + 1] * (1 - fy) * fx + n[y0 + 1][:, x0 + 1] * fy * fx)
        else:
            n = n[:H, :W]
        out += amp * n; tot += amp * amp; amp *= 0.5
    return out / np.sqrt(tot)


class Shape:
    def __init__(self, mask, name=''):
        self.mask = np.asarray(mask, bool)
        self.name = name
        self._sdf = None

    @property
    def sdf(self):
        if self._sdf is None:
            self._sdf = distance_transform_edt(~self.mask) - distance_transform_edt(self.mask)
        return self._sdf

    @property
    def bbox(self):
        ys, xs = np.nonzero(self.mask)
        return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

    def __or__(self, o): return Shape(self.mask | o.mask, self.name)
    def __and__(self, o): return Shape(self.mask & o.mask, self.name)
    def __sub__(self, o): return Shape(self.mask & ~o.mask, self.name)
