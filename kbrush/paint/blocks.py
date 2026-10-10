"""选区、边、软的图、台阶与颗粒、看：给图层（layers.py）用的。
- 选区：blob / lasso / grow / line，都返回 shapes.Shape（能 | & - 拼）。边是噪声推过的，不是几何形。
- 边：edges(选区, hard=, lost=) 定一条轮廓上哪段硬、哪段丢，给 layers.fill(edge=) 用；rim(选区, keep=) 取边上按顺序的一串点。
- 软的图：soft(选区, grow=, feather=, strength=) → 0..1，给 st.layer(mask=) 当羽化过的图层蒙版，也能给 stroke(within=)。
- 台阶与颗粒：smooth(点列) 把手勾的点抹顺；spray(区域, 喷距, rng) 一阶的蒙版，边外喷颗粒；paper(H, W) 整张的纸纹，最后 multiply 罩上。
- 看：look(st, 路径) 存一张，旁边再存一张 64px 的。
下面的 _qcov 那几个是 fill 用的：一块颜色的边干了长什么样，直接按边的种类和纸纹算出来。
"""
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, label
from PIL import Image, ImageDraw

from shapes import Shape, fbm

LOST, SOFT, HARD = 0, 1, 2      # 边：丢、软、硬


# ---------- 选区 ----------
def blob(H, W, cx, cy, rx, ry, ang=0.0, jag=0.12, scale=None, seed=0):
    """一团：椭圆的底，轮廓被噪声推拉 jag（占短半径的比例）。scale 是推拉的尺度（像素，默认短半径的一半）。"""
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    c, s = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    X = ((xx - cx) * c + (yy - cy) * s) / rx
    Y = (-(xx - cx) * s + (yy - cy) * c) / ry
    r = min(rx, ry)
    d = (np.hypot(X, Y) - 1) * r                       # 近似到像素的距离，扁的形长轴那头也不会被推得太狠
    if jag:
        d = d + jag * r * fbm(H, W, scale or r / 2, seed)
    return Shape(d < 0)


def lasso(H, W, pts, jag=5.0, scale=40, seed=0):
    """套索：按顺序连起来的点围成一块，边推拉 jag 像素。pts = [(x, y), ...]。"""
    im = Image.new('L', (W, H), 0)
    ImageDraw.Draw(im).polygon([tuple(map(float, p)) for p in pts], fill=1)
    m = np.asarray(im, bool)
    if not jag:
        return Shape(m)
    sd = distance_transform_edt(~m) - distance_transform_edt(m)
    return Shape(sd + jag * fbm(H, W, scale, seed) < 0)


def grow(H, W, scale, t=0.4, seed=0, within=None):
    """从噪声里长出来的一片：树影、叶团、积水、光斑。t 越大越稀。within：只长在这个选区里。"""
    m = fbm(H, W, scale, seed) > t
    if within is not None:
        m &= within.mask
    return Shape(m)


def edges(sel, hard=None, lost=None, default=SOFT):
    """哪段边硬、哪段丢：hard / lost 是整张纸大小的 bool 图（或选区），落在里面的那段边就是硬 / 丢。给 fill(edge=...) 用。"""
    E = np.full(sel.mask.shape, default)
    for m, v in ((lost, LOST), (hard, HARD)):
        if m is not None:
            E[getattr(m, 'mask', m)] = v
    return E


def rim(sel, keep=None, step=4):
    """选区的边，按顺序的一串点 [(x, y), ...]：贴地线、轮廓线、沿交界的一笔从这里取，别凭眼估坐标。
    keep：整张纸大小的 bool 图（或选区），只要落在里面的那段（取最长的一段）。step：隔几个点取一个。"""
    from skimage.measure import find_contours
    cs = find_contours(np.pad(sel.mask, 1).astype(float), 0.5)
    if not cs:
        return []
    c = max(cs, key=len) - 1                          # (行, 列)，去掉 pad
    if keep is not None:
        km = getattr(keep, 'mask', keep)
        H, W = km.shape
        ok = km[np.clip(c[:, 0].round().astype(int), 0, H - 1), np.clip(c[:, 1].round().astype(int), 0, W - 1)]
        if not ok.any():
            return []
        if ok.all():
            run = c
        else:
            k = int(np.argmin(ok))                        # 从一个不要的点起算，闭合轮廓上跨头尾的那段也算一整段
            c, ok = np.roll(c, -k, 0), np.roll(ok, -k)
            best, cur = (0, 0), None
            for i, v in enumerate(np.append(ok, False)):
                if v and cur is None:
                    cur = i
                elif not v and cur is not None:
                    best = max(best, (cur, i), key=lambda r: r[1] - r[0]); cur = None
            run = c[best[0]:best[1]]
        c = run
    return [(float(x), float(y)) for y, x in c[::step]]


def line(H, W, pts, width=3.0, seed=0, jag=0.0):
    """沿一串点 [(x, y), ...] 的一条线当选区：草叶朝光那一线、茎上的亮毛、花瓣的亮沿。
    width：宽（像素），或 f(t)（t 从头 0 到尾 1），比如 lambda t: 3 * (1 - t) 是越走越细的一线。jag：边推拉几像素。"""
    pts = np.asarray(pts, float)
    hw_max = (max(float(width(t)) for t in np.linspace(0, 1, 11)) if callable(width) else float(width)) / 2 + abs(jag) + 2
    ya = max(int(pts[:, 1].min() - hw_max), 0); yb = min(int(pts[:, 1].max() + hw_max) + 2, H)
    xa = max(int(pts[:, 0].min() - hw_max), 0); xb = min(int(pts[:, 0].max() + hw_max) + 2, W)
    yy, xx = np.mgrid[ya:yb, xa:xb].astype(float)
    dist = np.full(yy.shape, 1e9); along = np.zeros(yy.shape)
    seg = np.diff(pts, axis=0); L = np.hypot(seg[:, 0], seg[:, 1]); cum = np.concatenate([[0], np.cumsum(L)])
    for i in range(len(seg)):
        t = np.clip(((xx - pts[i, 0]) * seg[i, 0] + (yy - pts[i, 1]) * seg[i, 1]) / (L[i] ** 2 + 1e-9), 0, 1)
        d = np.hypot(xx - pts[i, 0] - t * seg[i, 0], yy - pts[i, 1] - t * seg[i, 1])
        c = d < dist
        dist = np.where(c, d, dist); along = np.where(c, cum[i] + t * L[i], along)
    hw = (np.vectorize(lambda t: float(width(t)))(along / max(cum[-1], 1e-9)) if callable(width) else float(width)) / 2
    if jag:
        hw = hw + jag * fbm(yb - ya, xb - xa, 8, seed)
    m = np.zeros((H, W), bool)
    m[ya:yb, xa:xb] = dist < hw
    return Shape(m)


# ---------- 软的图 ----------
def soft(sel, grow=0, feather=0.0, strength=1.0):
    """选区（或 bool 图、0..1 的图）→ 0..1 的图，选区里是 1。给图层蒙版（mask=，1 = 露）或 stroke(within=) 用。
    grow：先往外长（负数往里缩）几像素。feather：羽化宽度（像素，边从一成到九成走这么宽，骑在边上，一半在里一半在外）。
    strength：中间最多到几成。"""
    m = np.asarray(getattr(sel, 'mask', sel))
    if m.dtype == bool and not feather and strength == 1 and grow >= 0:
        return binary_dilation(m, iterations=int(grow)) if grow else m        # 老路子：一模一样
    if m.dtype == bool:
        if grow:
            sd = distance_transform_edt(~m) - distance_transform_edt(m)        # 带符号的距离，里面为负（边上的格子 ±1）
            m = sd <= grow if grow > 0 else sd < grow                         # 长 1 = 外面挨着的一圈；缩 1 = 去掉里面最外一圈
        m = m.astype(float)
    else:
        m = np.clip(m.astype(float), 0, 1)
        if grow > 0:
            from scipy.ndimage import grey_dilation
            r = int(round(grow)); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            m = grey_dilation(m, footprint=xx * xx + yy * yy <= grow * grow)
        elif grow < 0:
            from scipy.ndimage import grey_erosion
            r = int(round(-grow)); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            m = grey_erosion(m, footprint=xx * xx + yy * yy <= grow * grow)
    if feather:
        from scipy.ndimage import gaussian_filter
        m = gaussian_filter(m, feather / 2.563)                               # 一成到九成正好 feather 宽
    return np.clip(m, 0, 1) * strength


# ---------- 台阶与颗粒 ----------
def smooth(pts, n=3, closed=True):
    """手勾的点列 [(x, y), ...] 抹顺：切角 n 遍，每遍点数翻一倍。closed=False 是一条开的线，两头不动。"""
    p = np.array(pts, float)
    for _ in range(n):
        a, b = (p, np.roll(p, -1, 0)) if closed else (p[:-1], p[1:])
        q = np.stack([.75 * a + .25 * b, .25 * a + .75 * b], 1).reshape(-1, 2)
        p = q if closed else np.vstack([p[:1], q, p[-1:]])
    return [tuple(v) for v in p]


def spray(region, sp, rng, solid=.96, dens=.7, holes=0.):
    """一阶的蒙版（0..1）：区域里实实地盖上 solid，边外喷出颗粒，越远越稀，大约每 sp 像素稀一截。
    sp、solid 都可以是整张的图（各处喷多远、盖几成不一样）。dens：贴着边那圈多密。
    holes：阶里留几成针眼，底下那阶从针眼里透出来。
    rng：种子，或者画里那支 np.random.Generator（传进来，随机数就接着画里的顺序走）。"""
    region = np.asarray(getattr(region, 'mask', region), bool)
    r = np.random.default_rng(rng)
    dout = distance_transform_edt(~region)
    dots = (r.random(region.shape) < dens * np.exp(-dout / np.maximum(sp, .5))) & (gaussian_filter(r.random(region.shape), .5) > .47)
    if holes:
        g = gaussian_filter(r.random(region.shape), .7); g = (g - g.mean()) / g.std()
        solid = np.where(g > np.quantile(g, 1 - holes), .25, solid)
    return np.where(region, solid, dots * .95)


def paper(H, W, seed=77):
    """整张一张纸纹（H, W, 3），画完最后罩上去：fill(st.layer('纸', mode='multiply'), 全图, paper(H, W), edge=HARD)。
    牙口（细的凸起）加一点横的纤维；每样东西别各撒各的噪点。"""
    r = np.random.default_rng(seed)
    tooth = gaussian_filter(r.standard_normal((H, W)), .7); tooth /= tooth.std()
    fiber = gaussian_filter(r.standard_normal((H, W)), (.6, 3)); fiber /= fiber.std()
    return np.repeat(np.clip(1 - .06 * np.clip(tooth, 0, None) - .025 * fiber, 0, 1)[..., None], 3, 2)


# ---------- 看 ----------
def look(st, path, small=64):
    """render 存一张，旁边再存一张缩到长边 small 的（眯眼看：三四块还读得出来吗）。render 已是显示值，别再做 sRGB。"""
    img = (np.clip(st.render(), 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(img)
    im.save(path)
    H, W = img.shape[:2]
    s = small / max(H, W)
    th = im.resize((max(1, round(W * s)), max(1, round(H * s))), Image.LANCZOS)
    th.resize((th.width * 4, th.height * 4), Image.NEAREST).save(path.rsplit('.', 1)[0] + f'_{small}.png')
    return img


# ---------- fill 的边：干了的样子直接算 ----------
# 一块颜色干了以后边长什么样，按边的种类、纸纹、落点直接算出来，一步到位。
#   off：颜色淡到一半的那条线在选区边外几像素；fw：从九成淡到一成走几像素；rag：边被推拉多少像素
QUICK = {HARD: dict(off=0.3, fw=1.3, rag=1.0),
         SOFT: dict(off=2.0, fw=12.0, rag=2.5),
         LOST: dict(off=9.0, fw=22.0, rag=6.0)}
# grain= 的颗粒长什么样：两层细的平滑噪声（三像素上下的团），再跟纸纹的凹处兑在一起，落在纸的坑里。
# ponytail: 尺度是手挑的，不是从真画上量的；手上有量好的径向功率谱，可以换成"谱 + 随机相位"合成。


def _pgrain(h, w, rng):
    """(h, w) 的颗粒纹理（均值 0、方差 1）。"""
    return _z(_qnoise(h, w, 3, rng, octaves=2))


def _qnoise(h, w, scale, rng, octaves=1):
    """快的平滑噪声（均值 0、方差约 1），特征尺寸 scale 像素：在小图上生成，PIL 双三次放大。"""
    out = np.zeros((h, w)); amp = 1.0; tot = 0.0
    for o in range(octaves):
        s = max(scale / 2 ** o, 0.6)
        f = max(1, int(s / 3))
        hs, ws = h // f + 3, w // f + 3
        n = gaussian_filter(rng.standard_normal((hs, ws)).astype(np.float32), s / f / 2.5)
        if f > 1:
            n = np.asarray(Image.fromarray(n, 'F').resize((ws * f, hs * f), Image.BICUBIC))
        n = n[:h, :w].astype(float)
        out += amp * (n - n.mean()) / (n.std() + 1e-9); tot += amp * amp; amp *= 0.5
    return out / np.sqrt(tot)


def _z(a):
    return (a - a.mean()) / (a.std() + 1e-9)


def padded_sdf(mask, pad=48):
    """形的距离场，纸外面当成形里面：形碰到纸边的地方不算边（颜色一直铺出纸外）。"""
    m = np.pad(mask, pad, mode='edge')
    sd = distance_transform_edt(~m) - distance_transform_edt(m)
    return sd[pad:-pad, pad:-pad]


def _qedge(m, E, default, keys=('off', 'fw', 'rag')):
    """窗口里每个像素按离它最近的那段边取参数（QUICK 表）。没有 edges() 图、就是一种边：返回常数。"""
    ring = binary_dilation(m, structure=np.ones((3, 3), bool)) & ~m
    if not ring.any() or E is None:
        return {k: QUICK[default][k] for k in keys}, ring
    _, (iy, ix) = distance_transform_edt(~ring, return_indices=True)
    out = {}
    for k in keys:
        v = np.select([E == HARD, E == SOFT, E == LOST], [QUICK[HARD][k], QUICK[SOFT][k], QUICK[LOST][k]], QUICK[default][k])
        out[k] = gaussian_filter(v[iy, ix].astype(float), 3.0)      # 换边的地方顺着边匀一匀
    return out, ring


def _qreach(sel, E, default, wmul, rag):
    """fill 的边最远淡出到选区外多少像素（窗口往外扩这么多）。"""
    types = {default} | (set(np.unique(E[sel.mask ^ binary_dilation(sel.mask)]).tolist()) if E is not None else set())
    return max(QUICK[k]['off'] * wmul + 1.8 * QUICK[k]['fw'] * wmul + 3 * QUICK[k]['rag'] * rag for k in types) + 6


def _pinholes(m):
    """选区里一两像素的针眼（噪声推出来的）补上，不留白点；故意挖的洞（几十像素以上）照留。"""
    if (~m).any():
        lab, n = label(~m)
        if n > 1:
            sz = np.bincount(lab.ravel()); sz[0] = 0
            border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
            sz[border] = 0
            small = np.nonzero((sz > 0) & (sz < 30))[0]
            if len(small):
                m = m | np.isin(lab, small)
    return m


def _qcov(m, E, default, wmul, rag, fibz, rng):
    """fill 的边：窗口里的选区 m → 每个像素盖住几成（0..1）。
    E：窗口里的 edges() 图（没有就 None），fibz：窗口里的纸纤维（_z(log perm)）。"""
    h, w = m.shape
    fibs = _z(gaussian_filter(fibz, 0.7))                                       # 纤维，抹掉一像素的硬棱（不然淡出去那截像毛）
    P, ring = _qedge(m, E, default)
    if not ring.any():
        return np.ones((h, w))
    sd = padded_sdf(m, pad=16)
    sc = {k: (P[k] * wmul if k in ('off', 'fw') else P[k]) for k in P}
    sc['fw'] = np.maximum(sc['fw'], 0.8)
    vv = np.exp(0.45 * _qnoise(h, w, 90, rng))                              # 边上一段毛、一段光
    push = 0.55 * _qnoise(h, w, 70, rng, 2) + 0.85 * _qnoise(h, w, 30, rng, 3)
    x = sc['off'] - (sd + sc['rag'] * rag * vv * push)                      # 离"淡到一半"那条线往里多远（像素）
    # 淡出的那一截不是一圈匀的雾：有的方向淡得远（中尺度的起伏，越往外越大），顺着纤维的再远一点（细的毛），
    # 硬边只剩纸的牙口
    out = np.clip(-x / sc['fw'], 0, 2.5)
    reachn = _qnoise(h, w, 18, rng, 2)
    fine = _z(gaussian_filter(rng.standard_normal((h, w)), 0.7))
    x = (x + sc['fw'] * 0.16 * reachn * np.clip(out + 0.3, 0, 1.2) * (sc['fw'] > 3)
         + (0.1 + 0.03 * sc['fw']) * fibs * (1 + 0.5 * out) + 0.25 * fine)
    return 1 / (1 + np.exp(np.clip(-4.39 * x / sc['fw'], -50, 50)))
