"""我们自己的笔（kbrush）：一支有状态的笔 + 一张有纹理的纸 + 一只会抖的手。颜色层（layers.py）上画画就靠它。

一笔 = 蘸多少 × 压多重 × 走多快 × 怎么落怎么收。这些不是手调的遮罩，是笔、纸、手各自的状态，效果是它们碰出来的：
- 手（gesture）：路径会弯、会抖；起笔有露锋/藏锋/顿，收笔有出锋/回锋/顿收/提。勾线笔的笔尖拖在手后面（trail）。
- 笔（Brush）：鬃在纸上踩出的是一个二维的脚印——平头笔一排排，圆头笔压下去是个泪滴，越按越大。
  边上的鬃压得轻，所以边是毛的。每根鬃自己带颜料、多少不一样；分笔尖和笔肚子两段，放出去的是笔尖的。
  饱的时候颜料靠"流"下去（跟停留时间有关），干了只能靠"刮"（跟碰到多少纸纹有关），所以干笔越拖越少、越拖越碎，一根一根断掉。
- 纸（Paper）：有凹凸（纸纹）和纤维。干笔只碰得到纸纹的顶。
stroke() 返回一笔落下的颜料量，layers.stroke 把它当覆盖量，颜色直接给。
"""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


def smooth_noise_1d(n, knots, rng):
    """长度 n 的平滑随机曲线（均值 0、幅度约 1），knots 个起伏。"""
    k = rng.standard_normal(max(knots, 2) + 3)
    x = np.linspace(0, len(k) - 1, n)
    y = np.interp(x, np.arange(len(k)), k)
    y = gaussian_filter(y, n / max(knots, 2) / 3)
    return y / (y.std() + 1e-9)


class Paper:
    """纸：h 纸纹凹凸（0 凹 … 1 凸），hu 按高低排的名次（0..1 均匀），perm 纤维（fill 的软边顺着它毛），
    grain 纸面上更细的小坑（grain= 的颗粒落在这里），color 纸色。"""

    def __init__(self, H, W, seed=0, color=(0.96, 0.94, 0.89), fibers=1.0):
        self.H, self.W = H, W
        rng = np.random.default_rng(seed)
        n1 = gaussian_filter(rng.standard_normal((H, W)), 1.0)
        n2 = gaussian_filter(rng.standard_normal((H, W)), 3.5)
        # 纤维：一根根随机方向的细丝
        F = np.zeros((H, W))
        nf = int(H * W / 45 * fibers)
        L = rng.uniform(6, 30, nf)
        th = rng.uniform(0, np.pi, nf)
        cx = rng.uniform(0, W, nf); cy = rng.uniform(0, H, nf)
        for k in range(0, 31, 2):
            t = (k / 30 - 0.5)
            xs = np.clip((cx + np.cos(th) * L * t).astype(int), 0, W - 1)
            ys = np.clip((cy + np.sin(th) * L * t).astype(int), 0, H - 1)
            np.add.at(F, (ys, xs), 1.0)
        F = gaussian_filter(F, 0.7)
        F = (F - F.mean()) / (F.std() + 1e-9)
        h = n1 / n1.std() * 0.5 + n2 / n2.std() * 0.6 + F * 0.3
        self.h = (h - h.min()) / (h.max() - h.min())
        # 纸纹按高低排名（0..1 均匀）：depth=0.3 就是碰到最高的三成纸面
        r = np.empty(H * W); r[np.argsort(h, axis=None)] = np.linspace(0, 1, H * W)
        self.hu = r.reshape(H, W)
        perm = np.exp(0.55 * F + 0.25 * gaussian_filter(rng.standard_normal((H, W)), 6) * 4)
        self.perm = np.clip(perm / perm.mean(), 0.15, 2.0)
        # 纸面上细小的坑（比纸纹的起伏细），所以颗粒是沙粒一样的，不是皱纹
        g1 = gaussian_filter(rng.standard_normal((H, W)), 0.7)
        gr = g1 / g1.std() + 0.5 * n1 / n1.std()
        self.grain = (gr - gr.min()) / (gr.max() - gr.min())
        self.color = np.array(color)


_BASE = np.array([1.0, 0.0])     # 笔里的颜料分两份：[0] 蘸的颜色，[1] dip_tip 点在笔尖的第二色
_TIP = np.array([0.0, 1.0])
PRESS = 0.6 * 0.2                # 饱的鬃压过的地方，当场压进纸纹的那一份（每碰一下的比例）：鬃的细纹就是它留下的


class Brush:
    """一支笔。每根鬃自己带颜料。
    kind：'flat' 平头（铺大块、切边）、'round' 圆头（点、勾）、
          'rigger' 勾线笔（毛又细又长：肚子里存得多、放得慢，线拉得很长；笔尖拖在手后面，配 trail() 用）、
          'mop' 大羊毫（又软又胖、存得多，一次铺开一大片）、
          'fan' 扇形笔（薄薄一排、稀稀的几撮，毛梢排成一道弧：一笔是很多根细线，画草、树梢）。
    cap：存得下多少，是同样粗细的普通笔的几倍（勾线笔、大羊毫默认大）。"""

    def __init__(self, kind='flat', width=70, n=None, seed=1, missing=0.03, clumps=None, cap=None):
        rng = np.random.default_rng(seed)
        self.kind, self.width = kind, width
        per = {'flat': 1.6, 'round': 2.4, 'rigger': 4.0, 'mop': 2.4, 'fan': 0.9}[kind]
        n = n or max(int(width * per), 12)
        self.n = n
        cap = cap if cap is not None else {'rigger': 4.0, 'mop': 2.5}.get(kind, 1.0)
        if kind == 'fan':
            # 扇形笔：一排稀稀的毛。饱的时候粘成一撮一撮（撮和撮之间是空的），干了又散成一根一根
            clumps = clumps or max(6, width // 6)
            self.u = np.clip(np.linspace(-1, 1, n) + rng.normal(0, 0.8 / n, n), -1, 1)
            rng.shuffle(self.u)
            cut = np.sort(np.linspace(-1, 1, clumps + 1)[1:-1] + rng.normal(0, 0.35 / clumps, clumps - 1))
            cid = np.searchsorted(cut, self.u)
            self.d = rng.random(n) * 0.3
            self.len = (1 - 0.3 * self.u ** 2) * rng.uniform(0.85, 1.0, n)   # 中间的毛长，毛梢是一道弧
        else:
            # 横着在笔宽里的位置：大致均匀铺开（真笔有上千根毛，不会随机空出一条缝）
            self.u = np.clip(np.linspace(-1, 1, n) + rng.normal(0, 1.2 / n, n), -1, 1)
            rng.shuffle(self.u)
            self.d = rng.random(n) if kind == 'flat' else rng.random(n) ** 0.8   # 纵深：笔尖 0 … 笔肚子 1
            self.len = rng.uniform(0.78, 1.0, n)              # 鬃长短不齐
            self.len[rng.random(n) < missing] = 0.7           # 几根短毛
            clumps = clumps or max(5, width // 8)
            cut = np.sort(rng.uniform(-1, 1, clumps - 1))
            cid = np.searchsorted(cut, self.u)
        self.lm = np.exp(rng.normal(0, 0.45, n))          # 每根吃进去的颜料多少不一样
        rng.normal(0, 0.6, n)                             # 空抽一次：后面每根鬃的随机数跟以前一样（同一个 seed 还是同一支笔）
        self.tipfrac = rng.uniform(0.07, 0.16, n)         # 笔尖占多少
        self.rr = rng.uniform(0.25, 1.0, n)               # 笔肚子往笔尖补得快慢
        centers = np.array([self.u[cid == k].mean() if np.any(cid == k) else 0 for k in range(clumps)])
        self.clump_u = centers[cid]
        self.cid = cid                                     # 属于哪一撮
        self.spread = 0.12 if kind == 'fan' else 0.25     # 落下的颜料往两边匀多少（扇形笔的毛是一根一根的细线）
        self.cph = rng.uniform(0, 2 * np.pi, (2, clumps))  # 每一撮整体慢慢往旁边漂、时聚时散
        self.cfq = rng.uniform(0.008, 0.03, (2, clumps))
        self.hn = rng.normal(0, 1, n)                     # 同一撮里每根鬃高低也不一样（碰纸早晚）
        self.ph = rng.uniform(0, 2 * np.pi, (2, n))
        self.fq = np.stack([rng.uniform(0.004, 0.012, n), rng.uniform(0.02, 0.05, n)])
        self.jit = rng.normal(0, 1, n)                    # 笔尖参差（前后）
        self.k = 1.6 * width / n                          # 鬃多的笔每根少放一点
        self.order = np.argsort(self.u)
        self.Wbase = self.len.copy()                      # 一根普通的鬃吃得下多少（放得快慢、笔尖多大按它算）
        self.Wcap = self.len * cap                        # 这支笔一根鬃实际吃得下多少（长毛、胖肚子存得多）
        self.Wt = np.zeros(n); self.Wb = np.zeros(n)
        self.Pt = None; self.Pb = None
        self.paste = 0.0

    def dip(self, load=1.0, water=1.0, paste=None):
        """蘸（颜色层上用 layers.dip，它顺手记下颜色）。load 是蘸多浓，water 是笔吃了几成饱（0..1，0.25 上下是干笔）。
        paste：在碟边刮干了、鬃表面糊着膏（干笔法）。不给就按有多饱猜：
        按一根普通鬃算——勾线笔、大羊毫吃两三成饱也还是饱的。"""
        eff = min(water * float(np.mean(self.Wcap / self.Wbase)), 1.0)
        # 蘸饱了每根差不多满；在碟边刮干了，剩多剩少就很不均匀
        W = np.clip(water * self.Wcap * self.lm ** (0.25 + 0.9 * (1 - eff)), 0, self.Wcap)
        conc = load * self.lm ** 0.4
        P = (W * conc)[:, None] * _BASE[None]
        # 蘸饱的笔：颜料多半在笔肚子里，笔尖只是表面那一层；
        # 刮干了的笔（干笔法）：肚子里挤空了，剩下的颜料都糊在鬃的表面上
        tip = np.minimum(W, np.maximum(self.tipfrac * self.Wbase, (1 - eff) * np.minimum(W, self.Wbase)))
        r = tip / (W + 1e-12)
        self.Wt, self.Wb = tip, W - tip
        # 刮干的笔上是膏，碰到纸纹就被刮下去；饱的笔上是流得动的颜料
        self.paste = float(np.clip((0.5 - eff) / 0.4, 0, 1)) if paste is None else float(paste)
        self.Pt, self.Pb = P * r[:, None], P * (1 - r)[:, None]

    def dip_tip(self, load=1.5, depth=0.35, part='tip'):
        """蘸完一种颜色以后，笔尖（或平头笔的一个角）再点一下另一种（layers.dip_tip）。
        一笔里就有了深浅：圆头笔中锋画出来中间深两边浅，侧锋时深的那一边贴着笔尖那一侧；
        越画笔肚子里的浅色越往外补，深色就一路淡下去。part='tip' 笔尖那一截，'corner' 平头笔 u>0 那个角。"""
        if part == 'tip':
            k = np.clip((depth - self.d) / 0.12 + 0.5, 0, 1)
        else:
            k = np.clip((self.u - (1 - 2 * depth)) / 0.25 + 0.5, 0, 1)
        conc = load * self.lm ** 0.4
        self.Pt = self.Pt * (1 - 0.8 * k)[:, None] + (self.Wt * conc * 0.8 * k)[:, None] * _TIP[None]
        self.Pb = self.Pb * (1 - 0.25 * k)[:, None] + (self.Wb * conc * 0.25 * k)[:, None] * _TIP[None]

    def sat(self):
        return np.clip((self.Wt + self.Wb) / self.Wbase, 0, 1.2)

    def footprint(self, p, u, roll=0.0, side=0.0):
        """这一刻每根鬃踩在哪（横 across、纵 along，像素）以及够不够得着纸（reach 0..1）。
        roll：手腕往一边侧（平头笔一个角先着纸、一个角后离纸），正数是 u>0 那边压得重。
        side：圆头笔的侧锋（0 中锋 … ±1 笔尖完全躺到一边），泪滴转过去，笔尖那侧光、笔肚那侧毛。"""
        W = self.width
        if self.kind == 'flat':
            pe = p * (1 - 0.35 * self.u ** 2) * np.clip(1 + roll * self.u, 0, 2)   # 边上的鬃压得轻；侧着的那边更轻
            hw = W / 2 * (0.72 + 0.28 * p)
            across = u * hw
            Lc = W * 0.1 * (0.25 + p)
            along = (self.d - 0.5) * Lc + self.jit * (1 - self.len) * W * 0.12
            reach = pe * 1.35 * self.len - 0.12
        elif self.kind == 'fan':
            pe = p * np.clip(1 + roll * self.u, 0, 2)
            hw = W / 2 * (0.85 + 0.2 * p)                 # 按下去扇面张开一点
            across = u * hw
            along = -0.12 * W * self.u ** 2 + self.jit * 0.8   # 毛梢是一道弧：中间的伸得最远
            reach = pe * 1.3 * self.len - 0.25            # 边上的毛短，要按重一点才够得着
        else:
            # 圆头一类：压下去是个泪滴。tw 笔尖多尖，lk 躺下去多长，bl 肚子多胖
            tw, lk, bl = {'round': (0.06, 0.85, 0.55), 'rigger': (0.1, 2.6, 0.4), 'mop': (0.18, 0.7, 0.35)}[self.kind]
            pe = np.clip(p, 0, 1) ** 0.85                 # 按到多深
            inside = self.d < pe + 0.02
            rel = np.clip(self.d / (pe + 1e-6), 0, 1)
            hw = W / 2 * (tw + (1 - tw) * pe)
            across = u * hw * rel ** bl                 # 泪滴：笔尖窄、笔肚子宽
            Lc = W * lk * pe + 2
            along = (rel - 0.35) * Lc + self.jit * 0.6
            reach = np.where(inside, (0.35 + 0.8 * self.len) * (1.05 - 0.4 * rel * (1 - pe)), 0) - 0.05
            if side:
                th = side * np.pi / 2
                c, s_ = np.cos(th), np.sin(th)
                across, along = across * c + along * s_, along * c - across * s_
        return across, along, np.clip(reach, 0, 1)


def _splat(A, ys, xs, vals):
    """把 vals（n,）或（k,n）双线性地撒到 A 上。"""
    H, W = A.shape[-2:]
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    fx = xs - x0; fy = ys - y0
    for dy, dx, wt in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)),
                       (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
        yy = np.clip(y0 + dy, 0, H - 1); xx = np.clip(x0 + dx, 0, W - 1)
        if A.ndim == 2:
            np.add.at(A, (yy, xx), vals * wt)
        else:
            for k in range(A.shape[0]):
                if np.any(vals[k]):
                    np.add.at(A[k], (yy, xx), vals[k] * wt)


def _catmull(ctrl, n_per=40):
    """过这些点的一条顺滑曲线（向心 Catmull-Rom：不打结、不甩出去）。"""
    P = np.vstack([2 * ctrl[0] - ctrl[1], ctrl, 2 * ctrl[-1] - ctrl[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.hypot(*(p1 - p0)), 1e-3) ** 0.5
        t2 = t1 + max(np.hypot(*(p2 - p1)), 1e-3) ** 0.5
        t3 = t2 + max(np.hypot(*(p3 - p2)), 1e-3) ** 0.5
        t = np.linspace(t1, t2, n_per, endpoint=False)[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    out.append(ctrl[-1][None])
    return np.vstack(out)


def trail(pts, L, step=1.0):
    """勾线笔的笔尖拖在手后面：手往前走，笔尖只会被毛拽着、朝手的方向滑过去（曳物线）。
    所以手拐急弯时笔尖抄近路划出一道弧，手上细碎的抖动也被长毛滤掉了。
    pts 是手走的路，返回笔尖走的路（手在最后多送出去一截毛长，笔尖才到得了终点）。"""
    pts = np.asarray(pts, float)
    de = pts[-1] - pts[-2]
    de = de / (np.hypot(*de) + 1e-9)
    ext = np.vstack([pts, pts[-1] + de * L])
    seg = np.hypot(*np.diff(ext, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.arange(0, cum[-1], step)
    hx = np.interp(s, cum, ext[:, 0]); hy = np.interp(s, cum, ext[:, 1])
    qx, qy = pts[0]
    out = [(qx, qy)]
    for x, y in zip(hx, hy):
        dx, dy = x - qx, y - qy
        n = np.hypot(dx, dy)
        if n > L:
            qx, qy = x - dx / n * L, y - dy / n * L
            out.append((qx, qy))
    return np.array(out)


def gesture(p0, p1, width=70, bend=0.0, start='lu', end='ti', pressure=0.8, speed=1.0,
            wobble=1.0, seed=0, via=None):
    """一只手画的一笔：弯、抖、起笔、收笔。返回 stroke() 要的 pts / pressure / speed。

    start: 'lu' 露锋（笔尖轻轻落下再铺开）/ 'cang' 藏锋（先逆着进去再回头）/ 'dun' 顿（按下停一下）
    end:   'chu' 出锋（边走边提，尾巴尖、会飞白）/ 'hui' 回锋（往回收）/ 'dun' 顿收（按住停下）/ 'ti' 提（干脆提起）
    via:   中间点列表（折线/曲线路径）；bend: 整体弯多少（占长度的比例）
    pressure: 常数；或沿笔画的一串数（从起笔到收笔，均匀排开）；或函数 f(t)，t 从 0（起笔）到 1（收笔）。
           越来越细的藤、茎：pressure=lambda t: 0.9 - 0.6 * t。起笔收笔的轻重照样乘在上面。
    """
    rng = np.random.default_rng(seed)
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    ctrl = [p0] + [np.asarray(v, float) for v in (via or [])] + [p1]
    ctrl = np.array(ctrl)
    if via:
        ctrl = _catmull(ctrl)      # 过中间点的顺滑曲线，不是折线
    seg = np.hypot(*np.diff(ctrl, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    Lt = cum[-1]
    N = max(int(Lt / 3), 12)
    s = np.linspace(0, Lt, N)
    base = np.stack([np.interp(s, cum, ctrl[:, 0]), np.interp(s, cum, ctrl[:, 1])], 1)
    d = p1 - p0
    tdir = d / (np.hypot(*d) + 1e-9)
    ndir = np.array([-tdir[1], tdir[0]])
    u = s / Lt
    base += np.outer(4 * u * (1 - u) * bend * Lt, ndir)
    # 手抖：两种频率的慢漂
    base += np.outer(smooth_noise_1d(N, max(2, int(Lt / 160)), rng) * 1.6 * wobble
                     + smooth_noise_1d(N, max(3, int(Lt / 40)), rng) * 0.45 * wobble, ndir)
    pts = base
    r = width * 0.3
    if start == 'cang':   # 藏锋：先往反方向探一下再折回来
        pre = np.array([p0 + tdir * r * 0.9 + ndir * r * 0.25, p0 + tdir * r * 0.3 + ndir * r * 0.35,
                        p0 - tdir * r * 0.15 + ndir * r * 0.1])
        pts = np.vstack([pre, pts])
    if end == 'hui':      # 回锋：到头往回兜一下
        e = pts[-1]
        post = np.array([e + tdir * r * 0.25 - ndir * r * 0.2, e - tdir * r * 0.1 - ndir * r * 0.4,
                         e - tdir * r * 0.55 - ndir * r * 0.3])
        pts = np.vstack([pts, post])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    L = seg.sum()
    tk = np.linspace(0, 1, 200)
    if callable(pressure):
        pressure = np.broadcast_to(np.asarray(pressure(tk), float), tk.shape)
    elif np.ndim(pressure):
        pr = np.asarray(pressure, float).ravel()
        pressure = np.interp(tk, np.linspace(0, 1, len(pr)), pr) if len(pr) > 1 else np.full(200, pr[0])
    P = pressure * (1 + 0.07 * smooth_noise_1d(200, 4, rng))
    V = speed * (1 + 0.1 * smooth_noise_1d(200, 3, rng))
    a = min(0.18, 55 / L)
    if start == 'lu':
        P *= np.clip(0.06 + 0.94 * (tk / a) ** 0.7, 0, 1)
    elif start == 'dun':
        a0 = min(0.08, 22 / L)
        P = np.where(tk < a0, np.minimum(1.0, pressure * 1.25), P)
        V = np.where(tk < a0, speed * 0.3, V)
    elif start == 'cang':
        a0 = min(0.12, (r * 1.6) / L)
        P = np.where(tk < a0, pressure * 0.75, P)
        V = np.where(tk < a0, speed * 0.5, V)
    if end == 'chu':
        b = min(0.35, 220 / L)
        k = np.clip((1 - tk) / b, 0, 1)
        P *= k ** 1.2
        V *= 1 + 0.9 * (1 - k)
    elif end == 'ti':
        b = min(0.05, 16 / L)
        P *= np.clip((1 - tk) / b, 0, 1) ** 0.6
    elif end == 'dun':
        b = min(0.07, 20 / L)
        P = np.where(tk > 1 - b, np.minimum(1.0, pressure * 1.2), P)
        V = np.where(tk > 1 - b, speed * 0.25, V)
        P *= np.clip((1 - tk) / 0.01, 0, 1)
    elif end == 'hui':
        b = min(0.12, (r * 1.2) / L)
        k = np.clip((1 - tk) / b, 0, 1)
        P = np.where(tk > 1 - b, P * (0.4 + 0.6 * k), P)
        V = np.where(tk > 1 - b, V * 0.6, V)
    P = np.clip(P, 0, 1)
    # 手腕侧倾：平时有一点自己的习惯、慢慢晃；落笔时一个角先着纸，出锋时一个角最后离开
    side = rng.choice([-1, 1])
    Rl = rng.normal(0, 0.1) + 0.1 * smooth_noise_1d(200, 3, rng)
    if start in ('lu', 'cang'):
        a1 = min(0.25, 80 / L)
        Rl = Rl + side * 0.7 * np.clip(1 - tk / a1, 0, 1) ** 1.5
    if end == 'chu':
        b1 = min(0.35, 220 / L)
        Rl = Rl - side * 0.6 * np.clip(1 - (1 - tk) / b1, 0, 1) ** 1.5
    return dict(pts=pts, pressure=lambda t: np.interp(t, tk, P), speed=lambda t: np.interp(t, tk, V),
                roll=lambda t: np.interp(t, tk, Rl))


_SEED = None          # set_seed 给的根；None = 没设，没写 seed 的地方照旧每次不一样
_SEED_N = 0           # 从根上派生了几次


def set_seed(n):
    """调用以后，没写 seed 的地方（眼下就是 stroke：它默认 seed=None，每次落笔都不一样）按调用次序从 n 派生确定的种子，
    同一个脚本跑两遍逐像素一样。set_seed(None) 关掉。只管 kbrush 自己；写了 seed 的照用写的，默认 seed=0 的那些本来就是定的。"""
    global _SEED, _SEED_N
    _SEED, _SEED_N = (None if n is None else int(n)), 0


def _auto_seed(seed):
    """seed 写了就用写的；没写、又 set_seed 过，就派生第 k 个（[根, k]）；都没有就 None（每次不一样）。"""
    global _SEED_N
    if seed is not None or _SEED is None:
        return seed
    _SEED_N += 1
    return [_SEED, _SEED_N]


def stroke(paper, brush, pts, pressure=1.0, speed=1.0, roll=0.0, side=0.0, hold=None, step=0.6, gain=560.0, wgain=1.6,
           a_wet=2.2e-4, a_abr=1.2e-2, bridge=0.15, hcap=0.45, seed=None, streak=1.0):
    """沿 pts 在 paper 上拖一笔，返回落下的颜料 (2, H, W)：[0] 蘸的颜色，[1] dip_tip 点的第二色（layers.stroke 当覆盖量用）。
    笔里还剩多少、笔尖和笔肚子各剩多少留在 brush 上，同一支笔接着拖下一笔就接着用。
    pressure / speed / roll / side 可以是常数或 t(0..1) 的函数（K.gesture 给的就是函数）。
    hold：平头笔握的角度（度）。None = 笔宽永远横在路径上；给定角度 = 手腕不转，转弯时自己变粗细。
    seed：不写就每次不一样；K.set_seed(n) 过就按落笔次序派生，跑两遍一样。
    streak：干了以后碎成什么样。1（默认）= 干刷：只认细的纸纹，毛和毛高低差得开（有的根本碰不到纸），碰上了就贴着滑一段，
            碎是顺着笔走的长条。0 = 只碰纸纹最大那层起伏的包顶，每根毛落在同一个包上，碎成圆斑（像海绵）。
    gain / wgain / a_wet / a_abr / bridge / hcap 是笔的手感常数（换算、饱笔放得多快、干笔刮得多快、干笔跨坑、纸纹顶挂多少），平时不动。
    """
    rng = np.random.default_rng(_auto_seed(seed))
    pts = np.asarray(pts, float)
    seg = np.diff(pts, axis=0)
    Ls = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0], np.cumsum(Ls)])
    total = cum[-1]
    nS = max(int(total / step), 2)
    s = np.linspace(0, total, nS)
    px = np.interp(s, cum, pts[:, 0]); py = np.interp(s, cum, pts[:, 1])
    tx = gaussian_filter(np.gradient(px), 2, mode='nearest'); ty = gaussian_filter(np.gradient(py), 2, mode='nearest')
    tn = np.hypot(tx, ty) + 1e-9
    tx /= tn; ty /= tn
    if hold is None:
        nx, ny = -ty, tx
    else:
        a = np.deg2rad(hold)
        nx, ny = np.full(nS, np.cos(a)), np.full(nS, np.sin(a))
    t = s / total
    pr = pressure(t) if callable(pressure) else np.full(nS, float(pressure))
    sp = speed(t) if callable(speed) else np.full(nS, float(speed))
    ro = roll(t) if callable(roll) else np.full(nS, float(roll))
    sd = side(t) if callable(side) else np.full(nS, float(side))
    hu = paper.hu
    if streak:
        hf = getattr(paper, '_hu_fine', None)
        if hf is None:                       # 细纸纹：去掉十来像素的大起伏，只留一两像素的牙口，再按高低排名
            fine = paper.h - gaussian_filter(paper.h, 2.5)
            hf = np.empty(fine.size); hf[np.argsort(fine, axis=None)] = np.linspace(0, 1, fine.size)
            hf = hf.reshape(fine.shape); paper._hu_fine = hf
        # 每根毛沿着自己走的路一段贴纸、一段跳起来（约 10–25px 一段），再被细纸纹啃一啃边：碎是顺着笔的长条
        from scipy.stats import norm as _norm
        _nk = max(int(total / 14), 2)
        U = np.stack([_norm.cdf(smooth_noise_1d(nS, int(_nk * rng.uniform(0.6, 1.4)), rng)) for _ in range(brush.n)])
        hu_old = hu
        bridge = bridge * (1 + 1.5 * streak)
    b = brush
    H, W = hu.shape
    act = np.nonzero((b.Pt + b.Pb).sum(0) > 0)[0]    # 笔里有的那几份颜色
    # 落下的颜料分两份，最后加起来：D 是鬃直接压进纸纹的（干刷"挂满"的上限按它算），S 是饱的鬃留下的那层颜料
    S_a = np.zeros((len(act), H, W)); D_a = np.zeros((len(act), H, W))
    Pt = b.Pt[:, act].copy(); Pb = b.Pb[:, act].copy()
    wet = np.zeros((H, W))         # 这一笔自己走过、还没干的湿痕：后面的鬃再过来，连纸纹凹处也涂满（饱笔盖得匀就是它）
    Dst = np.zeros((H, W))         # 这一笔直接落在纸上的颜料（干刷挂满了就挂不上了）
    prev = np.zeros(b.n)
    for i in range(nS):
        p = pr[i]; v = max(sp[i], 0.15)
        # 饱不饱按一根普通鬃算：肚子里存得多的笔（勾线、大羊毫）一直是饱的，存货快见底了才开始碎
        sat = np.clip((b.Wt + b.Wb) / b.Wbase, 0, 1.2)
        # 鬃在走：慢慢晃、干了往各自那撮挤、越干越散
        dry = np.clip(1 - sat, 0, 1)
        if b.kind == 'fan':      # 扇形笔反过来：饱的时候粘成撮，干了散开
            clump = 0.95 * np.clip((sat - 0.25) / 0.35, 0, 1) * (0.9 + 0.1 * np.sin(b.cph[1, b.cid] + b.cfq[1, b.cid] * s[i]))
        else:
            clump = 0.45 * dry ** 1.5 * (0.65 + 0.35 * np.sin(b.cph[1, b.cid] + b.cfq[1, b.cid] * s[i]))
        cu = (1 - clump) * b.u + clump * b.clump_u
        across, along, reach = b.footprint(p, cu, ro[i], sd[i])
        across = across + np.sin(b.cph[0, b.cid] + b.cfq[0, b.cid] * s[i]) * (0.4 + 2.2 * dry)
        wan = (np.sin(b.ph[0] + b.fq[0] * s[i]) + 0.5 * np.sin(b.ph[1] + b.fq[1] * s[i]))
        across = across + wan * (0.5 + 1.8 * dry) * (0.4 + 0.6 * np.abs(b.u)) + rng.normal(0, 0.3, b.n)
        along = along + rng.normal(0, 0.3, b.n)
        bx = px[i] + nx[i] * across + tx[i] * along
        by = py[i] + ny[i] * across + ty[i] * along
        ix = np.clip(np.round(bx).astype(int), 0, W - 1)
        iy = np.clip(np.round(by).astype(int), 0, H - 1)
        touching = reach > 0
        # 碰到多深：饱的笔连纸纹凹处都涂满；干笔、快笔只碰得到凸起的顶
        # 鬃越干，只有越高的纸纹才刮得下颜料（load→0 时一点都刮不下）
        sat1 = np.minimum(sat, 1)
        # 刮干的笔：鬃表面糊着膏，碰到哪儿都刮得下，直到表面刮光
        avail = np.maximum(np.minimum(1, sat1 / 0.6) ** 0.7,
                           b.paste * np.minimum(1, b.Wt / (0.5 * b.tipfrac * b.Wbase)) ** 0.6)
        depth = reach * 1.1 * avail / (1 + 0.4 * max(v - 1, 0))
        cw = wet[iy, ix]
        # 鬃压弯了是贴着纸滑的，小凹坑会跨过去：已经贴上的鬃要掉进更深的坑才离开纸
        #（进门的门槛高、出门的门槛低），所以干笔是一条条的，不是一粒粒的
        thr = (1 - depth) - np.where(prev > 0.5, bridge * reach, 0) + (0.03 + 0.22 * streak * dry) * b.hn
        hsel = hu[iy, ix] if not streak else (1 - streak) * hu_old[iy, ix] + streak * (0.7 * U[:, i] + 0.3 * hf[iy, ix])
        contact = np.clip((hsel - thr) / 0.16 + 0.35, 0, 1) ** 1.5   # 越高的纸纹刮下的越多
        prev = contact.copy()
        # 够饱的鬃是一滴颜料压在纸上，凹处也涂满；踩在这一笔刚走过的湿痕上也一样
        film = np.clip((sat1 - 0.12) / 0.25, 0, 1) * (1 - b.paste)
        liquid = np.maximum(film, (cw > 0.06).astype(float))
        contact = np.maximum(contact, liquid * np.minimum(1, reach * 3)) * touching
        # 放出去多少：鬃里的颜料"流得动"时靠流（跟停留时间有关，快笔放得少），
        # 流不动了只能靠纸纹刮（跟碰到多少、鬃里还剩多少有关：越干攥得越紧）
        wetf = np.clip((sat1 - 0.1) / 0.6, 0, 1); wetf = wetf * wetf * (3 - 2 * wetf)
        dW = contact * b.k * (a_wet * b.Wbase * wetf * (0.3 + p) / v
                              + a_abr * b.paste * b.Wt * p * (1 - wetf) ** 2)
        # 干着刮的时候，一颗纸纹顶上挂得住的颜料有限（约等于同一笔颜料平涂一层那么厚），
        # 挂满了，后面的鬃再蹭过去也刮不下更多——所以飞白不会比同一管颜料的平涂更黑，只是更碎
        dW = np.minimum(dW, 0.6 * b.Wt)
        dryb = film <= 0.5
        if np.any(dryb):
            # 同一步里挤在一颗纸纹上的几根鬃一起算：想刮下的加起来，超出还挂得住的，按比例少刮
            q = np.where(dryb, dW * Pt.sum(1) / (b.Wt + 1e-12) * gain, 0)
            _, inv = np.unique(iy * W + ix, return_inverse=True)
            Q = np.bincount(inv, weights=q)[inv] * 0.5          # 落点那一格大约吃到一半（其余匀到旁边）
            cap = Pt.sum(1) / (b.Wt * wgain + 1e-9) * hcap
            room = np.clip(cap - Dst[iy, ix], 0, None)
            dW = np.where(dryb, dW * np.minimum(1, room / (Q + 1e-12)), dW)
        phi = dW / (b.Wt + 1e-12)
        dP = Pt * phi[:, None]
        Pt -= dP; b.Wt = b.Wt - dW
        # 笔肚子往笔尖补（每根快慢不同）
        need = np.clip(b.tipfrac * b.Wbase - b.Wt, 0, None)
        mv = np.minimum(need * b.rr * (0.1 + 0.9 * wetf), b.Wb)   # 肚子里流得动才补得上
        fr = mv / (b.Wb + 1e-12)
        Pt += Pb * fr[:, None]; Pb -= Pb * fr[:, None]
        b.Wt = b.Wt + mv; b.Wb = b.Wb - mv
        # 饱笔的鬃是连通的：相邻的鬃之间颜料水会匀开（颜料不跟着匀，所以还留着一丝丝的纹）
        #（圆头一类的笔肚子是靠这个往笔尖送的，送过去的是颜料水，颜料要跟着走，不然笔尖越画越淡）
        if i % 4 == 0:
            o = b.order
            share = 0.5 * wetf[o]
            carry = b.kind in ('round', 'rigger', 'mop')
            for arr, Pa in ((b.Wt, Pt), (b.Wb, Pb)):
                a = arr[o]
                arr[o] = a + share * (gaussian_filter1d(a, 4, mode='nearest') - a)
                if carry:
                    pa = Pa[o]
                    Pa[o] = pa + share[:, None] * (gaussian_filter1d(pa, 4, axis=0, mode='nearest') - pa)
        # 落到纸上：笔里的一点点 → 纸上一片（gain 换算）
        dWg = dW * gain * wgain; dPg = dP * gain
        # 湿痕不只在鬃尖上：往两边各匀一点，鬃与鬃之间的缝就连上了
        wv = dWg * 0.9 * film
        _splat(wet, by, bx, wv * (1 - 2 * b.spread))
        _splat(wet, by + ny[i], bx + nx[i], wv * b.spread)
        _splat(wet, by - ny[i], bx - nx[i], wv * b.spread)
        on_wet = ((cw + dWg * film) > 0.08) & (film > 0.5)
        # 饱的鬃压过的地方，也有一点颜料当场压进纸纹（PRESS）；干的、半干的鬃落下的全算直接压进去的
        press = np.where(on_wet[:, None], PRESS * contact[:, None], 1.0)
        _splat(S_a, by, bx, (dPg * (1 - press)).T)
        # 直接落下的颜料也不是一根头发丝那么细：一颗纸纹顶有好几根鬃宽，往两边各匀一点
        dd = (dPg * press).T
        for off, wt in ((0, 1 - 2 * b.spread), (1, b.spread), (-1, b.spread)):
            _splat(D_a, by + off * ny[i], bx + off * nx[i], dd * wt)
            _splat(Dst, by + off * ny[i], bx + off * nx[i], dd.sum(0) * wt)
        # 挂满了就挂不上：这一步落下去超出上限的那点，退回给刚才放它的干鬃（笔里还留着）
        if np.any(dryb):
            qd = dd[:, dryb]                               # (k, 干鬃)
            qt = qd.sum()
            if qt > 0:
                capv = np.average(Pt[dryb].sum(1) / (b.Wt[dryb] * wgain + 1e-9), weights=qd.sum(0) + 1e-12) * hcap
                y0 = max(iy.min() - 3, 0); y1 = min(iy.max() + 4, H)
                x0 = max(ix.min() - 3, 0); x1 = min(ix.max() + 4, W)
                loc = Dst[y0:y1, x0:x1]
                ex = np.maximum(loc - capv, 0)
                E = ex.sum()
                if E > 0:
                    comp = qd.sum(1) / qt                  # 这一步干刮下来的颜料配比
                    D_a[:, y0:y1, x0:x1] -= comp[:, None, None] * ex[None]
                    loc -= ex
                    back = min(E / qt, 1.0)                # 退回的比例
                    Pt[dryb] += (qd * back).T / gain
                    b.Wt[dryb] += dW[dryb] * back
        np.clip(wet, 0, 2.0, out=wet)
    b.Pt[:, act] = Pt; b.Pb[:, act] = Pb
    out = np.zeros((b.Pt.shape[1], H, W))
    out[act] = D_a + S_a
    return out
