# 许愿石 · 卡梅尔（10.2 晚）
# 这张在说什么：海刚退下去，湿沙上两块石头，各有一圈白。大的那块是一道细的、一整圈没断；小的那块是宽的，对着太阳微微透光。
# 光：太阳落在海那边（画框左上外面），很低，暖金。天光从头顶来，冷的淡紫——影子蓝紫不黑，朝右下拖得很长，离石头越远越虚。
# 明暗：石头最深；薄水膜映着天，最亮的一片在左上靠海那头；沙是中间调。
# 视线：先落到大石头那道白，顺着它到小石头的宽带，再沿影子出去。
# 质感：沙是细颗粒（每粒一个像素），水膜是光的几乎没颗粒，泡沫是镂空的花边，石头是软斑+几粒亮的晶，白带是一笔干笔。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.abspath('../../kbrush')                 # 仓库里的引擎；从仓库根目录跑：python examples/<名字>/<名字>.py
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(7)
from shapes import Shape
from blocks import lasso, line, rim, soft, look, smooth, spray, paper
from layers import Stack, fill, stroke, dip, dip_tip, HARD, SOFT, LOST
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt

H, W = 1200, 900
yy, xx = np.mgrid[0:H, 0:W].astype(float)
rng = np.random.default_rng(11)
def nz(sig):
    n = gaussian_filter(rng.standard_normal((H, W)), sig); return n / n.std()
def mix(a, b, t):
    t = np.asarray(t, float)[..., None] if np.ndim(t) else t
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t
def grains(c, amount, clump=0.6, dark=(.80, .76, .80), light=.18, p=.45):
    """细颗粒：两种墨按疏密点，amount 是这一处颗粒有多显"""
    n = gaussian_filter(rng.standard_normal((H, W)), clump); n /= n.std()
    pick = n < (np.asarray(p) * 2 - 1) * 1.0
    lo = c * np.array(dark); hi = c + (1 - c) * light
    g = np.where(pick[..., None], lo, hi)
    a = np.asarray(amount, float)[..., None] if np.ndim(amount) else amount
    return np.clip(c * (1 - a) + g * a, 0, 1)

SUN = np.array([.75, .62])                       # 光往右下走（从左上来）
st = Stack(K.Paper(H, W, seed=3), ground='#b39890')

# ---------- 1-2 沙和水：都是平的台阶（nerolette：俯视看不到横波纹；沙别模拟真沙）
SHORE = smooth([(-20, 650), (90, 610), (170, 572), (250, 588), (330, 558), (400, 500), (470, 470), (520, 500),
                     (560, 558), (620, 574), (690, 558), (730, 470), (760, 402), (820, 378), (850, 300), (920, 250)], 4, closed=False)
def resample(pts, step=2.0):
    p = np.array(pts); seg = np.hypot(*np.diff(p, axis=0).T); s_ = np.r_[0, np.cumsum(seg)]
    t = np.arange(0, s_[-1], step); return np.c_[np.interp(t, s_, p[:, 0]), np.interp(t, s_, p[:, 1])], t
def normals(p):
    d = np.gradient(p, axis=0); n = np.c_[d[:, 1], -d[:, 0]]; n /= np.linalg.norm(n, axis=1)[:, None] + 1e-9
    return n * np.sign(n @ np.array([.45, .9]))[:, None]           # 朝沙那边
def scallop(p, t, arcs, amp=.16):
    """沿着一条线，一段一段鼓出去的弧：arcs 是每段弧长（手挑的，长短不一），鼓的高度跟弧长成比例"""
    n = normals(p); edges_ = np.cumsum([0] + list(arcs) * 20); out = p.copy()
    k = np.searchsorted(edges_, t, side='right') - 1
    L = np.diff(edges_)[k]; u = (t - edges_[k]) / L
    return out + n * (amp * L * np.sin(np.pi * u))[:, None], n
shore_p, shore_t = resample(SHORE)
ARCS = [72, 46, 108, 58, 88, 50, 124, 64, 92, 40, 104, 70, 56, 118, 80, 48]
edge_p, edge_n = scallop(shore_p, shore_t, ARCS)
film = lasso(H, W, [(-120, 700)] + [tuple(v) for v in edge_p] + [(1020, 230), (1020, -120), (-120, -120)], jag=1.0, scale=6, seed=3)
ALL = Shape(np.ones((H, W), bool))
wob = lambda amp, sig: amp * nz(sig)
dsand = edt(~film.mask); dwater = edt(film.mask)
fill(st.layer('沙'), ALL, (.77, .66, .60), edge=HARD)                           # 干一点的沙
for k, (dist, col, sp) in enumerate(((190, (.69, .59, .57), 14), (55, (.61, .52, .53), 9))):   # 越贴水越湿越暗
    reg = ~film.mask & (dsand + wob(14, 25) < dist)
    fill(st.layer(f'湿{k}', mask=spray(reg, sp, rng, .95)), ALL, col, edge=HARD)
def shift(m, dy, dx):
    o = np.zeros_like(m); dy, dx = int(dy), int(dx)
    o[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = m[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return o
# ---------- 2-3 水边（三种小样，MODE 选）
#   rings：水是平的几阶，水边和往外几圈是弯弯扭扭的白泡沫线，一圈套一圈（nerolette 的印象）
#   paper：水留成纸，什么都不铺，只有那几圈线，水是从"沙停在哪"读出来的
#   spray：水是平的几阶，每一圈不是线，是一串喷点
MODE = os.environ.get('MODE', 'rings')
fill(st.layer('水'), film, (.70, .64, .66), edge=HARD)
for k, (dist, col, sp) in enumerate(((40, (.64, .60, .67), 10), (120, (.58, .56, .68), 12), (240, (.52, .52, .68), 14))):
    reg = film.mask & (dwater + wob(10, 20) > dist)                   # 深浅是透过水看见的沙底：糙的，边喷开（nerolette）
    fill(st.layer(f'深{k}', mask=spray(reg, sp, rng, .95)), ALL, col, edge=HARD)
# 水面上的泡沫线：两头收尖、中间鼓，有的地方分成两股再合上，围出扁扁的眼；边干净不喷（光滑的水面）。
# 全画最亮最脆的白留给石头那两圈：水线压透、压灰；贴着水边最新的那道最亮，越旧越淡（nerolette 和 Claude，10.2 晚）
RINGS = ((0, 2.4, (.94, .92, .93), .95), (40, 3, (.91, .89, .94), .8), (95, 3.4, (.88, .87, .94), .68),
         (170, 4.6, (.86, .85, .93), .58), (285, 7, (.88, .87, .94), .62))
# 只有最新那道贴着水边走；旧的几圈只跟水边的大势，不抄每个小疙瘩（nerolette 圈出来的那两条好，是因为离边远、是大圆弧）
def calm_dist(sig):                                                  # 水边抹圆了再往外推：越往外抹得越圆，拐角不再是 V
    return edt(gaussian_filter(film.mask.astype(float), sig) > .5)
def strands(D, w, k):
    jit = wob(9 + D * .05, 45)                                        # 路径：大的圆弧，不碎抖
    d = (dwater if k == 0 else calm_dist(30 + .25 * D)) + jit
    tp = nz(30 + 3 * k)
    thick = w * np.clip(tp * .75 + .55, 0, 1.4)                       # 粗细慢慢变，变到 0 就是收尖的断口
    main = np.abs(d - D) < thick
    p = np.clip((nz(38 + 2 * k) - .25) * 1.6, 0, 1)                    # 第二股：离开主线、再回来
    sep = (9 + .06 * D) * p
    second = (np.abs(d - (D + sep + thick)) < w * .65 * np.sqrt(p)) & (p > .03) & (thick > .75 * w)
    return film.mask & (main | second)
for k, (D_, w_, col, op) in enumerate(RINGS):
    m = strands(D_, w_, k)                                            # 照算一遍，好让后面那两条跟 nerolette 圈的时候一模一样
    if k in (0, 1, 2, 3, 4):
        continue                                                      # 这几圈都不画了；照算只为了随机数顺序不变                                                      # 只留水边那道和 nerolette 圈过的最外面那圈；其余换成手画的（下面"水纹"）
    if k:                                                             # 舌头那里拐成了一个对勾，擦掉
        m &= np.hypot((xx - 622) / 48, (yy - 400) / 85) > 1
    fill(st.layer(f'圈{k}', opacity=op, mask=soft(Shape(m), feather=1.2)), ALL, col, edge=HARD)
# 浪：前沿一下亮的、脆的；后面朝海那边拖细丝和白点，越往后越碎越淡（参考 Patricia Tokarz）
from PIL import Image as _I, ImageDraw as _D
def front_marks(p, n, t, seed, n_streak, reach, specks):
    r_ = np.random.default_rng(seed); SS = 2
    im = _I.new('L', (W * SS, H * SS), 0); dr = _D.Draw(im)
    wgt = gaussian_filter(r_.random(len(p)), 40); wgt = np.clip((wgt - wgt.mean()) / wgt.std() * .6 + .7, .08, None)
    pick = lambda: r_.choice(len(p), p=wgt / wgt.sum())               # 沿着浪有的地方泡沫厚、有的地方几乎没有
    for _ in range(n_streak):                                       # 细丝：在前沿后面，顺着浪拉长，越往后越短越淡
        i = pick(); d = 2 + r_.exponential(reach * .35)
        base = p[i] - n[i] * d
        tang = np.array([n[i][1], -n[i][0]]) * (1 if r_.random() < .5 else -1)
        ang = np.arctan2(tang[1], tang[0]) + r_.normal(0, .22)
        L = (8 + 38 * r_.random() ** 1.3) * max(.25, 1 - d / (reach * 1.4)); w = .6 + 1.2 * r_.random() * max(.3, 1 - d / reach)
        tip = base + L * np.array([np.cos(ang), np.sin(ang)]); mid = (base + tip) / 2 - n[i] * r_.normal(0, 2.5)
        side = np.array([-np.sin(ang), np.cos(ang)]) * w / 2
        dr.polygon([tuple(base * SS), tuple((mid + side) * SS), tuple(tip * SS), tuple((mid - side) * SS)], fill=int(130 + 125 * r_.random() * max(.3, 1 - d / reach)))
    for _ in range(specks):                                         # 白点：越往后越稀越小
        i = pick(); d = 3 + r_.exponential(reach * .45); c = p[i] - n[i] * d + r_.normal(0, 3, 2)
        rad = max(.5, 1.9 - d / reach * 1.3) * (.6 + .6 * r_.random())
        dr.ellipse([(c[0] - rad) * SS, (c[1] - rad) * SS, (c[0] + rad) * SS, (c[1] + rad) * SS], fill=int(140 + 115 * r_.random()))
    return np.asarray(im.resize((W, H), _I.LANCZOS), float) / 255
def front_line(p, n, w0, seed):
    r_ = np.random.default_rng(seed); SS = 2
    im = _I.new('L', (W * SS, H * SS), 0); dr = _D.Draw(im)
    wv = w0 * (.55 + .45 * np.abs(np.sin(np.cumsum(r_.normal(0, .06, len(p))))))
    for i in range(len(p) - 1):
        dr.line([tuple(p[i] * SS), tuple(p[i + 1] * SS)], fill=255, width=max(1, int(wv[i] * SS)))
    return np.asarray(im.resize((W, H), _I.LANCZOS), float) / 255
from scipy.spatial import cKDTree
# 泡沫带和网（拆自那张平涂参考）：不是一根根线，是一整片浅色上挖洞。洞是圆圆的、大小不一的块挤在一起，
# 洞之间剩下的窄条就是网，三条交汇处自然鼓一点。尺寸按石头来：浪头的带子拇指宽，带里的洞指甲盖大，往外的网格有石头五分之一大。
from scipy.ndimage import gaussian_filter1d
# 第二层是第一层的回声：把岸线整条往海里平移（不沿法线推，推到凹角会折尖），再抹掉一半起伏；第一层在中间往海里收的地方，它也收（nerolette 10.2 画的那道黄线）
lift = np.interp(shore_p[:, 0], [0, 450, 900], [170, 215, 265])[:, None] * np.array([.1, -1])   # 左边离岸近、右边远：不平行
far_p = shore_p + lift
far_p = .25 * far_p + .75 * np.c_[gaussian_filter1d(far_p[:, 0], 45, mode='nearest'), gaussian_filter1d(far_p[:, 1], 45, mode='nearest')]   # 只留一点回声
far_p, far_t = resample([tuple(v) for v in far_p])
far_p, far_n = scallop(far_p, far_t, [130, 62, 96, 150, 54, 110, 78, 140], amp=.12)
far_line = np.zeros((H, W), bool)
fi = far_p.astype(int); ok_ = (fi[:, 0] >= 0) & (fi[:, 0] < W) & (fi[:, 1] >= 0) & (fi[:, 1] < H)
far_line[fi[ok_, 1], fi[ok_, 0]] = True
dfar = edt(~far_line)
_ft = cKDTree(far_p); _yy, _xx = np.nonzero(film.mask); _, _ix = _ft.query(np.c_[_xx, _yy])
seaward = np.zeros((H, W), bool)
seaward[_yy, _xx] = ((np.c_[_xx, _yy] - far_p[_ix]) * far_n[_ix]).sum(1) < 0   # 外浪那条线朝海的一侧
# 洞的大小：贴着两道浪头小，离浪头越远越大
dwave = np.minimum(dwater, np.where(seaward, dfar, 1e9))
pr = np.random.default_rng(41)
cand = pr.random((30000, 2)) * [W, H]; ci = cand.astype(int)
dd_c = dwave[ci[:, 1], ci[:, 0]]
size = 24 + .5 * np.clip(dd_c, 0, 220)
keep = film.mask[ci[:, 1], ci[:, 0]] & (pr.random(len(cand)) < (24 / size) ** 2 * .9)
tree = cKDTree(cand[keep])
fy, fx = np.nonzero(film.mask)
dd, _ = tree.query(np.c_[fx, fy], k=2)
wall = np.full((H, W), 99.0); wall[fy, fx] = dd[:, 1] - dd[:, 0]
wall = gaussian_filter(wall, 1.6)                                   # 洞的角磨圆，交汇处鼓
def band_mask(dist, width, seed_sig, solid_frac=.28, hole_a=1.6, hole_b=4.0):
    bw = width * np.clip(1 + .35 * nz(seed_sig), .35, 1.6)
    inner = dist < bw
    solid = dist < bw * solid_frac                                   # 靠岸那边实的，不挖
    holes = wall > (hole_a + hole_b * np.clip(1 - dist / bw, 0, 1))  # 往海那边洞越挖越大
    return film.mask & inner & (solid | ~holes)
band1 = band_mask(dwater, 62, 40)                # 第一层：v32 原样
_d2 = np.where(seaward, dfar, 1e9)
_d2 = np.where(_d2 > 22, 22 + (_d2 - 22) / 2.0, _d2)                 # 前沿 22px 以内原样；后面的斑驳往海里拉长一倍（nerolette，v39）
band2 = band_mask(_d2, 74, 50, solid_frac=.15, hole_a=1.0, hole_b=3.0)   # 第二层：形是 v36 那个（nerolette 打勾的弯），尾巴拉长、更稀，左边不断
netw = 1.0 + .7 * np.clip(1 - dwave / 260, 0, 1)
net = film.mask & (wall < netw) & ~band1 & ~band2
fade = np.clip(1 - (dwave - 20) / 230, 0, 1) ** 1.2
fill(st.layer('网', opacity=.55, mask=soft(Shape(net), feather=.8) * fade), ALL, (.80, .79, .92), edge=HARD)
fill(st.layer('外浪带', opacity=.62, mask=soft(Shape(band2), feather=.8)), ALL, (.86, .85, .93), edge=HARD)
fill(st.layer('浪带', opacity=.85, mask=soft(Shape(band1), feather=.8)), ALL, (.90, .89, .95), edge=HARD)
fill(st.layer('浪', mask=front_line(edge_p - edge_n * 1.2, edge_n, 2.8, 8) * film.mask), ALL, (.95, .93, .94), edge=HARD)
fill(st.layer('水边', mode='multiply'), Shape((dsand < 6) & ~film.mask), (.82, .76, .80), edge=HARD)

# ---------- 3.5 刚退下去的痕迹：贴着水边的沙更湿更暗、带一点天光；回流绕过石头留下的细沙脊；几处冒过气泡的小眼
def bubbles(cx, cy, n, spread, seed):
    r_ = np.random.default_rng(seed); holes = np.zeros((H, W), bool); rims = np.zeros((H, W), bool)
    for _ in range(n):
        x, y = cx + spread * r_.standard_normal(), cy + spread * .7 * r_.standard_normal(); rr = 1.2 + 1.6 * r_.random()
        d = np.hypot(xx - x, yy - y); holes |= d < rr; rims |= (d >= rr) & (d < rr + 1.6) & (yy > y)
    return Shape(holes), Shape(rims)
for (cx, cy, n, sp, sd) in ((150, 760, 9, 40, 1), (110, 1050, 6, 30, 2), (640, 1080, 5, 25, 3), (760, 620, 4, 18, 4)):
    h_, r_ = bubbles(cx, cy, n, sp, sd)
    fill(st.layer(f'眼{sd}', mode='multiply'), h_, (.55, .48, .52), edge=HARD)
    fill(st.layer(f'眼沿{sd}'), r_, (.86, .78, .74), edge=HARD, opacity=.6)

# ---------- 4 两块石头（手勾）
s1 = lasso(H, W, smooth([(380, 705), (398, 645), (455, 603), (540, 586), (625, 592), (692, 624), (712, 676), (690, 742), (628, 792), (540, 812), (452, 800), (396, 762)]), jag=1.2, scale=5, seed=8)
s2 = lasso(H, W, smooth([(242, 885), (262, 832), (320, 806), (392, 812), (440, 852), (446, 912), (410, 962), (340, 978), (276, 958), (246, 922)]), jag=1.2, scale=5, seed=9)

# 影：朝右下拖，靠近石头硬，远了虚
def sel_r(sel): return np.sqrt(sel.mask.sum() / np.pi)
def cast(sel, length, steps=40, taper=True):
    m = np.zeros((H, W), bool); core = np.zeros((H, W), bool)
    din = edt(sel.mask); emax = din.max()
    for k in range(steps + 1):
        d = SUN * length * k / steps
        # 鹅卵石是扁的一团：下半截整块往外扫，上半截按椭球一圈圈收，尾巴是圆的、跟石头差不多宽
        # （v38 以前是越拖越瘦，nerolette："像钟乳石而不是鹅卵石"——那是尖顶的东西的影子）
        s_ = max(0., (k / steps - .5) / .5)
        shrink = din > emax * (1 - np.sqrt(max(0., 1 - s_ * s_)))
        sh = shift(sel.mask & shrink if (k and taper) else sel.mask, d[1], d[0])
        m |= sh
        if k < steps * .45: core |= sh
    return Shape(m & ~sel.mask), Shape(core & ~sel.mask)
SH = (.66, .64, .84)
for sel, ln, nm in ((s1, 295, '影1'), (s2, 237, '影2')):              # 圆尾巴的影子比尖的重，短两成，大的那条不碰画框
    far, core = cast(sel, ln)
    away = edt(~sel.mask)                                            # 离石头多远
    farm = (far.mask & (edt(far.mask) + wob(5, 18) > 6)) | (far.mask & (away < 80))   # 外沿往里收一点让它喷开；贴着石头两侧那两条细边不收，不然影子从石头后面出来先窄一截、尾巴鼓成一团（v41，nerolette 看出来的）
    fill(st.layer(nm, mode='multiply', opacity=.8, mask=spray(farm, 3 + .06 * away, rng, .95)), ALL, SH)   # 一个太阳一个影子一个颜色：远了只是边虚
    lay = st.layer(nm + '缝', mode='multiply')               # 贴地那道缝：最深最锐，在背光的那半圈
    fill(lay, Shape((edt(~sel.mask) < 4) & ~sel.mask & (xx * SUN[0] + yy * SUN[1] > 0) & cast(sel, 10, 5, taper=False)[0].mask), (.45, .40, .55), edge=HARD)

# 石头本身：台阶，不是渐变（nerolette 看木内那只盆看出来的）。
#   底：整块先铺迎光的暖色；
#   一阶：冷的中间色，盖过背光的一大半；二阶：更深；三阶：换一个新颜色（最深），只压在交界后面那条带上。
#   每一阶里面密度不变，过渡全在它的边上——往光那边喷开一圈颗粒。
def u_of(sel):
    ys, xs = np.nonzero(sel.mask); cy, cx = ys.mean(), xs.mean(); r = np.sqrt(sel.mask.sum() / np.pi)
    return ((xx - cx) * SUN[0] + (yy - cy) * SUN[1]) / r       # -1 迎光 … +1 背光
def ramp(v, a, b):
    t = np.clip((v - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
def stone(sel, i, base_c, steps):
    u = u_of(sel)
    wob = 4 * nz(12) / 100                                       # 台阶的边别是尺子拉的
    c = np.array(base_c) * (1 + .03 * nz(6))[..., None]
    spark = (nz(.6) > 3.1) & (u < .2)                              # 几粒亮的晶，只在迎光那边看得见
    c = np.where(spark[..., None], np.minimum(c + .3, 1), c)
    fill(st.layer(f'石{i}底'), sel, np.clip(c, 0, 1), edge=HARD)
    for k, (lo, hi, col, sp) in enumerate(steps):
        reg = (u + wob > lo) & (u + wob < hi)
        tex = nz(.8)                                                 # 阶里不是一样实：深浅抖一点，留几个针眼
        fill(st.layer(f'石{i}阶{k}', mask=spray(reg, sp, rng, np.clip(.92 + .12 * tex, 0, 1) * (tex > -2.2), dens=.75) * sel.mask), sel, col, edge=HARD)
    refl = np.clip(1 - edt(sel.mask) / 12, 0, 1) * ramp(u, .55, .95)
    fill(st.layer(f'石{i}反', mode='screen', mask=refl * .4), sel, (.42, .42, .58), edge=HARD)
    return u
u1 = stone(s1, 1, [.47, .41, .40],                                  # 黑石头迎光那面也是暗的（v38 是 .60 跟沙一样亮，白线在左半边不跳）
             [(-.05, 9, (.38, .36, .44), 14), (.30, 9, (.24, .24, .35), 10), (.42, .78, (.12, .12, .24), 7)])
u2 = stone(s2, 2, [.82, .56, .42], [(-.05, 9, (.62, .40, .44), 12), (.32, 9, (.44, .27, .38), 9), (.45, .80, (.24, .14, .30), 6)])

# ---------- 5 白：大的一道细的，绕过去；小的一道宽的，背光那半透出暖。
#   也是平铺的阶，不用刷子：这套画法自己就长得出质感，刷子有点赘余（nerolette）。v39 以前是两笔干笔，一层层的带子像海边常捡到的；一整条不断的更稀罕
def band(pts, w, seed, shift_=0.):
    """一条带子的形：中线抹顺，两边按宽度偏出去，围成一块；shift_ 把带子往一侧挪（取它的一条边用）"""
    c, _ = resample(smooth(pts, 3, closed=False), 1.5)
    d = np.gradient(c, axis=0); n = np.c_[d[:, 1], -d[:, 0]]; n /= np.linalg.norm(n, axis=1)[:, None]
    t = np.linspace(0, 1, len(c)); ww = np.array([w(v) for v in t])[:, None]
    lo = c + n * (shift_ - .5) * ww; hi = c + n * (shift_ + .5) * ww
    return lasso(H, W, [tuple(v) for v in lo] + [tuple(v) for v in hi[::-1]], jag=.9, scale=6, seed=seed)
#   嵌在石头里的样子：带子实实地盖上，边往外喷一圈细颗粒，只留在石头里（给的是种子，自己的随机数，不动 rng 的顺序）
# 大的：一道细的，中间宽两头细（绕到背面去了）
v1 = band([(566, 578), (588, 640), (586, 712), (560, 776), (514, 822)], lambda t: 5 + 9 * np.sin(np.pi * t), 21) & s1
fill(st.layer('白1', mask=spray(v1, 2.5, 31, .95) * s1.mask), ALL, (.96, .93, .86), edge=HARD)
# 小的：宽的一条，两阶——奶白一整条，朝背光那边贴一条暖的（透光的那层）
pts2 = [(262, 822), (296, 868), (345, 912), (400, 940), (446, 948)]
v2 = band(pts2, lambda t: 30 + 14 * np.sin(np.pi * t), 22) & s2
fill(st.layer('白2', mask=spray(v2, 4, 32, .95) * s2.mask), ALL, (.95, .90, .84), edge=HARD)
_w2 = band(pts2, lambda t: 30 + 14 * np.sin(np.pi * t), 23, shift_=.36) & v2
fill(st.layer('白2暖', mask=spray(_w2, 3, 33, .95) * v2.mask), ALL, (.97, .84, .68), edge=HARD)
# 白带上也有明暗：背光那半罩一层冷的，可小的那块透光，背光那半反而暖
fill(st.layer('白上冷', mode='multiply', mask=soft(Shape(s1.mask & (u1 > .15)), feather=18)), s1, (.80, .80, .92))
fill(st.layer('透', mode='screen', opacity=.35, mask=soft(Shape(s2.mask & (u2 > .2)), feather=20)), s2, (.85, .55, .30))

# ---------- 6 湿：顶上映一片天（软、冷、宽），朝太阳那侧边上一粒暖的反光（小、锐）；边上的金线只在最朝光那一小段
for i, sel in enumerate((s1, s2)):
    ys, xs = np.nonzero(sel.mask); cy, cx = ys.mean(), xs.mean(); r = np.sqrt(sel.mask.sum() / np.pi)
    d = np.hypot((xx - (cx - .12 * r)) / (r * .75), (yy - (cy - .2 * r)) / (r * .45))
    fill(st.layer(f'天{i}', mode='screen', opacity=.35, mask=spray((d + .15 * nz(10) < .62) & sel.mask, 6, rng, .95) * sel.mask), sel, (.55, .52, .66), edge=HARD)
    u = u_of(sel)
    lip = sel.mask & (edt(sel.mask) < 1.5 + 1.5 * np.clip(-u - .6, 0, 1) * (nz(6) > -.3)) & (u < -.7)
    fill(st.layer(f'金边{i}'), Shape(lip), (1, .88, .66), edge=HARD, opacity=.9)
    g = np.hypot((xx - (cx - .55 * r)) / 9, (yy - (cy - .5 * r)) / 5)
    fill(st.layer(f'闪{i}', mode='screen', mask=np.clip(1.2 - g, 0, 1) * sel.mask), sel, (1, .85, .6), edge=HARD)

# （回流的细线去掉了：在这张里读成了石头在飞的速度线）
# ---------- 8 纸：整张同一张纹，钉在画布上
fill(st.layer('纸', mode='multiply'), ALL, paper(H, W, 77), edge=HARD)
look(st, 'render.png')                              # 成品是同一文件夹的 wishstone.png（v41）
