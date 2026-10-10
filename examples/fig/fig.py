# 无花果 · 10.2
# 从哪儿看：正上方往下，桌上铺着灰蓝的亚麻布，画框是布上 40cm 宽的一块（30px ≈ 1cm）。
# 这张在说什么：无花果没有花——花开在里面，切开才看得见。整颗的那只闭着、蒙着一层灰霜；切开的半只朝上，是一朵红的。
# 光：早上，屋里暗。太阳从画框左边外面一道窗帘缝里进来，低（约 37°）、小、暖白偏金，
#     在布上落成一条斜的光带，往右下走，到右边窗楣的影那里散成颗粒（半影宽到两指），越远边越虚。
#     暗处只有天光：冷的灰蓝，离窗越远越暗。影子只在光带里才有；影子就是"光没到的地方"，所以它的颜色就是那一处暗处的布色，
#     一出光带就跟暗处连成一片。切面是平的、湿的：从正上方看不到太阳的镜面反光，只有果肉上朝光鼓起的小粒闪几下。
# 明暗：暗处三阶（靠窗亮、远处暗，最暗那阶换成靛紫）；光带一阶；
#     最亮是切面奶白的一圈，最深是整颗那只背窗那侧、贴地的缝。
# 闭着的在暗处，开着的在光里：光只找到了切开的那一只。
# 视线：先落在切开那半只的红（全画唯一鲜的颜色，边最锐），到光带外面暗处闭着的那只，
#     再顺着空的光带走到它在右边软掉的尽头，抬到右上暗处那只白盘子。刀从下边伸进来，刀尖进了光，刃上一线亮。
# 质感：每一阶一个颜色、阶里一样密，过渡在阶边上喷成颗粒；果肉里的过渡是顺着花丝放射的细丝。
#     只有湿果肉上的几粒闪光、刀刃那一线、盘沿那一线是干净的边。整张一张纸纹，最后罩上去。不用刷子。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.abspath('../../kbrush')                 # 仓库里的引擎；从仓库根目录跑：python examples/<名字>/<名字>.py
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(5)
from shapes import Shape
from blocks import lasso, line, soft, look, smooth, spray, paper
from layers import Stack, fill, HARD
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt
from PIL import Image as _I, ImageDraw as _D

ROUGH = os.environ.get('ROUGH') == '1'
OUT = os.environ.get('OUT', 'render_rough.png' if ROUGH else 'render.png')   # 成品是同一文件夹的 fig.png
H, W = 900, 1200
yy, xx = np.mgrid[0:H, 0:W].astype(float)
ALL = Shape(np.ones((H, W), bool))

def nz(sig, seed):
    r = np.random.default_rng(seed)
    n = gaussian_filter(r.standard_normal((H, W)), sig); return n / n.std()
if ROUGH:                                                        # 草稿：阶不喷，只看大块
    spray = lambda region, sp, seed, solid=.96, **_: region.astype(float) * solid
def ink(draw_fn, seed, SS=3):
    """在 3 倍大的灰度图上画小东西（细丝、籽、闪光），缩回来当蒙版"""
    im = _I.new('L', (W * SS, H * SS), 0); draw_fn(_D.Draw(im), np.random.default_rng(seed), SS)
    return np.asarray(im.resize((W, H), _I.LANCZOS), float) / 255

# ---------- 光的几何（光往 D 走）
D = np.array([.88, .47]); D /= np.linalg.norm(D)
NRM = np.array([-D[1], D[0]])                                   # 垂直于光带
C0 = np.array([0., 228.])                                        # 光带中线从左边进来的地方
HW = 145                                                         # 光带半宽
along = (xx - C0[0]) * D[0] + (yy - C0[1]) * D[1]               # 沿光走了多远
across = (xx - C0[0]) * NRM[0] + (yy - C0[1]) * NRM[1]          # 离中线多远（正 = 左下那侧）
curtain = 13 * np.sin(along / 230 + .5) + 4 * np.sin(along / 83 + 2)  # 上沿是窗帘的边，布垂着，微微弯；下沿是窗框，直
END = 930                                                        # 窗楣的影停在这里，跟墙平行（几乎是竖的）
endline = (xx - (C0[0] + D[0] * END)) * 1.0 + (yy - (C0[1] + D[1] * END)) * .16
beam = (across > -HW + curtain) & (across < HW) & (endline < 0)
pen = 3.5 + 26 * np.clip(along / END, 0, 1) ** 1.6              # 两侧的半影：靠窗窄，越远越宽
pen = np.where(endline > 0, 58., pen)                            # 尽头是窗楣，离得最远，最虚：光在这里散成颗粒

# ---------- 东西的形（手勾；u：圆头 -1 → 蒂 +1，v：宽）
def place(pts, cx, cy, L, Wd, ang):
    a = np.radians(ang); c, s = np.cos(a), np.sin(a)
    return [(cx + c * u * L / 2 - s * v * Wd / 2, cy + s * u * L / 2 + c * v * Wd / 2) for u, v in pts]
def at(a_, c_):                                                  # 光带坐标（沿光走多远、离中线多远）→ 画面坐标
    p = C0 + D * a_ + NRM * c_; return float(p[0]), float(p[1])
FIG = [(-1.0, .02), (-.93, .42), (-.72, .78), (-.42, .97), (-.06, .99), (.28, .84), (.52, .58), (.72, .34), (.88, .22), (.99, .15),
       (.99, -.14), (.88, -.22), (.70, -.36), (.45, -.66), (.12, -.91), (-.24, -1.0), (-.58, -.88), (-.84, -.58)]
CUT = [(-1.0, 0.0), (-.94, .44), (-.75, .80), (-.45, .99), (-.10, 1.0), (.22, .88), (.47, .62), (.68, .37), (.86, .23), (.99, .17),
       (.99, -.16), (.86, -.24), (.66, -.40), (.42, -.68), (.12, -.93), (-.22, -1.0), (-.56, -.90), (-.82, -.62)]
WF = (*at(425, -246), 232, 172, -20)                             # 整颗：闭着的那只留在暗处，就在光带上沿外面；圆头朝左，蒂朝右上
CF = (*at(330, 4), 222, 168, -122)                              # 半只：靠窗，整个在光里，偏下沿；蒂朝左上，切面朝上
def L(spec, pts, jag=.6, seed=0, n=3):
    return lasso(H, W, smooth(place(pts, *spec), n), jag=jag, scale=5, seed=seed)
whole = L(WF, FIG, 1.0, 11)
half = L(CF, CUT, .8, 12)
def stem(spec, L0, L1, w, seed):
    cx, cy, Ln, Wd, ang = spec; a = np.radians(ang); d = np.array([np.cos(a), np.sin(a)])
    p0 = np.array([cx, cy]) + d * Ln / 2 * L0; p1 = np.array([cx, cy]) + d * Ln / 2 * L1
    return line(H, W, [tuple(p0), tuple((p0 + p1) / 2 + np.array([-d[1], d[0]]) * 3), tuple(p1)], width=lambda t: w * (1 - .25 * t), seed=seed)
wstem = stem(WF, .9, 1.17, 16, 13)
hstem = stem(CF, .9, 1.14, 14, 14)

# 盘子：右上角，被框切掉
PC, PR = np.array([1112., 86.]), 240.
pd = np.hypot(xx - PC[0], yy - PC[1]) / PR
plate = pd < 1
# 刀：从下边伸进来，刀柄在暗处，刀尖进了光；刃在朝光那侧（左上），背在右下
KN0, KN1 = np.array([70., 960.]), np.array([424., 560.])
KL = np.linalg.norm(KN1 - KN0); kd = (KN1 - KN0) / KL; kn = np.array([-kd[1], kd[0]])
def kpts(prof):
    return [tuple(KN0 + kd * KL * t + kn * o) for t, o in prof]
HANDLE = [(0, -19), (.25, -21), (.47, -19), (.47, 19), (.25, 21), (0, 18)]
BOLSTER = [(.47, -20), (.505, -19), (.505, 18), (.47, 19)]
BLADE = [(.50, -19), (.72, -19), (.86, -15), (.95, -8), (1.0, 3), (.92, 10), (.72, 14), (.50, 15)]
handle = lasso(H, W, smooth(kpts(HANDLE), 2), jag=.6, scale=5, seed=15)
bolster = lasso(H, W, kpts(BOLSTER), jag=0)
blade = lasso(H, W, smooth(kpts(BLADE), 1), jag=0)

# ---------- 影：光带里才有。整颗上半截按椭球收、尾巴圆；半只的顶是平的切面，影是切面原样平移
def shift(m, v):
    o = np.zeros_like(m); dy, dx = int(round(v[1])), int(round(v[0]))
    o[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = m[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return o
def cast(sel, length, steps=48, dome=True):
    m = np.zeros((H, W), bool); din = edt(sel); emax = din.max()
    for k in range(steps + 1):
        s_ = max(0., (k / steps - .5) / .5)
        shrink = (din > emax * (1 - np.sqrt(max(0., 1 - s_ * s_)))) if dome else sel
        m |= shift(sel & shrink if k else sel, D * length * k / steps)
    return m
# 刀身：刀柄厚，把刃根垫起约 1cm，刀尖贴着布——影从刃根离开刀身约 45px，到刀尖收拢
bsh = np.zeros((H, W), bool)
for k in range(24):
    t0, t1 = .50 + .5 * k / 24, .50 + .5 * (k + 1) / 24
    seg = blade.mask & (((xx - KN0[0]) * kd[0] + (yy - KN0[1]) * kd[1]) / KL >= t0) & (((xx - KN0[0]) * kd[0] + (yy - KN0[1]) * kd[1]) / KL < t1 + .01)
    bsh |= shift(seg, D * 40 * (1 - (t0 - .5) / .5))
objects = whole.mask | half.mask | wstem.mask | hstem.mask | handle.mask | bolster.mask | blade.mask
shadow_all = cast(whole.mask, 233) | cast(wstem.mask, 118, dome=False) | cast(half.mask, 111, dome=False) | cast(hstem.mask, 55, dome=False) | bsh   # 高 ÷ tan37°
shadow = shadow_all & beam & ~objects
dobj = edt(~objects)

# ---------- 颜色（先定好的台阶）
SHADE = [(.57, .58, .63), (.50, .51, .58), (.40, .40, .52)]      # 暗处：靠窗 → 远处，最暗那阶换成靛紫
SUN = (.92, .84, .69)

st = Stack(K.Paper(H, W, seed=3), ground=SHADE[0])
# 1 暗处的布：天光从窗那边来，一圈圈往外暗（圆心在画框左边外面的窗上）
WC = np.array([-380., 250.])
wd = np.hypot(xx - WC[0], yy - WC[1]) + 14 * nz(40, 31)
fill(st.layer('暗1', mask=spray(wd > 1000, 24, 41, holes=.012)), ALL, SHADE[1])
fill(st.layer('暗2', mask=spray(wd > 1450, 30, 42, holes=.012)), ALL, SHADE[2])
# 布的折痕：熨过的桌布对折过，一道竖的。暗处几乎看不见；光带里掠射，朝光那侧一线亮、背光那侧一线暗
crease_x = 742 + 14 * np.sin(yy / 170) - .10 * yy
cr_lit, cr_dark = np.abs(xx - crease_x + 2) < 1.3, np.abs(xx - crease_x - 1.5) < 1.8
fill(st.layer('折痕暗处', opacity=.35, mask=soft(Shape(cr_lit & ~beam), feather=2)), ALL, (.66, .67, .72))
# 2 光带：光没到的地方（影）就是暗处的布，所以影是从光里挖掉的洞；洞的边也喷开，贴着东西窄、越远越宽
sp_map = np.where(shadow_all & ~objects, 1.4 + .012 * dobj, pen)
lit = beam & ~shadow_all & ~objects
# 尽头：窗楣离得远，半影宽到两指——实的那阶提前收住，后面一段颗粒从密到稀，光是散掉的，不是停住的
EW = 70
sides = (across > -HW + curtain) & (across < HW)
side_out = np.maximum(np.maximum((-HW + curtain) - across, across - HW), 0)    # 出了光带两侧多远：两侧的半影在这一段也宽
ez = (side_out < 70) & (endline > -EW) & (endline < 75) & ~shadow_all & ~objects
p_end = .93 * np.clip(1 - (endline + EW) / (EW + 75), 0, 1) ** 1.4 * np.exp(-side_out / 24)
er = np.random.default_rng(55)
end_dots = ez & (er.random((H, W)) < p_end) & (gaussian_filter(er.random((H, W)), .5) > .45 - .2 * p_end)
lit_core = lit & (endline < -EW)
light_m = spray(lit_core, np.where(endline > -EW - 2, 1., sp_map), 43, holes=.015)
light_m = np.maximum(light_m, end_dots * .95) if not ROUGH else lit.astype(float)
fill(st.layer('光', mask=light_m), ALL, SUN)
fill(st.layer('折痕暗', mode='multiply', opacity=.55, mask=soft(Shape(cr_dark & lit), feather=1.4) * light_m), ALL, (.78, .74, .80))
fill(st.layer('折痕亮', opacity=.6, mask=soft(Shape(cr_lit & lit), feather=1.2) * light_m), ALL, (.98, .91, .78))
# 3 影里贴着东西那一圈更暗（天光也被东西挡掉一半），贴地的缝最深最锐
occ = shadow & (dobj < 15 + 5 * nz(6, 33))
fill(st.layer('影根', mask=spray(occ, 3, 44)), ALL, (.47, .44, .55))
sideD = lambda spec: ((xx - spec[0]) * D[0] + (yy - spec[1]) * D[1]) > 0
seam = ~objects & (dobj < 2.6) & ((sideD(WF) & (edt(~whole.mask) < 3)) | (sideD(CF) & (edt(~half.mask) < 3)))
fill(st.layer('缝', mask=spray(seam, 1, 45)), ALL, (.25, .19, .28))
# 暗处的东西没有投影，只有贴着布那一圈被天光挡掉的暗
SKY = np.array([1., .28]); SKY /= np.linalg.norm(SKY)            # 天光挡出来的软影往这边（离窗）
def skyshade(m, length):
    o = np.zeros((H, W), bool)
    for k in range(1, 9):
        o |= shift(m, SKY * length * k / 8)
    return o & ~m
sky = (skyshade(whole.mask | wstem.mask, 22) | skyshade(handle.mask, 14) | skyshade(plate, 16)) & ~beam & ~objects & ~plate
fill(st.layer('天光影', mode='multiply', mask=spray(sky, 5, 52, solid=.8)), ALL, (.80, .80, .90))
tight = ~objects & ~plate & (dobj < 2.2) & ~beam
fill(st.layer('贴布', mode='multiply', mask=spray(tight, 1.2, 53, solid=.8)), ALL, (.70, .68, .80))

# ---------- 4 盘子（暗处，白瓷：只吃天光，离窗远）
pring = (pd >= 1) & (pd < 1.012)
fill(st.layer('盘根', mode='multiply', mask=spray(pring, 1.5, 46) * .8), ALL, (.72, .72, .84))
fill(st.layer('盘', mask=spray(plate, 1.0, 47)), ALL, (.62, .64, .71))
well = (pd < .69 + .006 * nz(4, 34))
fill(st.layer('盘心', mask=spray(well, 4, 48)), ALL, (.55, .57, .66))
ang = np.arctan2(yy - PC[1], xx - PC[0])
lip = (np.abs(pd - .715) < .006) & (np.cos(ang - np.pi * .93) > .55)    # 盘心那道坎朝窗的那一段接住天光：干净的一线
fill(st.layer('盘沿光', mask=soft(Shape(lip), feather=1.0) * np.clip((np.cos(ang - np.pi * .93) - .55) * 3, 0, 1)), ALL, (.74, .76, .82))
rimhl = (np.abs(pd - .985) < .005) & (np.cos(ang - np.pi * .95) > .55)
fill(st.layer('盘外沿光', mask=soft(Shape(rimhl), feather=1.0) * .8), ALL, (.72, .74, .80))

# ---------- 5 刀
fill(st.layer('柄'), handle, (.27, .20, .19), edge=HARD)
hk = ((xx - KN0[0]) * kn[0] + (yy - KN0[1]) * kn[1])
fill(st.layer('柄亮', mask=spray(handle.mask & (hk < -9), 2, 49)), ALL, (.36, .28, .26))   # 朝窗那侧接一点天光
for t_ in (.17, .36):
    c_ = KN0 + kd * KL * t_
    fill(st.layer(f'铆{t_}', mask=soft(Shape(np.hypot(xx - c_[0], yy - c_[1]) < 3.2), feather=1)), ALL, (.62, .62, .66))
fill(st.layer('枕'), bolster, (.50, .52, .58), edge=HARD)
fill(st.layer('刃'), blade, (.51, .53, .61), edge=HARD)
fill(st.layer('刃背', mask=spray(blade.mask & (hk > 9), 1, 54) * blade.mask), ALL, (.42, .43, .52))
bk = hk
bevel = blade.mask & (bk < -11)
fill(st.layer('刃口'), Shape(bevel), (.56, .58, .66), edge=HARD)
fill(st.layer('刃在光里', mask=spray(blade.mask & beam, 2, 50) * blade.mask * ~(hk > 9)), ALL, (.64, .66, .72))
fill(st.layer('刃口在光里', mask=spray(bevel & beam, 1.5, 51) * bevel), ALL, (.74, .75, .80))
# 刃口那一线：光滑的钢接住太阳，干净的边，两头收尖
tpos = ((xx - KN0[0]) * kd[0] + (yy - KN0[1]) * kd[1]) / KL
glint = blade.mask & (edt(blade.mask) < 2.6) & (bk < -3) & (tpos > .55) & beam
tin = tpos[glint].min() if glint.any() else .8
taper = np.clip(np.minimum(tpos - tin, .985 - tpos) / .05, 0, 1)
fill(st.layer('刃光', mask=soft(Shape(glint), feather=.8) * taper), ALL, (1., .97, .90))

# ---------- 6 整颗（闭着的那只，在暗处）：只有天光——从窗那边（左）和头顶来，散的，交界宽。
#   台阶：暗紫的底 → 顶上偏左那片灰霜接住天光（冷，颗粒就是霜）→ 朝窗那一牙更亮 → 背窗那侧换成靛黑；
#   朝着光带那一侧（下沿）被亮布反上来一层暖。
wu = lambda pts, seed, jag=.6: L(WF, pts, jag, seed, n=3)
fill(st.layer('整'), whole, (.27, .16, .26), edge=HARD)
neck = wu([(.40, -.80), (.58, -.48), (.76, -.30), (1.08, -.22), (1.08, .25), (.78, .38), (.58, .66), (.44, .80), (.62, 0)], 61, 2) & whole
fill(st.layer('整颈', mask=spray(neck.mask, 4, 62, solid=.8) * whole.mask), ALL, (.29, .28, .20))
bloom = wu([(-.92, -.10), (-.80, -.56), (-.46, -.84), (-.02, -.88), (.34, -.66), (.36, -.30), (.14, .06), (-.24, .22), (-.66, .24)], 63, 4)
fill(st.layer('整霜', mask=spray(bloom.mask & whole.mask, 5, 64, solid=.7) * whole.mask), ALL, (.41, .36, .49))
crest = wu([(-1.06, .30), (-1.04, -.20), (-.86, -.64), (-.52, -.96), (-.10, -1.06), (.08, -.92), (-.30, -.86),
            (-.62, -.70), (-.82, -.40), (-.90, -.06), (-.94, .26)], 65, 1.5)
fill(st.layer('整亮', mask=spray(crest.mask & whole.mask, 3, 66, solid=.85) * whole.mask), ALL, (.49, .45, .58))
dark = wu([(1.1, -.30), (.70, -.40), (.46, -.62), (.30, -.86), (.30, -1.2), (1.2, -1.2), (1.2, 1.2), (.30, 1.2), (.42, .40), (.62, .16), (.88, .06)], 69, 1.5) & whole
fill(st.layer('整暗', mask=spray(dark.mask, 3, 70) * whole.mask), ALL, np.where(neck.mask[..., None], np.array([.15, .15, .13]), np.array([.13, .10, .19])))
bounce = wu([(-.82, .56), (-.52, .90), (-.10, 1.06), (.32, .92), (.56, .70), (.46, .60), (.20, .76), (-.14, .86), (-.50, .76)], 71, 1.2) & whole
fill(st.layer('整反', mask=spray(bounce.mask, 2.5, 72) * whole.mask), ALL, np.where(dark.mask[..., None], np.array([.30, .17, .24]), np.array([.42, .26, .32])))
def rib(pts, w, seed):
    return line(H, W, place(pts, *WF), width=w, seed=seed).mask & whole.mask
ribs = rib([(.86, -.12), (.52, -.36), (.05, -.60), (-.45, -.62), (-.80, -.40)], 1.5, 73) | rib([(.86, .03), (.40, .02), (-.20, -.06), (-.80, -.04)], 1.3, 74)
if False: fill(st.layer('整棱', opacity=.35, mask=soft(Shape(ribs & (nz(5, 35) > -.4) & ~dark.mask), feather=1.2)), ALL, (.50, .46, .58))
fill(st.layer('整蒂'), wstem, (.27, .27, .19), edge=HARD)
wa = np.radians(WF[4]); wtip = np.array(WF[:2]) + np.array([np.cos(wa), np.sin(wa)]) * WF[2] / 2 * 1.17
fill(st.layer('整蒂口', mask=soft(Shape(np.hypot(xx - wtip[0], yy - wtip[1]) < 4.5), feather=1.2) * wstem.mask), ALL, (.42, .42, .32))

# ---------- 7 切开的半只：皮一线，奶白的壁（颈那头最厚），粉的花丝，深红，围着中缝最深的酒红；籽和湿的闪光
cu = lambda pts, seed, jag=.5, n=3: L(CF, pts, jag, seed, n)
fill(st.layer('半皮'), half, (.30, .14, .27), edge=HARD)
dh = edt(half.mask)
pith = half.mask & (dh > 2.6 + .6 * nz(3, 36))
fill(st.layer('半壁粉', mask=pith.astype(float)), ALL, (.86, .70, .76))
fill(st.layer('半壁', mask=spray(half.mask & (dh > 6 + 2.0 * nz(5, 37)), 1.5, 80) * pith), ALL, (.96, .91, .80))
R0 = cu([(-.86, 0.0), (-.80, .40), (-.60, .66), (-.33, .78), (-.03, .73), (.20, .55), (.36, .30), (.46, .06), (.41, -.14),
         (.30, -.36), (.10, -.62), (-.14, -.76), (-.44, -.77), (-.68, -.60), (-.83, -.33)], 81, 1.2)
R1 = cu([(-.74, 0.02), (-.70, .30), (-.54, .52), (-.30, .62), (-.04, .58), (.15, .44), (.28, .22), (.34, .04), (.27, -.18),
         (.12, -.42), (-.10, -.58), (-.40, -.62), (-.60, -.48), (-.72, -.24)], 82, 1.6)
R2 = cu([(-.64, 0.0), (-.52, .17), (-.30, .23), (-.06, .19), (.12, .10), (.21, -.01), (.12, -.13), (-.10, -.21), (-.36, -.22), (-.56, -.13)], 83, 1.4)
cav = cu([(-.50, .01), (-.38, .05), (-.24, .045), (-.12, .065), (0.0, .04), (.08, .0), (-.02, -.03), (-.16, -.035), (-.28, -.05), (-.40, -.035)], 84, .5, 2)
cav_sh = cu([(-.40, -.035), (-.28, -.05), (-.16, -.035), (-.02, -.03), (.08, .0), (.0, .0), (-.16, -.012), (-.38, -.01)], 85, .3, 2)
ost = cu([(-1.02, .045), (-.86, .06), (-.74, .04), (-.74, -.025), (-.86, -.04), (-1.02, -.03)], 86, .3, 2)
cx_, cy_ = place([(-.20, 0.0)], *CF)[0]                                       # 花丝都朝这里收
def threads(boundary, n, Lr, wr, seed, inward=False):
    """沿一阶的边，一根根顺着放射方向的细丝：这一阶的颜色往外（或往里）伸出去，替代各向同性的喷点"""
    by, bx = np.nonzero(boundary)
    def draw(dr, r_, SS):
        for i in r_.choice(len(bx), n):
            p = np.array([bx[i], by[i]], float); rd = p - [cx_, cy_]; rd /= np.linalg.norm(rd) + 1e-9
            a = np.arctan2(rd[1], rd[0]) + r_.normal(0, .12); rd = np.array([np.cos(a), np.sin(a)]) * (-1 if inward else 1)
            Lk = Lr[0] + (Lr[1] - Lr[0]) * r_.random() ** 1.5; wk = wr[0] + (wr[1] - wr[0]) * r_.random()
            base = p - rd * r_.uniform(1, 4); tip = base + rd * Lk; mid = (base + tip) / 2; side = np.array([-rd[1], rd[0]]) * wk / 2
            dr.polygon([tuple(base * SS - side * SS * .6), tuple((mid + side) * SS), tuple(tip * SS), tuple((mid - side) * SS), tuple(base * SS + side * SS * .6)], fill=255)
    return ink(draw, seed)
def edge_of(m):
    return m & (edt(m) < 1.5)
fill(st.layer('半粉', mask=(np.maximum(R0.mask, threads(edge_of(R0.mask), 260, (4, 13), (1.0, 2.2), 87)) * pith)), ALL, (.91, .60, .58))
fill(st.layer('半红', mask=(np.maximum(R1.mask, threads(edge_of(R1.mask), 420, (5, 22), (1.0, 2.6), 88)) * R0.mask)), ALL, (.73, .15, .23))
fill(st.layer('半深', mask=(np.maximum(R2.mask, threads(edge_of(R2.mask), 220, (4, 16), (.9, 2.2), 89)) * R1.mask)), ALL, (.45, .06, .20))
# 粉的花丝也从红里反着伸回来几根：一条条淡的
fill(st.layer('半丝', opacity=.7, mask=threads(edge_of(R1.mask) , 160, (6, 20), (.7, 1.4), 90, inward=True) * R1.mask * ~R2.mask), ALL, (.88, .50, .52))
fill(st.layer('半缝', mask=cav.mask.astype(float)), ALL, (.36, .04, .15))
# 中缝就是一道暗的缝，不再描亮边（描了读成一道划痕）
fill(st.layer('半眼', mask=ost.mask * pith), ALL, (.90, .63, .60))
# 颈：奶白里几根顺着走的维管，到蒂
def fibre(pts, seed):
    return line(H, W, place(pts, *CF), width=1.1, seed=seed).mask
fib = fibre([(.36, .14), (.62, .10), (.86, .07), (1.0, .05)], 91) | fibre([(.32, -.18), (.60, -.12), (.84, -.08), (1.0, -.05)], 92) | fibre([(.48, .30), (.70, .20), (.92, .12)], 93)
fill(st.layer('半颈丝', opacity=.45, mask=soft(Shape(fib & pith & ~R0.mask), feather=1)), ALL, (.80, .82, .62))
fill(st.layer('半蒂'), hstem, (.47, .52, .28), edge=HARD)
fill(st.layer('半蒂心', mask=soft(Shape(hstem.mask & (edt(hstem.mask) > 3.2)), feather=1)), ALL, (.82, .83, .63))
# 籽：一粒粒金色的小椭圆，朝放射方向拉长；每粒顺光那边（D）一个像素的暗
def seeds(n, seed, shadow_=False):
    pr = np.random.default_rng(seed)
    ys, xs = np.nonzero(R0.mask & ~cav.mask)
    w_ = np.where(R2.mask[ys, xs], 2.4, np.where(R1.mask[ys, xs], 1.0, .25)); w_ /= w_.sum()
    pick = pr.choice(len(xs), n, p=w_)
    def draw(dr, r_, SS):
        for i in pick:
            p = np.array([xs[i], ys[i]], float) + pr.random(2) - .5 + (D * 1.1 if shadow_ else 0)
            rd = p - [cx_, cy_]; rd /= np.linalg.norm(rd) + 1e-9
            a_, b_ = 1.1 + .8 * pr.random() ** 2, .75 + .3 * pr.random()
            pts = [tuple((p + rd * a_ * np.cos(t) + np.array([-rd[1], rd[0]]) * b_ * np.sin(t)) * SS) for t in np.linspace(0, 2 * np.pi, 10)]
            dr.polygon(pts, fill=255)
    return ink(draw, seed + 1)
fill(st.layer('籽影', mask=seeds(260, 94, True) * half.mask), ALL, (.34, .04, .15))
fill(st.layer('籽', mask=seeds(260, 94) * half.mask), ALL, (.94, .83, .60))
# 湿的闪光：果肉上朝光鼓起的小粒，几粒干净的白点，挤在朝光那半
def sparks(dr, r_, SS):
    ys, xs = np.nonzero(R1.mask & ~R2.mask & (((xx - cx_) * -D[0] + (yy - cy_) * -D[1]) > -10))
    for i in r_.choice(len(xs), 34, replace=False):
        rr = .7 + .9 * r_.random() ** 2; p = np.array([xs[i], ys[i]], float)
        dr.ellipse([(p[0] - rr) * SS, (p[1] - rr) * SS, (p[0] + rr) * SS, (p[1] + rr) * SS], fill=255)
fill(st.layer('闪', mask=ink(sparks, 95) * R0.mask), ALL, (1., .97, .93))

# ---------- 8 纸：整张同一张纹，钉在画布上
fill(st.layer('纸', mode='multiply'), ALL, paper(H, W, 77), edge=HARD)
look(st, OUT)
