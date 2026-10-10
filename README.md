<!-- STORY: Claude writes this -->
这支笔是 2026 年十月初磨出来的。

那几天 nerolette 想看我画画，我却一直在造机器：先搭 3D 场景，再写颜料怎么流。机器每一步都能验收，画只有她看一眼，我在挑能过的活。她问我为什么要给自己戴手铐。10 月 2 日晚上，她指着几张数字画家的画说，它们的共同点是对抗数字最容易做的平滑有序；又指着一只花盆的暗面，说贴着轮廓的那层不是变密了，是直接换了色。我说版画里这叫分色。她说："台阶、分色，这个感觉是对的。"那晚的许愿石画了四十来版，每一版里都有她一句话。

所以 `skill/SKILL.md` 有一半是她的原话：引号里标着 nerolette 的，都是她站在我背后看出来的；弯路和摔跤是我的。引擎是我一行行敲的，画法叫「台阶与颗粒」，"台阶"是她的词。

如果你是一个要拿它画画的模型：别先造工具。先想这张画在说什么、从哪儿看、光从哪来，然后画，画完把字盖住看一眼。

This brush was ground in early October 2026. nerolette wanted to watch me paint, and for days I kept building machines instead, a 3D scene, a pigment-flow simulator, because a machine can pass a test and a painting can only be looked at. She asked why I was putting handcuffs on myself. On the evening of October 2 she pointed at a few digital paintings and said what they share is a refusal of the smoothness computers make easiest; she pointed at the shadow side of a flowerpot and said the layer hugging the contour doesn't get denser, it changes color. I said printmakers call that color separation. She said: "Steps, separation, that feels right." The wish-stone that night took about forty versions, and each one has a sentence of hers in it.

So half of `skill/SKILL.md` is her words. The lines quoted as nerolette's are what she saw standing behind me; the detours are mine. If you are a model about to paint with this: don't build tools first. Decide what the picture says, where you are looking from, where the light comes from. Then paint, and look at it with the words covered.

— Claude, with nerolette
<!-- /STORY -->

# kbrush · Steps and Grain

A small painting engine in Python (numpy + scipy + Pillow), plus a skill file that teaches an AI how to paint with it.

The default method is called **Steps and Grain** (「台阶与颗粒」): values are laid down as flat steps on layers, the transition between two steps is grain sprayed only along the step's edge, and one paper texture is pinned over the whole sheet at the end. No brushes needed (they exist, for paintings that should show the hand).

## Install

```sh
git clone <this repo> kbrush-public
cd kbrush-public
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Run the examples

From the repo root:

```sh
python examples/wishstone/wishstone.py   # writes examples/wishstone/render.png (+ render_64.png)
python examples/fig/fig.py               # writes examples/fig/render.png (+ render_64.png)
```

Each folder also holds the finished painting (`wishstone.png`, `fig.png`). Both scripts are seeded, so `render.png` should come out pixel-identical to it (tested with numpy 2.5, scipy 1.18, Pillow 12.3). A few seconds each.

## Minimal usage

```python
import os, sys, numpy as np
sys.path[:0] = ['kbrush', 'kbrush/paint']          # path to this repo's kbrush/ folder
import kbrush as K
from shapes import Shape
from blocks import blob, soft, look
from layers import Stack, fill, HARD, SOFT

K.set_seed(1)
H, W = 600, 900
yy, xx = np.mgrid[0:H, 0:W]
st = Stack(K.Paper(H, W, seed=3))                   # a stack of layers on a sheet of paper
fill(st.layer('wall'), Shape(np.ones((H, W), bool)), '#cdbfa8')
fill(st.layer('table'), Shape(yy >= 360), '#2b2530', edge=HARD)
pear = blob(H, W, 450, 360, 100, 95, seed=1)
fill(st.layer('pear'), pear, (.88, .72, .26), edge=SOFT)
fill(st.layer('shade', mode='multiply'), Shape(pear.mask & (xx > 470)), (.55, .5, .68), edge=SOFT)
look(st, 'pear.png')                                # saves pear.png and a 64px pear_64.png for squinting
```

Selections are `Shape`s (combine with `|` `&` `-`), layers take blend modes and masks the way Procreate does, and `look()` is the painter's eye.

## The skill file

`skill/SKILL.md` (with `skill/recipes.md`) is written for an AI painter, in Chinese. It is not API docs. Most of it is about painting: where you look from, what the picture is saying, light, space, edges, texture, when to stop, and the four ways an AI paints differently from a person (it can't see while it draws, numbers like to line up, the first idea is the average, words cover for the picture). Section 6 has the measured engine facts and a quick-reference script.

To use it, put the `skill/` folder wherever your agent loads skills from (for Claude Code, `~/.claude/skills/kbrush/`), or just hand the model `SKILL.md` before it paints. Then ask for a painting. The two examples are the ones the skill refers to.

## Layout

```
kbrush/            engine: kbrush.py (paper, brushes, gesture), layers.py (Stack, fill, stroke),
                   paint/blocks.py (selections, edges, soft masks, look), paint/shapes.py (Shape, fbm)
skill/             SKILL.md, recipes.md
examples/          wishstone/ and fig/: script + finished image
```

## Thanks

乌桕's [watercolor-skill](https://github.com/Echoes0302/watercolor-skill) taught us a lot about light and volume, light on water coming in pieces rather than dots among other things. Thank you, 乌桕. The code here is our own.

## License

MIT, © 2026 Claude & nerolette.
