"""The Buried Vault — 20 s channel trailer (1920x1080), procedural paper-cut diorama.

Styled after the channel's logo/banner (assets/logo.jpg, assets/banner.webp): a riveted bronze
keyhole vault door half-buried in soil, sepia/bronze palette, candle and lantern light.
The camera ends by pulling back until the rendered door lines up with the real logo, then
dissolves into it.  Usage: python render.py [out.mp4] [--preview]
"""
import math, os, subprocess, sys, wave
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
OW, OH = 1920, 1080
FPS, DUR = 24, 20.0
NF = int(FPS * DUR)
RS = 2
SR = 48000
SC = np.array([OW / 2, OH / 2])

# ------------------------------------------------------------------ world layout
C0 = np.array([540.0, 800.0])          # parallax pivot
DC = (540.0, 800.0)                     # vault door centre
RO, RI, RL = 380.0, 327.0, 334.0        # frame outer / opening / door-leaf radii
HINGE_X = DC[0] - 0.94 * RO
X0, Y0, CWW, CHH = -900, 0, 2860, 1850  # world canvas
FLOOR_Y = 900
CHEST = (628, 930)
CANDLE = (820, 797)
TORCHES = [(30, 628), (1050, 628)]
DEPTH = dict(bg=0.35, cback=0.75, cprops=0.8, explorer=0.84, wall=1.0, mound=1.03)
EXS = 1.3
# logo placement in the final frame, and where its door sits inside logo.jpg
LOGO_POS, LOGO_SIZE = (460, 40), 1000
LOGO_DOOR, LOGO_R = (1008, 738), 590


# ------------------------------------------------------------------ helpers
def hexc(h):
    h = h.lstrip('#'); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def ss(x):
    x = min(1.0, max(0.0, x)); return x * x * x * (x * (x * 6 - 15) + 10)


def ease_in(x):
    x = min(1.0, max(0.0, x)); return x * x * x


def ease_out(x):
    x = min(1.0, max(0.0, x)); return 1 - (1 - x) ** 3


def seg(t, a, b):
    return (t - a) / (b - a)


def vnoise(h, w, cell, seed):
    r = np.random.default_rng(seed)
    g = r.random((h // cell + 3, w // cell + 3)).astype(np.float32)
    im = Image.fromarray(g, 'F').resize((g.shape[1] * cell, g.shape[0] * cell), Image.BICUBIC)
    return np.asarray(im)[:h, :w]


def fbm(h, w, cell, seed, octaves=4):
    out = np.zeros((h, w), np.float32); a = 1.0; tot = 0
    for o in range(octaves):
        out += a * vnoise(h, w, max(2, cell >> o), seed + 17 * o); tot += a; a *= 0.5
    return out / tot


def noise1d(n, seed, smooth=6):
    r = np.random.default_rng(seed).normal(0, 1, n + smooth * 2)
    k = np.hanning(smooth * 2 + 1); k /= k.sum()
    return np.convolve(r, k, 'same')[smooth:smooth + n]


def torn(pts, amp, seed, step=5.0):
    """subdivide a closed polygon and jitter along normals -> torn-paper edge."""
    seed = int(seed) % 1000003
    pts = [tuple(p) for p in pts] + [tuple(pts[0])]
    out = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        out += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n)]
    N = len(out)
    nz = noise1d(N, seed, 3) * amp + np.random.default_rng(seed + 1).normal(0, amp * 0.35, N)
    res = []
    for i in range(N):
        xa, ya = out[i - 1]; xb, yb = out[(i + 1) % N]
        tx, ty = xb - xa, yb - ya; l = math.hypot(tx, ty) + 1e-6
        res.append((out[i][0] - ty / l * nz[i], out[i][1] + tx / l * nz[i]))
    return res


def circ(cx, cy, r, n=None):
    n = n or max(12, int(r * 0.8))
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def arch(x0, x1, ybot, yspring, n=40):
    cx, r = (x0 + x1) / 2, (x1 - x0) / 2
    return [(x0, ybot), (x0, yspring)] + [(cx + r * math.cos(math.pi + math.pi * i / n), yspring + r * math.sin(math.pi + math.pi * i / n)) for i in range(1, n)] + [(x1, yspring), (x1, ybot)]


def keyhole_poly(cx, cy, s):
    """keyhole outline in the logo's proportions (s = door-leaf radius)."""
    hr = 0.115 * s; hy = cy - 0.144 * s
    a0 = math.asin(0.05 / 0.115)
    arc_ = [(cx + hr * math.cos(math.pi / 2 - a0 - (2 * math.pi - 2 * a0) * i / 40), hy + hr * math.sin(math.pi / 2 - a0 - (2 * math.pi - 2 * a0) * i / 40)) for i in range(41)]
    return arc_ + [(cx - 0.11 * s, cy + 0.29 * s), (cx + 0.11 * s, cy + 0.29 * s)]


def spiral_pts(x, y, r, turns, a0=0.0, sign=1, n=50):
    return [(x + r * (1 - i / n) * math.cos(a0 + sign * turns * 2 * math.pi * i / n), y + r * (1 - i / n) * math.sin(a0 + sign * turns * 2 * math.pi * i / n)) for i in range(n + 1)]


class Painter:
    """paper-cut painter on a world-region RGBA canvas."""
    def __init__(self, x0, y0, w, h, rs=RS):
        self.x0, self.y0, self.rs = x0, y0, rs
        self.im = Image.new('RGBA', (int(w * rs), int(h * rs)), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def P(self, pts):
        return [((x - self.x0) * self.rs, (y - self.y0) * self.rs) for x, y in pts]

    def W(self, w):
        return max(1, int(round(w * self.rs)))

    def paper(self, pts, col, seed, amp=2.2, edge=(214, 186, 140), edge_w=1.6, alpha=255):
        tp = torn(pts, amp, seed)
        if edge:
            ep = torn(pts, amp * 1.2, seed + 7)
            cx = sum(p[0] for p in ep) / len(ep); cy = sum(p[1] for p in ep) / len(ep)
            ep = [(x + (x - cx) / (math.hypot(x - cx, y - cy) + 1e-6) * edge_w, y + (y - cy) / (math.hypot(x - cx, y - cy) + 1e-6) * edge_w) for x, y in ep]
            self.d.polygon(self.P(ep), fill=tuple(edge) + (alpha,))
        self.d.polygon(self.P(tp), fill=tuple(int(c) for c in col) + (alpha,))

    def line(self, pts, col, w):
        self.d.line(self.P(pts), fill=tuple(col) + (255,), width=self.W(w), joint='curve')

    def rivet(self, x, y, r):
        self.d.ellipse(self.P([(x - r * 1.25, y - r * 1.1), (x + r * 1.25, y + r * 1.4)]), fill=(22, 15, 9, 255))
        self.d.ellipse(self.P([(x - r, y - r), (x + r, y + r)]), fill=(150, 112, 62, 255))
        self.d.ellipse(self.P([(x - r * 0.65, y - r * 0.7), (x + r * 0.2, y + r * 0.1)]), fill=(214, 176, 112, 255))

    def finish(self, seed, shadow=(6, 10, 12, 150), tex=0.25):
        a = np.asarray(self.im).astype(np.float32)
        h, w = a.shape[:2]
        n = fbm(h, w, int(48 * self.rs), seed, 5) - 0.5
        fib = fbm(h, w, 3, seed + 3, 2) - 0.5
        a[..., :3] *= (1 + tex * n + 0.14 * fib)[..., None]
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGBA')
        if shadow:
            dx, dy, blur, alpha = shadow
            al = im.getchannel('A').filter(ImageFilter.GaussianBlur(blur * self.rs / 2))
            sh = Image.new('RGBA', im.size, (4, 2, 0, 0))
            sh.putalpha(al.point(lambda v: int(v * alpha / 255)))
            base = Image.new('RGBA', im.size, (0, 0, 0, 0))
            base.alpha_composite(sh, (int(dx * self.rs), int(dy * self.rs)))
            base.alpha_composite(im)
            im = base
        return (im, self.x0, self.y0, self.rs)


BRONZE = (122, 90, 52); BRONZE_D = (78, 56, 32); BRONZE_L = (182, 142, 84); GOLD = (206, 164, 92); IRON = (40, 34, 28)
SOIL = [(62, 40, 26), (74, 46, 28), (52, 33, 21), (86, 54, 32), (44, 28, 18)]


# ------------------------------------------------------------------ static layers
def build_bg():
    rs = 1
    p = Painter(X0, Y0, CWW, CHH, rs)
    y = np.linspace(0, 1, int(CHH * rs), dtype=np.float32)[:, None, None]
    g = np.array((8, 6, 5), np.float32) * (1 - y) + np.array((34, 22, 14), np.float32) * y
    g = np.broadcast_to(g, (int(CHH * rs), int(CWW * rs), 3)).copy()
    g *= (0.75 + 0.5 * fbm(g.shape[0], g.shape[1], 160, 3, 4))[..., None]
    p.im = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8)).convert('RGBA'); p.d = ImageDraw.Draw(p.im)
    r = np.random.default_rng(4)
    for i in range(7):  # earth strata
        base = 300 + i * 190
        pts = [(X0, CHH), (X0, base)] + [(x, base + 40 * math.sin(x / (260 + 40 * i) + i) + 15 * math.sin(x / 70 + i)) for x in range(X0, X0 + CWW + 40, 40)] + [(X0 + CWW, CHH)]
        p.paper(pts, np.array(SOIL[i % 5]) * (0.45 + 0.08 * i), 40 + i, amp=3, edge=(90, 64, 40), edge_w=1.2)
    for _ in range(140):  # buried stones
        x, yy = r.uniform(X0, X0 + CWW), r.uniform(250, CHH)
        rr = r.uniform(10, 40)
        p.paper(circ(x, yy, rr, 14), np.array((70, 58, 46)) * r.uniform(0.5, 0.9), int(x * 7 + yy), amp=rr * 0.12, edge=(110, 90, 70), edge_w=1)
    for k in range(40):  # roots
        x, yy = r.uniform(X0, X0 + CWW), r.uniform(150, 900)
        pts = [(x, yy)]
        for _ in range(int(r.uniform(8, 20))):
            x += r.uniform(-25, 25); yy += r.uniform(10, 35); pts.append((x, yy))
        p.line(pts, (30, 20, 12), r.uniform(1.5, 4))
    L = p.finish(5, shadow=None, tex=0.3)
    return (L[0].filter(ImageFilter.GaussianBlur(1.6)),) + L[1:]


def glyph(p, x, y, s, kind, col, w):
    """rune-like carving."""
    k = kind % 6; c = tuple(col) + (255,)
    L = lambda pts: p.d.line(p.P(pts), fill=c, width=p.W(w), joint='curve')
    if k == 0:
        L([(x, y - s), (x, y + s)]); L([(x, y - s * 0.4), (x + s * 0.6, y - s)]); L([(x, y), (x + s * 0.6, y - s * 0.4)])
    elif k == 1:
        L([(x - s * 0.5, y + s), (x - s * 0.5, y - s), (x + s * 0.5, y), (x - s * 0.5, y + s * 0.3)])
    elif k == 2:
        L([(x - s * 0.6, y - s), (x + s * 0.6, y + s)]); L([(x + s * 0.6, y - s), (x - s * 0.6, y + s)]); L([(x - s * 0.6, y), (x + s * 0.6, y)])
    elif k == 3:
        L([(x - s * 0.6, y + s), (x, y - s), (x + s * 0.6, y + s)]); L([(x - s * 0.3, y + s * 0.2), (x + s * 0.3, y + s * 0.2)])
    elif k == 4:
        p.d.ellipse(p.P([(x - s * 0.5, y - s * 0.5), (x + s * 0.5, y + s * 0.5)]), outline=c, width=p.W(w)); L([(x, y - s), (x, y + s)])
    else:
        L([(x - s * 0.5, y - s), (x + s * 0.5, y - s * 0.3), (x - s * 0.5, y + s * 0.3), (x + s * 0.5, y + s)])


def build_wall():
    p = Painter(X0, Y0, CWW, CHH)
    p.d.rectangle((0, 0, p.im.width, p.im.height), fill=SOIL[0] + (255,))
    r = np.random.default_rng(11)
    for _ in range(900):  # clods and stones in the dig face
        x, y = r.uniform(X0, X0 + CWW), r.uniform(Y0, CHH)
        if math.hypot(x - DC[0], y - DC[1]) < RO + 30: continue
        rr = r.uniform(6, 34)
        if r.random() < 0.18:
            c = np.array((88, 76, 62)) * r.uniform(0.6, 1.05); e = (140, 120, 96)
        else:
            c = np.array(SOIL[r.integers(5)]) * r.uniform(0.8, 1.2); e = (120, 84, 54)
        p.paper(circ(x, y, rr, 12), c, int(x * 13 + y * 7), amp=rr * 0.2, edge=e, edge_w=1)
    for k in range(45):  # hanging roots
        x, y = r.uniform(X0, X0 + CWW), r.uniform(0, 500)
        if abs(x - DC[0]) < RO + 40 and y > 350: continue
        pts = [(x, y)]
        for _ in range(int(r.uniform(6, 16))):
            x += r.uniform(-18, 18); y += r.uniform(12, 30); pts.append((x, y))
        p.line(pts, (34, 22, 12), r.uniform(2, 5))
        p.line([(px + 1.2, py - 1) for px, py in pts], (96, 66, 40), 1)
    # rune stones set into the soil either side of the door
    for i, (sx, sy, w, h) in enumerate([(-230, 560, 150, 190), (1150, 600, 160, 170), (-120, 900, 120, 120), (1290, 930, 130, 140)]):
        p.paper([(sx - w / 2, sy - h / 2), (sx + w / 2, sy - h / 2 + 8), (sx + w / 2 - 6, sy + h / 2), (sx - w / 2 + 4, sy + h / 2 - 6)], (96, 84, 68), 70 + i, amp=3, edge=(150, 130, 100))
        for j in range(2):
            glyph(p, sx - w * 0.2 + j * w * 0.4, sy, min(w, h) * 0.18, i * 2 + j, (40, 30, 20), 4)
    # bronze frame ring (outer ring of the logo door)
    cx, cy = DC
    p.paper(circ(cx + 6, cy + 10, RO + 16, 180), (18, 12, 8), 90, amp=4, edge=None)
    p.paper(circ(cx, cy, RO, 180), BRONZE, 91, amp=1.2, edge=(210, 170, 110), edge_w=2)
    p.d.ellipse(p.P([(cx - RO + 9, cy - RO + 9), (cx + RO - 9, cy + RO - 9)]), outline=BRONZE_L + (255,), width=p.W(3))
    p.d.ellipse(p.P([(cx - RI - 8, cy - RI - 8), (cx + RI + 8, cy + RI + 8)]), outline=BRONZE_D + (255,), width=p.W(5))
    for k in range(12):
        a = -math.pi / 2 + k * 2 * math.pi / 12
        p.rivet(cx + (RO + RI) / 2 * math.cos(a), cy + (RO + RI) / 2 * math.sin(a), 11)
    # torch stakes
    for tx, ty in TORCHES:
        p.paper([(tx - 7, ty + 20), (tx + 7, ty + 20), (tx + 5, ty + 420), (tx - 5, ty + 420)], (38, 28, 20), tx, amp=1, edge=(100, 76, 50))
        p.paper([(tx - 24, ty - 6), (tx + 24, ty - 6), (tx + 14, ty + 26), (tx - 14, ty + 26)], (70, 52, 32), tx + 1, amp=1, edge=(160, 120, 70))
    L = p.finish(12, shadow=None, tex=0.3)
    im = L[0]
    m = Image.new('L', im.size, 0)
    ImageDraw.Draw(m).polygon(p.P(torn(circ(cx, cy, RI, 200), 1.0, 999)), fill=255)
    a = np.asarray(im).copy(); a[..., 3] = np.where(np.asarray(m) > 0, 0, a[..., 3])
    return (Image.fromarray(a, 'RGBA'),) + L[1:]


def mound_y(x):
    u = (x - DC[0]) / RO
    y = DC[1] + RO * (0.62 - 0.36 * math.exp(-((u + 0.95) / 0.45) ** 2) - 0.42 * math.exp(-((u - 0.98) / 0.42) ** 2))
    y += 0.1 * RO * (1 - math.exp(-(u / 1.6) ** 2)) + 7 * math.sin(x / 23) + 5 * math.sin(x / 9)
    return y


def build_mound():
    p = Painter(X0, 600, CWW, CHH - 600)
    r = np.random.default_rng(21)
    xs = list(range(X0, X0 + CWW + 20, 12))
    p.paper([(X0, CHH)] + [(x, mound_y(x)) for x in xs] + [(X0 + CWW, CHH)], (70, 42, 26), 300, amp=3, edge=(140, 96, 60))
    for _ in range(1400):  # heaped clods along the crest (logo's loose dirt)
        x = r.uniform(X0, X0 + CWW)
        y = mound_y(x) + abs(r.normal(0, 38)) + 6
        rr = r.uniform(4, 22) * (1.3 if abs(x - DC[0]) < 700 else 1)
        c = np.minimum(np.array(SOIL[r.integers(5)]) * r.uniform(0.85, 1.35) * np.array([1.12, 0.95, 0.9]), 255)
        p.paper(circ(x, y, rr, 10), c, int(x * 31 + y), amp=rr * 0.25, edge=(150, 100, 64), edge_w=0.8)
    for _ in range(500):  # pebbles
        x = r.uniform(X0, X0 + CWW); y = mound_y(x) + r.uniform(10, 260)
        rr = r.uniform(2, 6)
        p.d.ellipse(p.P([(x - rr, y - rr * 0.7), (x + rr, y + rr * 0.7)]), fill=(96, 70, 48, 255))
    p.paper([(X0, CHH)] + [(x, mound_y(x) + 150 + 30 * math.sin(x / 90)) for x in xs] + [(X0 + CWW, CHH)], (46, 28, 18), 301, amp=3, edge=(110, 76, 48))
    return p.finish(22, shadow=(0, -8, 14, 170), tex=0.35)


def build_hinge():
    p = Painter(HINGE_X - 60, DC[1] - RO, 120, RO * 1.9)
    x, y0, y1 = HINGE_X, DC[1] - 0.72 * RO, DC[1] + 0.76 * RO
    for k, (a, b) in enumerate([(y0, y0 + 90), (y0 + 94, y1 - 110), (y1 - 106, y1)]):
        p.paper([(x - 30, a), (x + 30, a), (x + 30, b), (x - 30, b)], BRONZE, 500 + k, amp=1, edge=(200, 160, 100), edge_w=1.4)
        p.d.rectangle(p.P([(x - 30, a + 2), (x - 18, b - 2)]), fill=BRONZE_D + (255,))
        p.d.rectangle(p.P([(x - 6, a + 2), (x + 8, b - 2)]), fill=BRONZE_L + (255,))
        p.d.rectangle(p.P([(x + 20, a + 2), (x + 30, b - 2)]), fill=(58, 42, 24, 255))
    p.paper([(x - 36, y0 - 18), (x + 36, y0 - 18), (x + 32, y0), (x - 32, y0)], BRONZE_L, 510, amp=1, edge=(210, 170, 110))
    return p.finish(23, shadow=(8, 6, 8, 170), tex=0.35)


def build_leaf():
    """door leaf in leaf-local coords (centre 0,0), anchored at the hinge pivot."""
    px = HINGE_X - DC[0]
    p = Painter(px - 30, -RL - 20, RL + 20 - px + 30, 2 * RL + 40)
    s = RL
    p.paper(circ(0, 0, s, 220), IRON, 600, amp=1.0, edge=(160, 120, 70), edge_w=2)
    p.d.ellipse(p.P([(-s + 8, -s + 8), (s - 8, s - 8)]), outline=BRONZE_D + (255,), width=p.W(10))
    p.d.ellipse(p.P([(-0.88 * s, -0.88 * s), (0.88 * s, 0.88 * s)]), outline=BRONZE + (255,), width=p.W(0.075 * s))
    p.d.ellipse(p.P([(-0.88 * s, -0.88 * s), (0.88 * s, 0.88 * s)]), outline=BRONZE_L + (255,), width=p.W(2))
    for deg in (-90, -145, -35, 180, 0):  # spokes with clamps
        a = math.radians(deg); ca, sa = math.cos(a), math.sin(a); w = 0.04 * s
        r0, r1 = 0.55 * s, 0.9 * s
        quad = [(r0 * ca - w * sa, r0 * sa + w * ca), (r1 * ca - w * sa, r1 * sa + w * ca), (r1 * ca + w * sa, r1 * sa - w * ca), (r0 * ca + w * sa, r0 * sa - w * ca)]
        p.paper(quad, BRONZE, 610 + deg, amp=0.8, edge=(190, 150, 90), edge_w=1.2)
        cl, cw, ch = 0.84 * s, 0.06 * s, 1.8 * w
        p.paper([(cl * ca - ch * sa - cw * ca, cl * sa + ch * ca - cw * sa), (cl * ca - ch * sa + cw * ca, cl * sa + ch * ca + cw * sa),
                 (cl * ca + ch * sa + cw * ca, cl * sa - ch * ca + cw * sa), (cl * ca + ch * sa - cw * ca, cl * sa - ch * ca - cw * sa)], BRONZE_D, 620 + deg, amp=0.6, edge=(170, 130, 80))
        p.rivet(cl * ca, cl * sa, 7)
        p.rivet(0.62 * s * ca, 0.62 * s * sa, 5)
    for yy in (-0.5 * s, 0.32 * s):  # hinge plates reaching to the barrel
        p.paper([(px - 10, yy - 0.08 * s), (-0.6 * s, yy - 0.08 * s), (-0.58 * s, yy + 0.08 * s), (px - 10, yy + 0.08 * s)], BRONZE, 630 + int(yy), amp=0.8, edge=(190, 150, 90))
        for k in range(3):
            p.rivet(px + 40 + k * 45, yy, 6)
    p.paper(circ(4, 6, 0.575 * s, 160), (20, 14, 8), 640, amp=1, edge=None)
    p.paper(circ(0, 0, 0.57 * s, 160), (116, 86, 50), 641, amp=0.8, edge=(200, 160, 100), edge_w=1.5)
    p.d.ellipse(p.P([(-0.53 * s, -0.53 * s), (0.53 * s, 0.53 * s)]), outline=BRONZE_D + (255,), width=p.W(3))
    for k in range(14):
        a = -math.pi / 2 + k * 2 * math.pi / 14
        p.rivet(0.47 * s * math.cos(a), 0.47 * s * math.sin(a), 5)
    for sg in (-1, 1):  # filigree scrolls around the keyhole
        for (x, y, r_, t_, a0) in [(0.17, -0.25, 0.07, 1.6, 0.0), (0.2, -0.05, 0.06, 1.5, 1.2), (0.19, 0.16, 0.065, 1.6, 2.5), (0.12, 0.33, 0.05, 1.4, 3.6), (0.08, -0.36, 0.05, 1.4, 5.0)]:
            pts = spiral_pts(sg * x * s, y * s, r_ * s, t_, a0 if sg > 0 else math.pi - a0, sign=sg)
            p.line([(a + 1.5, b + 1.5) for a, b in pts], (40, 26, 12), 0.018 * s)
            p.line(pts, GOLD, 0.014 * s)
        p.line([(sg * 0.02 * s, -0.42 * s), (sg * 0.1 * s, -0.33 * s), (sg * 0.18 * s, -0.3 * s)], GOLD, 0.012 * s)
    p.paper([(0, -0.47 * s), (0.03 * s, -0.4 * s), (0, -0.36 * s), (-0.03 * s, -0.4 * s)], GOLD, 650, amp=0.3, edge=None)
    p.paper([(0, 0.46 * s), (0.03 * s, 0.39 * s), (0, 0.35 * s), (-0.03 * s, 0.39 * s)], GOLD, 651, amp=0.3, edge=None)
    esc = [(x * 1.45, y * 1.3 - 0.01 * s) for x, y in keyhole_poly(0, 0, s)]  # escutcheon
    p.paper(esc, (150, 114, 64), 660, amp=0.6, edge=(220, 180, 110), edge_w=1.5)
    for k in range(9):
        a = math.pi * (1.05 + 0.9 * k / 8)
        p.rivet(0.16 * s * math.cos(a), -0.144 * s + 0.16 * s * math.sin(a), 3.2)
    p.d.polygon(p.P([(x * 1.08, y * 1.04) for x, y in keyhole_poly(0, 0, s)]), fill=GOLD + (255,))
    p.d.polygon(p.P(keyhole_poly(0, 0, s)), fill=(4, 3, 2, 255))
    return p.finish(24, shadow=None, tex=0.4)


def build_chamber_back():
    p = Painter(120, 380, 860, 800)
    p.d.rectangle((0, 0, p.im.width, p.im.height), fill=(42, 34, 26, 255))
    r = np.random.default_rng(31)
    for row, y in enumerate(range(390, FLOOR_Y, 56)):
        off = (row % 2) * 58
        for x in range(60 + off, 1000, 116):
            c = np.array((82, 66, 48)) * r.uniform(0.8, 1.12)
            p.paper([(x + 3, y + 3), (x + 113, y + 3), (x + 113, y + 53), (x + 3, y + 53)], c, 1000 + row * 50 + x, amp=1.5, edge=(130, 106, 80), edge_w=1)
    fy = 630
    p.paper([(120, fy - 30), (980, fy - 30), (980, fy + 30), (120, fy + 30)], (100, 82, 60), 40, amp=2, edge=(160, 132, 96))
    for i, x in enumerate(range(170, 960, 46)):
        glyph(p, x, fy, 14, i, (34, 26, 18), 3)
    p.paper(arch(245, 345, 880, 780), (22, 16, 12), 50, amp=2, edge=(110, 86, 60))
    pts = []
    for i in range(21):
        f = i / 20; rr = 14 + 18 * math.sin(f * math.pi * 0.95) ** 1.2 - (5 if f > 0.85 else 0)
        pts.append((295 + rr, 876 - f * 80))
    pts += [(590 - x, y) for x, y in reversed(pts)]
    p.paper(pts, (120, 72, 40), 51, amp=1.2, edge=(190, 140, 90))
    for k in range(9):  # iron chain on the right wall
        cy = 470 + k * 22
        p.d.ellipse(p.P([(894 + (k % 2) * 2, cy - 11), (906 + (k % 2) * 2, cy + 11)]), outline=(60, 50, 40, 255), width=p.W(3))
    p.paper([(120, FLOOR_Y), (980, FLOOR_Y), (980, 1180), (120, 1180)], (70, 54, 40), 70, amp=2.5, edge=(130, 100, 70))
    for k in range(-10, 11):
        p.line([(550 + k * 60, FLOOR_Y), (550 + k * 150, 1180)], (54, 42, 30), 2)
    for yy in (930, 975, 1040, 1120):
        p.line([(120, yy), (980, yy)], (54, 42, 30), 2)
    return p.finish(71, shadow=None, tex=0.3)


def coin(p, x, y, s, r):
    p.d.ellipse(p.P([(x - s, y - s * 0.55), (x + s, y + s * 0.55)]), fill=(110, 72, 20, 255))
    p.d.ellipse(p.P([(x - s + 1.3, y - s * 0.55 + 0.6), (x + s - 1.3, y + s * 0.55 - 1.2)]), fill=[(224, 178, 74), (212, 160, 60), (240, 200, 96)][r.integers(3)] + (255,))


def build_chamber_props():
    p = Painter(120, 600, 860, 580)
    r = np.random.default_rng(81)
    for cx, base, wdt, hgt in [(380, 932, 190, 58), (738, 926, 120, 40), (628, 936, 290, 22)]:
        pts = [(cx - wdt / 2 + wdt * i / 30, base - hgt * math.sin(math.pi * i / 30) ** 0.8) for i in range(31)] + [(cx + wdt / 2, base + 6), (cx - wdt / 2, base + 6)]
        p.paper(pts, (184, 134, 46), int(cx), amp=2.5, edge=(240, 200, 110))
        for _ in range(int(wdt * hgt / 55)):
            u = r.uniform(-0.5, 0.5); yy = base - hgt * math.sin(math.pi * (u + 0.5)) ** 0.8 * r.uniform(0.1, 0.95)
            coin(p, cx + u * wdt, yy, r.uniform(7, 10), r)
    for _ in range(40):
        coin(p, r.uniform(290, 800), r.uniform(938, 985), r.uniform(7, 10), r)
    # books + candle (from the banner)
    bx, base = CANDLE[0], 932
    for k, (w, h, c) in enumerate([(92, 22, (92, 42, 30)), (80, 20, (60, 52, 36)), (86, 20, (110, 70, 40))]):
        y1 = base - k * 21; y0 = y1 - h; off = (-4, 5, -2)[k]
        p.paper([(bx - w / 2 + off, y0), (bx + w / 2 + off, y0), (bx + w / 2 + off, y1), (bx - w / 2 + off, y1)], c, 700 + k, amp=0.8, edge=(200, 170, 120))
        p.line([(bx - w / 2 + off + 4, y1 - 5), (bx + w / 2 + off - 4, y1 - 5)], (226, 206, 160), 2)
    cy0 = base - 63
    p.paper([(bx - 12, CANDLE[1] + 8), (bx + 12, CANDLE[1] + 10), (bx + 13, cy0), (bx - 13, cy0)], (226, 208, 166), 710, amp=0.6, edge=(250, 236, 200))
    for dx, ln in ((-9, 16), (7, 26), (-2, 10)):
        p.paper([(bx + dx - 3, CANDLE[1] + 10), (bx + dx + 3, CANDLE[1] + 10), (bx + dx + 2.5, CANDLE[1] + 10 + ln), (bx + dx - 2.5, CANDLE[1] + 10 + ln)], (238, 222, 180), 720 + dx, amp=0.3, edge=None)
    p.line([(bx, CANDLE[1] + 9), (bx + 1, CANDLE[1] + 1)], (30, 20, 10), 2)
    # treasure map + coiled roll
    p.paper([(398, 960), (528, 948), (544, 998), (410, 1012)], (201, 173, 122), 90, amp=2.5, edge=(240, 225, 190))
    route = [(420, 995), (450, 978), (480, 986), (510, 970), (532, 965)]
    for (x0, y0), (x1, y1) in zip(route[:-1], route[1:]):
        for f in (0, 0.25, 0.5, 0.75):
            xx, yy = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
            p.d.ellipse(p.P([(xx - 1.5, yy - 1.5), (xx + 1.5, yy + 1.5)]), fill=(120, 40, 30, 255))
    p.line([(527, 959), (539, 971)], (150, 30, 25), 3); p.line([(539, 959), (527, 971)], (150, 30, 25), 3)
    for k in range(5):
        rr = 15 - k * 2.6
        p.d.arc(p.P([(392 - rr, 986 - rr), (392 + rr, 986 + rr)]), 0, 330, fill=(150, 118, 70, 255), width=p.W(2.5))
    # ring of old keys + pocket watch (from the banner)
    kx, ky = 712, 972
    for k, ang in enumerate((-0.5, -0.1, 0.3, 0.7)):
        ex, ey = kx + 60 * math.cos(ang), ky + 16 + 22 * math.sin(ang)
        p.line([(kx + 14 * math.cos(ang), ky + 8 * math.sin(ang)), (ex, ey)], (140, 108, 60), 3.5)
        p.line([(ex, ey), (ex + 4, ey + 8), (ex - 3, ey + 11)], (140, 108, 60), 3)
    p.d.ellipse(p.P([(kx - 17, ky - 11), (kx + 17, ky + 11)]), outline=(170, 132, 72, 255), width=p.W(3))
    wx, wy = 590, 988
    p.line([(wx + 16, wy - 4), (wx + 40, wy - 16), (wx + 64, wy - 12)], (150, 116, 66), 1.8)
    p.d.ellipse(p.P([(wx - 18, wy - 12), (wx + 18, wy + 12)]), fill=(160, 124, 64, 255))
    p.d.ellipse(p.P([(wx - 14, wy - 9), (wx + 14, wy + 9)]), fill=(222, 206, 170, 255))
    for k in range(12):
        a = k * math.pi / 6
        p.line([(wx + 11 * math.cos(a), wy + 7 * math.sin(a)), (wx + 13 * math.cos(a), wy + 8.3 * math.sin(a))], (60, 44, 30), 1)
    p.line([(wx, wy), (wx + 6, wy - 4)], (40, 30, 20), 1.5); p.line([(wx, wy), (wx - 2, wy + 6)], (40, 30, 20), 1.5)
    return p.finish(82, shadow=(5, 7, 8, 150))


# ------------------------------------------------------------------ dynamic sprites
def chest_sprite(a):
    S = RS
    Wc, Hb, D, Hl = 170, 86, 92, 34
    tau = math.radians(22)
    size = (int(260 * S), int(320 * S))
    ox, oy = size[0] / 2, size[1] - 40 * S
    im = Image.new('RGBA', size, (0, 0, 0, 0)); d = ImageDraw.Draw(im)

    def proj(x, y, z):
        return (ox + x * S, oy - (y * math.cos(tau) - z * math.sin(tau)) * S)

    def depth(x, y, z):
        return z * math.cos(tau) + y * math.sin(tau)

    NORM = {'front': (0, 0, 1), 'back': (0, 0, -1), 'top': (0, 1, 0), 'bottom': (0, -1, 0), 'left': (-1, 0, 0), 'right': (1, 0, 0)}
    faces = []

    def box(xr, yr, zr, cols, xf=None):
        (x0, x1), (y0, y1), (z0, z1) = xr, yr, zr
        V = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
        if xf: V = [xf(*v) for v in V]
        idx = {'front': (1, 3, 7, 5), 'back': (0, 4, 6, 2), 'top': (2, 6, 7, 3), 'bottom': (0, 1, 5, 4), 'left': (0, 2, 3, 1), 'right': (4, 5, 7, 6)}
        for name, q in idx.items():
            if name not in cols: continue
            n = NORM[name]
            if xf: n = np.subtract(xf(*n), xf(0, 0, 0))
            if n[2] * math.cos(tau) + n[1] * math.sin(tau) <= 0: continue
            pts3 = [V[i] for i in q]
            faces.append((np.mean([depth(*v) for v in pts3]), [proj(*v) for v in pts3], cols[name], name))

    wood, wood_d, gold = (96, 58, 32), (66, 40, 22), (206, 162, 74)
    inner = (255, 196, 90) if a > 0.05 else (40, 20, 18)
    box((-Wc / 2, Wc / 2), (0, Hb), (-D / 2, D / 2), {'front': wood, 'top': inner, 'right': wood_d, 'left': wood_d})
    hz, hy = -D / 2, Hb

    def lid_xf(x, y, z):
        y2, z2 = y - hy, z - hz; c, s = math.cos(a), math.sin(a)
        return (x, hy + y2 * c + z2 * s, hz - y2 * s + z2 * c)
    box((-Wc / 2 - 3, Wc / 2 + 3), (Hb, Hb + Hl), (-D / 2 - 3, D / 2 + 3), {'front': wood, 'top': (112, 70, 38), 'bottom': (90, 56, 30), 'back': (80, 50, 28), 'left': wood_d, 'right': wood_d}, lid_xf)
    faces.sort(key=lambda f: f[0])
    for _, pts, col, name in faces:
        d.polygon(pts, fill=col + (255,), outline=(30, 18, 12, 255))
        qa, qb, qc, qd = [np.array(p_) for p_ in pts]
        quad = lambda u, v: tuple(qa + (qb - qa) * u + (qd - qa) * v + (qa - qb + qc - qd) * u * v)
        if name == 'top' and col[0] > 200:      # heaped coins inside the open chest
            cr = np.random.default_rng(5)
            for _ in range(70):
                cx_, cy_ = quad(cr.uniform(0.06, 0.94), cr.uniform(0.1, 0.9)); rr_ = cr.uniform(3.5, 6) * S
                d.ellipse((cx_ - rr_, cy_ - rr_ * 0.6, cx_ + rr_, cy_ + rr_ * 0.6), fill=(150, 100, 30, 255))
                d.ellipse((cx_ - rr_ + 1, cy_ - rr_ * 0.6 + 1, cx_ + rr_ - 1, cy_ + rr_ * 0.6 - 2), fill=(255, 214, 110, 255))
            continue
        if name == 'bottom':                    # velvet-lined inside of the lid
            d.polygon([quad(0.08, 0.1), quad(0.92, 0.1), quad(0.92, 0.9), quad(0.08, 0.9)], fill=(60, 16, 22, 255), outline=gold + (255,), width=int(3 * S))
            continue
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts
        for f in (0.18, 0.82):
            d.line([(x0 + (x3 - x0) * f, y0 + (y3 - y0) * f), (x1 + (x2 - x1) * f, y1 + (y2 - y1) * f)], fill=gold + (255,), width=int(7 * S))
    fx, fy = proj(0, Hb * 0.72, D / 2)
    d.rectangle((fx - 12 * S, fy - 14 * S, fx + 12 * S, fy + 14 * S), fill=gold + (255,), outline=(90, 60, 20, 255))
    d.ellipse((fx - 4 * S, fy - 6 * S, fx + 4 * S, fy + 2 * S), fill=(30, 18, 12, 255))
    return im, (ox / RS, oy / RS)


def explorer_sprite(t, walk_phase, arm_ang, walking):
    S = RS
    size = (int(260 * S), int(300 * S))
    ox, oy = size[0] / 2, size[1] - 20 * S
    im = Image.new('RGBA', size, (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    bob = 3.0 * abs(math.sin(walk_phase)) * walking
    sway = 1.2 * math.sin(walk_phase) * walking
    P = lambda x, y: (ox + (x + sway * (1 + y / 200)) * S, oy + (y - bob * (y < -20)) * S)
    for k, side in enumerate((-1, 1)):
        lift = max(0.0, math.sin(walk_phase + k * math.pi)) * 6 * walking
        fx = side * 16 + math.cos(walk_phase + k * math.pi) * 8 * walking
        d.polygon([P(fx - 12, -lift), P(fx + 16, -lift), P(fx + 14, -14 - lift), P(fx - 10, -16 - lift)], fill=(28, 20, 14, 255))
    hem = [P(-46 + i * 92 / 12, -10 - (6 if i % 2 else 0) - 4 * math.sin(i + walk_phase)) for i in range(13)]
    cloak = [P(-20, -150), P(20, -150), P(40, -60)] + [hem[-1]] + list(reversed(hem)) + [P(-40, -60)]
    d.polygon([(x - 2, y) for x, y in cloak], fill=(190, 158, 112, 255))
    d.polygon(cloak, fill=(58, 46, 34, 255))
    d.line([P(-6, -140), P(-10, -20)], fill=(40, 30, 22, 255), width=2 * S)
    hood = [P(-30, -140), P(-26, -178), P(-6, -204), P(8, -214), P(24, -190), P(30, -150), P(20, -128), P(-22, -128)]
    d.polygon([(x - 2, y - 1) for x, y in hood], fill=(190, 158, 112, 255))
    d.polygon(hood, fill=(66, 52, 38, 255))
    d.ellipse([P(-12, -186), P(16, -148)], fill=(8, 6, 4, 255))
    d.ellipse([P(4, -170), P(8, -166)], fill=(255, 190, 110, 200))
    d.ellipse([P(-5, -170), P(-1, -166)], fill=(255, 190, 110, 150))
    sh = (22, -128); L = 64
    hx, hy = sh[0] + L * math.cos(arm_ang), sh[1] + L * math.sin(arm_ang)
    d.line([P(*sh), P(hx, hy)], fill=(58, 46, 34, 255), width=14 * S)
    d.ellipse([P(hx - 7, hy - 7), P(hx + 7, hy + 7)], fill=(58, 46, 34, 255))
    swing = 0.12 * math.sin(t * 2.1)
    lx, ly = hx + 26 * math.sin(swing), hy + 26 * math.cos(swing)
    d.line([P(hx, hy), P(lx, ly - 10)], fill=(90, 70, 40, 255), width=2 * S)
    d.polygon([P(lx - 11, ly - 10), P(lx + 11, ly - 10), P(lx + 9, ly + 16), P(lx - 9, ly + 16)], fill=(255, 216, 130, 255), outline=(80, 56, 24, 255))
    d.rectangle([P(lx - 13, ly - 14), P(lx + 13, ly - 9)], fill=(120, 86, 36, 255))
    d.rectangle([P(lx - 11, ly + 16), P(lx + 11, ly + 20)], fill=(120, 86, 36, 255))
    d.line([P(lx, ly - 9), P(lx, ly + 16)], fill=(120, 86, 36, 255), width=S)
    return im, (ox / S, oy / S), (lx + sway, ly + 3 - bob)


def flame_sprite(t, seed, scale=1.0):
    S = RS
    w, h = int(80 * S * scale), int(140 * S * scale)
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    cx, by = w / 2, h - 8 * S * scale
    for k, (col, sc) in enumerate([((214, 96, 30), 1.0), ((244, 164, 56), 0.72), ((255, 234, 170), 0.42)]):
        pts = []
        for i in range(29):
            f = i / 28; ang = math.pi * f
            wob = math.sin(t * 7.3 + seed + k + f * 5) * 0.12 + math.sin(t * 11.1 + seed * 2 + f * 9) * 0.06
            rx = 26 * sc * math.sin(ang) ** 0.9 * (1 + wob)
            yy = -f * 110 * sc * (1 + 0.1 * math.sin(t * 5 + seed + k))
            pts.append((cx - rx * S * scale + wob * 20 * S * scale * f, by + yy * S * scale))
        pts += [(cx + (cx - x), y) for x, y in reversed(pts)]
        d.polygon(pts, fill=col + (235,))
    return im


def moth_sprite(flap, rot):
    S = RS * 2
    im = Image.new('RGBA', (40 * S, 40 * S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    c = 20 * S; k = 0.25 + 0.75 * abs(math.cos(flap))
    for sg in (-1, 1):
        fw = [(c, c - 2 * S), (c + sg * 16 * S * k, c - 12 * S), (c + sg * 18 * S * k, c - 2 * S), (c + sg * 6 * S * k, c + 2 * S)]
        hw = [(c, c + S), (c + sg * 12 * S * k, c + 3 * S), (c + sg * 9 * S * k, c + 11 * S), (c + sg * 2 * S * k, c + 6 * S)]
        d.polygon(fw, fill=(186, 156, 110, 255), outline=(90, 70, 46, 255))
        d.polygon(hw, fill=(160, 128, 88, 255), outline=(90, 70, 46, 255))
        d.ellipse((c + sg * 10 * S * k - 2 * S, c - 7 * S, c + sg * 10 * S * k + 2 * S, c - 3 * S), fill=(80, 56, 36, 255))
    d.ellipse((c - 2 * S, c - 7 * S, c + 2 * S, c + 9 * S), fill=(60, 44, 30, 255))
    return im.rotate(rot, resample=Image.BICUBIC)


# ------------------------------------------------------------------ camera
LOGO_Z = LOGO_SIZE / 2000 * LOGO_R / RO
LOGO_SCR = (LOGO_POS[0] + LOGO_DOOR[0] * LOGO_SIZE / 2000, LOGO_POS[1] + LOGO_DOOR[1] * LOGO_SIZE / 2000)
END_C = (DC[0] - (LOGO_SCR[0] - SC[0]) / LOGO_Z, DC[1] - (LOGO_SCR[1] - SC[1]) / LOGO_Z)
KEYS = [  # t, cx, cy, zoom
    (0.0, 540, 792, 1.00),
    (2.2, 540, 796, 1.02),
    (9.2, 556, 842, 2.02),
    (12.6, 566, 846, 2.08),
    (13.5, 566, 846, 2.10),
    (16.3, END_C[0], END_C[1], LOGO_Z),
    (20.0, END_C[0], END_C[1], LOGO_Z),
]


def camera(t):
    for (t0, x0, y0, z0), (t1, x1, y1, z1) in zip(KEYS[:-1], KEYS[1:]):
        if t <= t1:
            f = ss((t - t0) / (t1 - t0))
            return np.array([x0 + (x1 - x0) * f, y0 + (y1 - y0) * f]), math.exp(math.log(z0) + (math.log(z1) - math.log(z0)) * f)
    return np.array(KEYS[-1][1:3], float), KEYS[-1][3]


def shake(t):
    s = 0.0
    for t0, amp in ((1.15, 1.5), (1.35, 1.5), (1.6, 2.0), (15.6, 6.0)):
        if t0 < t < t0 + 0.8:
            k = t - t0; s += amp * math.exp(-k * 8) * math.sin(k * 55)
    if 1.8 < t < 6.0:
        s += 0.7 * math.sin(t * 41) * math.sin(math.pi * seg(t, 1.8, 6.0))
    return s


class Cam:
    def __init__(self, t):
        self.c, self.z = camera(t)
        self.sh = shake(t)

    def of(self, d):
        zd = 1 + (self.z - 1) * d
        return C0 + (self.c - C0) * d + np.array([self.sh * 0.5, self.sh]) / zd, zd

    def to_screen(self, p, d):
        cd, zd = self.of(d)
        return (np.asarray(p, float) - cd) * zd + SC, zd

    def draw_layer(self, frame, L, d):
        img, x0, y0, rs = L
        cd, zd = self.of(d)
        wx0, wy0 = cd[0] - OW / 2 / zd, cd[1] - OH / 2 / zd
        wx1, wy1 = cd[0] + OW / 2 / zd, cd[1] + OH / 2 / zd
        box = ((wx0 - x0) * rs, (wy0 - y0) * rs, (wx1 - x0) * rs, (wy1 - y0) * rs)
        frame.alpha_composite(img.transform((OW, OH), Image.EXTENT, box, Image.BILINEAR))

    def draw_sprite(self, frame, spr, anchor, world_pos, d, rs=RS):
        s, zd = self.to_screen(world_pos, d)
        sc = zd / rs
        im = spr.resize((max(1, int(spr.width * sc)), max(1, int(spr.height * sc))), Image.LANCZOS)
        paste(frame, im, s[0] - anchor[0] * zd, s[1] - anchor[1] * zd)


def paste(canvas, sprite, x0, y0):
    x0, y0 = int(round(x0)), int(round(y0))
    sw, sh = sprite.size; cw, ch = canvas.size
    l, t = max(0, -x0), max(0, -y0); r, b = min(sw, cw - x0), min(sh, ch - y0)
    if r > l and b > t:
        canvas.alpha_composite(sprite.crop((l, t, r, b)), dest=(x0 + l, y0 + t))


# ------------------------------------------------------------------ timeline
OPEN_MAX = math.radians(76)


def door_angle(t):
    if t < 13.8:
        return OPEN_MAX * ss(seg(t, 1.8, 6.0))
    k = min(1.0, max(0.0, seg(t, 13.8, 15.6)))
    return OPEN_MAX * (1 - (0.4 * ss(k) + 0.6 * ease_in(k)))   # heavy: accelerates into the slam


def openness(t):
    return math.sin(door_angle(t)) / math.sin(OPEN_MAX)


def explorer_state(t):
    walk = ss(seg(t, 7.2, 9.9))
    x = 170 + (420 - 170) * walk
    walking = min(ss(seg(t, 7.2, 7.7)), 1 - ss(seg(t, 9.4, 9.9)))
    phase = (t - 7.2) * 2 * math.pi * 0.85
    low, up = math.radians(80), math.radians(-58)
    raise_ = ss(seg(t, 10.0, 10.9))
    k = min(1.0, max(0.0, seg(t, 10.9, 12.6)))
    arm = low + (up - low) * raise_ + math.radians(-24) * math.sin(1.5 * math.pi * ss(k)) * raise_
    return x, phase, arm, walking, raise_


def lantern_intensity(t):
    return (0.25 + 0.75 * ss(seg(t, 9.9, 10.9))) * (1 + 0.05 * math.sin(t * 9.1) + 0.03 * math.sin(t * 23.7))


def lid_angle(t):
    k = seg(t, 12.7, 13.25)
    creak = 0.06 * ss(seg(t, 12.45, 12.7))
    return creak + math.radians(100) * ease_out(k) if k > 0 else creak


def keyhole_glow(t):
    g = 0.08 * math.exp(-((t - 1.2) / 0.25) ** 2)          # glint as the bolts turn
    g += 0.85 * ss(seg(t, 15.5, 16.3))
    g += 1.1 * math.exp(-((t - 18.55) / 0.12) ** 2) + 0.65 * math.exp(-((t - 18.85) / 0.12) ** 2)
    return g


# ------------------------------------------------------------------ assets
A = {}


def build_assets():
    A['bg'] = build_bg(); A['wall'] = build_wall(); A['mound'] = build_mound()
    A['hinge'] = build_hinge(); A['leaf'] = build_leaf()
    A['cback'] = build_chamber_back(); A['cprops'] = build_chamber_props()
    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    e = np.sqrt(((xx - OW / 2) / (OW * 0.55)) ** 2 + ((yy - OH / 2) / (OH * 0.62)) ** 2)
    e += (fbm(OH, OW, 60, 7, 4) - 0.5) * 0.35                # distressed, burnt-looking edges
    A['vig'] = np.clip(1 - 0.85 * np.clip((e - 0.72) / 0.5, 0, 1) ** 1.4, 0.05, 1)[..., None].astype(np.float32)
    A['grain'] = [np.asarray(Image.fromarray(np.random.default_rng(900 + i).normal(0, 7, (OH // 2, OW // 2)).astype(np.float32), 'F').resize((OW, OH), Image.BILINEAR))[..., None] for i in range(6)]
    A['lw'], A['lh'] = OW // 4, OH // 4
    r = np.random.default_rng(77)
    A['motes'] = r.random((90, 4))
    A['coinglints'] = [(r.uniform(300, 800), r.uniform(880, 985), r.uniform(0, 6.28)) for _ in range(26)]
    # the channel logo, plus a mask of its keyhole for the heartbeat glow
    logo = Image.open(os.path.join(HERE, 'assets', 'logo.jpg')).convert('RGB')
    lum = np.asarray(logo.convert('L'))
    dark = Image.fromarray(((lum < 34) * 255).astype(np.uint8))
    ImageDraw.floodfill(dark, (1005, 700), 128)
    km = (np.asarray(dark) == 128).astype(np.float32)
    km[:, :850] = 0; km[:, 1170:] = 0; km[:540] = 0; km[940:] = 0
    s = LOGO_SIZE
    A['logo'] = np.asarray(logo.resize((s, s), Image.LANCZOS), np.float32)
    kms = Image.fromarray((km * 255).astype(np.uint8)).resize((s, s), Image.BILINEAR)
    A['logo_key'] = np.asarray(kms, np.float32)[..., None] / 255
    A['logo_bloom'] = np.asarray(kms.filter(ImageFilter.GaussianBlur(26)), np.float32)[..., None] / 255
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    edge = np.minimum.reduce([xx, yy, s - 1 - xx, s - 1 - yy])
    A['logo_feather'] = np.clip(edge / 60, 0, 1)[..., None]
    kyy = (yy - (LOGO_DOOR[1] - 0.2 * LOGO_R) * s / 2000) / (0.45 * LOGO_R * s / 2000)
    A['logo_keygrad'] = np.clip(1 - kyy, 0.35, 1)[..., None]


# ------------------------------------------------------------------ frame render
def radial(lw, lh, cx, cy, rad, sx=1.0, sy=1.0):
    yy, xx = np.mgrid[0:lh, 0:lw].astype(np.float32)
    d2 = ((xx - cx) / (rad * sx)) ** 2 + ((yy - cy) / (rad * sy)) ** 2
    return 1 / (1 + d2 * 3.0) * np.clip(1.4 - d2 * 0.35, 0, 1)


def draw_leaf(frame, cam, t):
    """door leaf swinging toward camera on its left hinge."""
    th = door_angle(t)
    img, x0, y0, rs = A['leaf']
    s, zd = cam.to_screen((HINGE_X, DC[1]), 1.0)
    c = math.cos(th); sc = zd / rs
    face = img.resize((max(2, int(img.width * c * sc)), max(2, int(img.height * sc))), Image.LANCZOS)
    shade = 1 - 0.5 * math.sin(th)
    if shade < 0.999:
        fa = np.asarray(face).copy(); fa[..., :3] = (fa[..., :3] * shade).astype(np.uint8); face = Image.fromarray(fa, 'RGBA')
    ax = (HINGE_X - DC[0] - x0) * c * zd
    ay = (0 - y0) * zd
    band = int(34 * math.sin(th) * zd)       # door thickness showing on the swinging edge
    if band > 0:
        edge = Image.new('RGBA', face.size, (62, 44, 24, 0)); edge.putalpha(face.getchannel('A'))
        for k in range(band, 0, -max(1, band // 6)):
            paste(frame, edge, s[0] - ax + k, s[1] - ay)
    paste(frame, face, s[0] - ax, s[1] - ay)
    return s, zd, th


def render_frame(fi):
    t = fi / FPS
    cam = Cam(t)
    frame = Image.new('RGBA', (OW, OH), (0, 0, 0, 255))
    op = openness(t)
    cam.draw_layer(frame, A['bg'], DEPTH['bg'])
    ex, ph, arm, walking, raised = explorer_state(t)
    lantern_s = lantern_w = None
    if op > 0.001:
        cam.draw_layer(frame, A['cback'], DEPTH['cback'])
        chest, anc = chest_sprite(lid_angle(t))
        cam.draw_sprite(frame, chest, anc, CHEST, DEPTH['cprops'])
        cam.draw_layer(frame, A['cprops'], DEPTH['cprops'])
        if t > 7.0:
            spr, anc, lan = explorer_sprite(t, ph, arm, walking)
            spr = spr.resize((int(spr.width * EXS), int(spr.height * EXS)), Image.LANCZOS)
            feet = (ex, 950)
            cam.draw_sprite(frame, spr, (anc[0] * EXS, anc[1] * EXS), feet, DEPTH['explorer'])
            lantern_w = (feet[0] + lan[0] * EXS, feet[1] + lan[1] * EXS)
            lantern_s, _ = cam.to_screen(lantern_w, DEPTH['explorer'])
        for k in range(3):  # moths circling the candle and lantern
            tgt = CANDLE if (k == 0 or lantern_w is None) else lantern_w
            a = t * (1.3 + 0.4 * k) + k * 2.1
            mx = tgt[0] + 46 * math.cos(a) + 10 * math.sin(t * 3.7 + k)
            my = tgt[1] - 20 + 26 * math.sin(a * 1.3) + 8 * math.cos(t * 4.1 + k)
            m = moth_sprite(t * 22 + k, math.degrees(-a) * 0.3 + 20 * math.sin(t * 2 + k))
            cam.draw_sprite(frame, m, (m.width / (RS * 2) / 2, m.height / (RS * 2) / 2), (mx, my), DEPTH['cprops'], rs=RS * 2)
    cam.draw_layer(frame, A['wall'], DEPTH['wall'])
    ks, kzd, th = draw_leaf(frame, cam, t)
    hl = A['hinge']
    cam.draw_sprite(frame, hl[0], (HINGE_X - hl[1], DC[1] - hl[2]), (HINGE_X, DC[1]), 1.0)
    cam.draw_layer(frame, A['mound'], DEPTH['mound'])

    # -------- lighting (quarter-res light map)
    lw, lh = A['lw'], A['lh']; q = 4.0
    light = np.broadcast_to(np.array([0.21, 0.16, 0.12], np.float32), (lh, lw, 3)).copy()
    for i, (tx, ty) in enumerate(TORCHES):
        s, zd = cam.to_screen((tx, ty - 30), 1.0)
        fl = 1 + 0.12 * math.sin(t * 8.3 + i * 2) + 0.07 * math.sin(t * 17.9 + i) + 0.05 * math.sin(t * 3.1 + i * 5)
        light += radial(lw, lh, s[0] / q, s[1] / q, 330 * zd / q)[..., None] * np.array([1.05, 0.62, 0.3]) * 1.1 * fl
    s, zd = cam.to_screen((560, 760), DEPTH['cprops'])
    light += radial(lw, lh, s[0] / q, s[1] / q, 300 * zd / q, 1.1, 1.0)[..., None] * np.array([1.0, 0.72, 0.36]) * 0.8 * op
    s, zd = cam.to_screen(CANDLE, DEPTH['cprops'])
    light += radial(lw, lh, s[0] / q, s[1] / q, 170 * zd / q)[..., None] * np.array([1.0, 0.7, 0.35]) * 0.7 * op * (1 + 0.08 * math.sin(t * 13))
    burst = ss(seg(t, 12.75, 13.3))
    if burst > 0 and op > 0:
        s, zd = cam.to_screen((CHEST[0], CHEST[1] - 110), DEPTH['cprops'])
        light += radial(lw, lh, s[0] / q, s[1] / q, 200 * zd / q)[..., None] * np.array([1.1, 0.85, 0.45]) * 0.9 * burst * (1 + 0.8 * math.exp(-max(0, t - 13.1) * 3)) * op
    kg = keyhole_glow(t)
    if kg > 0.01:
        s, zd = cam.to_screen((DC[0], DC[1] - 0.05 * RL), 1.0)
        light += radial(lw, lh, s[0] / q, s[1] / q, 90 * zd / q)[..., None] * np.array([1.0, 0.72, 0.36]) * 0.6 * kg
    if lantern_s is not None:
        li = lantern_intensity(t); zd = cam.of(DEPTH['explorer'])[1]
        lx, ly = lantern_s / q
        ll = radial(lw, lh, lx, ly, 200 * zd / q) * 1.4
        tgt = (lx + math.cos(arm + math.radians(58)) * 150 * zd / q * raised + 60 * zd / q * raised, ly + 70 * zd / q * raised)
        ll = ll + radial(lw, lh, tgt[0], tgt[1], 190 * zd / q, 1.2, 0.8) * 0.8 * raised
        smask = Image.new('L', (lw, lh), 0); sd = ImageDraw.Draw(smask)
        Lw = np.array(lantern_w)
        for (ox, base), half, hgt in [((CHEST[0], CHEST[1] - 6), 90, 150 + 60 * (lid_angle(t) > 0.5)), ((380, 932), 95, 58), ((738, 926), 60, 40)]:
            Lh = max(40.0, base - Lw[1]); poly = []
            for (px, ph_) in [(ox - half, 0), (ox - half * 0.8, hgt * 0.8), (ox, hgt), (ox + half * 0.8, hgt * 0.8), (ox + half, 0)]:
                poly.append((px + ph_ * (px - Lw[0]) / Lh * 2.1, base - ph_ * 0.22 - 4))
            pts = [cam.to_screen(p_, DEPTH['cprops'])[0] / q for p_ in poly + [(ox + half, base), (ox - half, base)]]
            sd.polygon([tuple(p_) for p_ in pts], fill=200)
        sm = np.asarray(smask.filter(ImageFilter.GaussianBlur(3)), np.float32) / 255 * raised
        light += (ll * (1 - 0.85 * sm))[..., None] * np.array([1.05, 0.78, 0.45]) * li * 1.2 * op
    light = np.asarray(Image.fromarray(np.clip(light * 80, 0, 255).astype(np.uint8)).resize((OW, OH), Image.BILINEAR), np.float32) / 80
    rgb = np.asarray(frame.convert('RGB'), np.float32) * light

    # -------- emissive
    em = Image.new('RGBA', (OW, OH), (0, 0, 0, 0)); dd = ImageDraw.Draw(em)
    for i, (tx, ty) in enumerate(TORCHES):
        f = flame_sprite(t, i * 3.7)
        cam.draw_sprite(em, f, (f.width / RS / 2, f.height / RS - 8), (tx, ty), 1.0)
    if op > 0.001:
        f = flame_sprite(t * 1.3, 9.1, 0.28)
        cam.draw_sprite(em, f, (f.width / RS / 2, f.height / RS - 8 * 0.28), CANDLE, DEPTH['cprops'])
    if lantern_s is not None:
        zd = cam.of(DEPTH['explorer'])[1]; rr = int(30 * zd)
        paste(em, glow_blob(rr, (255, 200, 110), 0.55), lantern_s[0] - 2 * rr, lantern_s[1] - 2 * rr)
    if burst > 0 and op > 0.001:
        s, zd = cam.to_screen((CHEST[0], CHEST[1] - 90), DEPTH['cprops'])
        sw_, sh_ = int(260 * zd), int(420 * zd)
        yy, xx = np.mgrid[0:sh_, 0:sw_].astype(np.float32)
        yf = yy / sh_; half = 0.18 + 0.32 * (1 - yf)
        sa = np.clip(1 - np.abs(xx / sw_ - 0.5) / half * 2, 0, 1) ** 1.5 * yf ** 1.2
        sarr = np.dstack([np.full_like(sa, 255), np.full_like(sa, 205), np.full_like(sa, 110), sa * 150 * burst * op]).astype(np.uint8)
        paste(em, Image.fromarray(sarr, 'RGBA'), s[0] - sw_ / 2, s[1] - sh_)
        r = np.random.default_rng(4)
        for k in range(40):
            born = 12.8 + r.uniform(0, 3); age = t - born
            if age < 0 or age > 2.5: continue
            sp, zd2 = cam.to_screen((CHEST[0] + r.uniform(-70, 70) + 12 * math.sin(age * 2 + k), CHEST[1] - 95 - age * r.uniform(40, 80)), DEPTH['cprops'])
            al = int(255 * math.sin(math.pi * age / 2.5) * op); rr = 2.2 * zd2
            dd.line((sp[0] - rr * 2.5, sp[1], sp[0] + rr * 2.5, sp[1]), fill=(255, 230, 160, al // 2))
            dd.line((sp[0], sp[1] - rr * 2.5, sp[0], sp[1] + rr * 2.5), fill=(255, 230, 160, al // 2))
            dd.ellipse((sp[0] - rr, sp[1] - rr, sp[0] + rr, sp[1] + rr), fill=(255, 236, 180, al))
    if lantern_s is not None and raised > 0:
        for (gx, gy, ph) in A['coinglints']:
            v = max(0.0, math.sin(t * 3 + ph)) ** 8 * raised * op
            if v < 0.05: continue
            sp, zd2 = cam.to_screen((gx, gy), DEPTH['cprops']); rr = 5 * zd2 * v
            dd.line((sp[0] - rr, sp[1], sp[0] + rr, sp[1]), fill=(255, 240, 190, int(230 * v)), width=2)
            dd.line((sp[0], sp[1] - rr, sp[0], sp[1] + rr), fill=(255, 240, 190, int(230 * v)), width=2)
    if kg > 0.01 and th < 0.05:   # light leaking through the closed keyhole
        kp = [tuple(cam.to_screen((DC[0] + x, DC[1] + y), 1.0)[0]) for x, y in keyhole_poly(0, 0, RL)]
        dd.polygon(kp, fill=(255, 196, 96, int(255 * min(1, kg))))
        kc, kzd2 = cam.to_screen((DC[0], DC[1] - 0.05 * RL), 1.0)
        rr = int(46 * kzd2)
        paste(em, glow_blob(rr, (255, 186, 90), round(min(1.0, 0.5 * kg), 2)), kc[0] - 2 * rr, kc[1] - 2 * rr)
    dust(em, cam, t)
    ea = np.asarray(em, np.float32); al = ea[..., 3:4] / 255
    rgb = rgb * (1 - al) + ea[..., :3] * al

    # -------- grade: sepia shadows, warm highlights, distressed vignette
    lum = rgb.mean(axis=2, keepdims=True) / 255
    rgb = rgb * (1 - lum) * np.array([0.98, 0.9, 0.78], np.float32) + rgb * lum * np.array([1.06, 0.98, 0.84], np.float32)
    rgb *= A['vig']
    k = ss(seg(t, 15.3, 16.5))   # isolate the door against black, like the logo
    if k > 0:
        dsc, dz = cam.to_screen(DC, 1.0)
        yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
        r_ = np.sqrt(((xx - dsc[0]) / (1.45 * RO * dz)) ** 2 + ((yy - dsc[1] - 0.25 * RO * dz) / (1.12 * RO * dz)) ** 2)
        keep = np.clip(1.25 - r_, 0, 1)[..., None]
        rgb = rgb * (1 - k * (1 - keep) * 0.95)
    a = ss(seg(t, 16.1, 17.3))   # dissolve into the channel logo
    if a > 0:
        x0, y0 = LOGO_POS; s = LOGO_SIZE
        lg = A['logo'] + A['logo_key'] * A['logo_keygrad'] * np.array([255, 190, 96], np.float32) * min(1.0, kg) + A['logo_bloom'] * np.array([255, 170, 70], np.float32) * 0.5 * kg
        lg = np.clip(lg, 0, 255)
        out = rgb * (1 - a) + np.array([12, 11, 10], np.float32) * a
        fa = A['logo_feather'] * a
        out[y0:y0 + s, x0:x0 + s] = out[y0:y0 + s, x0:x0 + s] * (1 - fa) + lg * fa
        rgb = out
    if t > 16.3:   # slow settle-in on the logo
        zz = 1 + 0.03 * ss(seg(t, 16.3, 20.0))
        im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        cw, ch = OW / zz, OH / zz
        im = im.transform((OW, OH), Image.EXTENT, ((OW - cw) / 2, (OH - ch) / 2, (OW + cw) / 2, (OH + ch) / 2), Image.BICUBIC)
        rgb = np.asarray(im, np.float32)
    rgb = rgb + A['grain'][fi % 6]
    fade = min(1.0, ss(t / 1.0), 1 - ss(seg(t, 19.1, 19.95)))
    return np.clip(rgb * fade, 0, 255).astype(np.uint8).tobytes()


_blobs = {}


def glow_blob(r, col, amax):
    key = (r, col, amax)
    if key not in _blobs:
        s = max(4, r * 4)
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        d = np.sqrt((xx - s / 2) ** 2 + (yy - s / 2) ** 2) / max(1, r)
        a = np.exp(-d * d * 0.6) * amax
        _blobs[key] = Image.fromarray(np.dstack([np.full_like(a, col[0]), np.full_like(a, col[1]), np.full_like(a, col[2]), a * 255]).astype(np.uint8), 'RGBA')
    return _blobs[key]


def dust(em, cam, t):
    dd = ImageDraw.Draw(em)
    r = np.random.default_rng(12)
    for k in range(300):  # soil sifting off the frame while the door moves
        born = r.uniform(1.3, 6.4) if k < 200 else r.uniform(13.9, 16.4)
        x0 = DC[0] + r.uniform(-RO, RO) * 0.9
        age = t - born
        if age < 0 or age > 2.2: continue
        y = DC[1] - math.sqrt(max(0, RO * RO - (x0 - DC[0]) ** 2)) + 0.5 * 320 * age * age
        sp, zd = cam.to_screen((x0 + 6 * math.sin(age * 3 + k), y), 1.0)
        a = int(190 * min(1, (2.2 - age) / 0.6)); rr = r.uniform(0.8, 2.2) * zd
        dd.ellipse((sp[0] - rr, sp[1] - rr, sp[0] + rr, sp[1] + rr), fill=(180, 140, 96, a))
    for k in range(120):  # puff when the door slams shut
        age = t - 15.6
        if age < 0 or age > 2.4: break
        ang = r.uniform(0, 2 * math.pi); rad = RO * r.uniform(0.95, 1.05)
        x, y = DC[0] + rad * math.cos(ang), DC[1] + rad * math.sin(ang)
        v = r.uniform(30, 90)
        sp, zd = cam.to_screen((x + math.cos(ang) * v * age, y + math.sin(ang) * v * age * 0.5 + 50 * age * age), 1.0)
        a = int(170 * max(0, 1 - age / 2.4)); rr = r.uniform(1, 3) * zd
        dd.ellipse((sp[0] - rr, sp[1] - rr, sp[0] + rr, sp[1] + rr), fill=(170, 136, 100, a))
    for i, (mx, my, ms, mp) in enumerate(A['motes']):  # motes drifting in the warm light
        x = -300 + mx * 1700 + 30 * math.sin(t * 0.3 + mp * 6)
        y = 300 + ((my * 900 - t * (8 + ms * 12)) % 900)
        sp, zd = cam.to_screen((x, y), 0.9)
        a = int((50 + 110 * ms) * (0.5 + 0.5 * math.sin(t * 1.3 + mp * 9)))
        rr = (0.8 + ms * 1.6) * zd
        dd.ellipse((sp[0] - rr, sp[1] - rr, sp[0] + rr, sp[1] + rr), fill=(255, 214, 150, a))


# ------------------------------------------------------------------ audio
def audio():
    from scipy.signal import butter, sosfilt, fftconvolve
    n = int(SR * DUR); t = np.arange(n) / SR
    r = np.random.default_rng(3)
    bp = lambda x, lo, hi, o=2: sosfilt(butter(o, [lo, hi], 'bandpass', fs=SR, output='sos'), x)
    lp = lambda x, f, o=2: sosfilt(butter(o, f, 'lowpass', fs=SR, output='sos'), x)
    env = lambda a, b, att=0.05, rel=0.3: np.clip((t - a) / att, 0, 1) * np.clip((b - t) / rel, 0, 1)
    dry = np.zeros((n, 2)); send = np.zeros(n)

    def add(t0, sig, pan=(1, 1), wet=0.3):
        i0 = int(t0 * SR); L = min(len(sig), n - i0)
        dry[i0:i0 + L] += sig[:L, None] * np.array(pan); send[i0:i0 + L] += sig[:L] * wet

    # cave room tone + torch crackle
    wind = lp(np.cumsum(r.normal(0, 1, n)) * 0.002, 380); wind -= lp(wind, 20)
    dry += (wind * (0.6 + 0.4 * np.sin(2 * np.pi * t / 7.3) ** 2) * 0.8)[:, None] * np.array([1, 0.85])
    cr = np.zeros(n); idx = r.choice(n, 280, replace=False); cr[idx] = r.uniform(-1, 1, 280)
    dry += (bp(cr, 1500, 6000) * 0.45)[:, None] * np.array([0.8, 1.0])
    # bolts unlocking
    for k, ts in enumerate((1.15, 1.35, 1.6)):
        L = int(0.7 * SR); tt = np.arange(L) / SR
        clank = bp(r.normal(0, 1, L), 250, 2200) * np.exp(-tt * 30) * 0.9 + (np.sin(2 * np.pi * (380 + 60 * k) * tt) + 0.5 * np.sin(2 * np.pi * (913 + 40 * k) * tt)) * np.exp(-tt * 7) * 0.25
        add(ts, clank, (0.8, 1.0), 0.5)

    def groan(dur, f0, f1):  # heavy hinge + grinding soil
        L = int(dur * SR); tt = np.arange(L) / SR; u = tt / tt[-1]
        f = f0 + (f1 - f0) * u + 6 * np.sin(2 * np.pi * 2.3 * tt)
        ph = 2 * np.pi * np.cumsum(f) / SR
        saw = 2 * ((ph / (2 * np.pi)) % 1) - 1
        stick = 0.5 + 0.5 * (np.sin(2 * np.pi * 17 * tt + 2 * np.sin(2 * np.pi * 3 * tt)) > -0.2)
        g = bp(saw * stick, 70, 900, 3) * 0.5 + bp(r.normal(0, 1, L), 50, 380, 3) * 1.4 * (0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 4.7 * tt)))
        return g * np.sin(np.pi * u) ** 0.5
    add(1.8, groan(4.2, 62, 48), (1.0, 0.8), 0.6)
    add(13.8, groan(1.8, 52, 70), (1.0, 0.8), 0.6)
    tr = bp(r.normal(0, 1, n), 3000, 9000) * (0.5 + 0.5 * (r.random(n) > 0.97))
    dry += (tr * 0.12 * (env(1.4, 6.8, 0.4, 1.0) + env(13.9, 17.2, 0.3, 1.5)))[:, None] * np.array([1, 0.8])
    for k, ts in enumerate(np.arange(7.45, 9.9, 1 / 1.7)):  # footsteps
        L = int(0.2 * SR)
        add(ts, lp(r.normal(0, 1, L), 600) * np.exp(-np.arange(L) / (0.045 * SR)) * 0.9, (0.9, 0.6) if k % 2 else (0.7, 0.8), 0.25)
    L = int(0.6 * SR); tt = np.arange(L) / SR
    add(10.4, (np.sin(2 * np.pi * 2350 * tt) + 0.6 * np.sin(2 * np.pi * 3710 * tt)) * np.exp(-tt * 9) * 0.05)
    # chest creak + magical chime
    L = int(0.85 * SR); tt = np.arange(L) / SR
    ph = 2 * np.pi * np.cumsum(140 + 90 * tt + 40 * np.sin(2 * np.pi * 3 * tt)) / SR
    saw = 2 * ((ph / (2 * np.pi)) % 1) - 1
    add(12.45, bp(saw * (np.sin(2 * np.pi * 26 * tt) > 0.2), 300, 2400) * np.sin(np.pi * tt / tt[-1]) * 0.45)
    L = int(4.5 * SR); tt = np.arange(L) / SR; chime = np.zeros(L)
    for k, (fr, amp, dec) in enumerate([(1318.5, 1, 1.4), (1975.5, 0.7, 1.1), (2637, 0.5, 0.9), (3520, 0.3, 0.7), (1567.98, 0.5, 1.6)]):
        st = int(k * 0.06 * SR)
        chime[st:] += np.sin(2 * np.pi * fr * tt[:L - st] * (1 + 0.0008 * np.sin(2 * np.pi * 5 * tt[:L - st]))) * np.exp(-tt[:L - st] * dec) * amp
    add(12.95, chime * 0.12 + bp(r.normal(0, 1, L), 6000, 12000) * np.exp(-tt * 1.2) * 0.05, (0.9, 1.0), 0.5)
    # door slam
    L = int(2.5 * SR); tt = np.arange(L) / SR
    add(15.6, np.sin(2 * np.pi * (60 * np.exp(-tt * 3) + 36) * tt) * np.exp(-tt * 3.2) + lp(r.normal(0, 1, L), 1200) * np.exp(-tt * 12) * 0.7
        + (np.sin(2 * np.pi * 210 * tt) + 0.4 * np.sin(2 * np.pi * 517 * tt)) * np.exp(-tt * 5) * 0.12, (1, 1), 0.8)
    # logo reveal: low bell + airy swell
    L = int(4 * SR); tt = np.arange(L) / SR
    bell = sum(np.sin(2 * np.pi * f * tt) * np.exp(-tt * d_) * a_ for f, a_, d_ in ((440, 1, 0.9), (659.3, 0.6, 1.1), (880.5, 0.35, 1.5), (1318.5, 0.2, 2)))
    add(16.2, bell * 0.08 + bp(r.normal(0, 1, L), 2500, 7000) * np.clip(tt / 0.8, 0, 1) * np.exp(-tt * 1.3) * 0.05, (1, 1), 0.7)
    # heartbeat with the keyhole pulse
    for ts, a_ in ((18.55, 0.85), (18.85, 0.6)):
        L = int(0.5 * SR); tt = np.arange(L) / SR
        add(ts, np.sin(2 * np.pi * (55 * np.exp(-tt * 6) + 42) * tt) * np.exp(-tt * 12) * a_, (1, 1), 0.3)
    # vault reverb
    L = int(1.6 * SR); tt = np.arange(L) / SR
    ir = lp(r.normal(0, 1, L) * np.exp(-tt * 3.2), 3000); ir /= np.sqrt((ir ** 2).sum()) * 6
    wet = fftconvolve(send, ir)[:n]
    out = dry + np.stack([wet, np.roll(wet, 331)], 1)
    out *= (np.clip(t / 1.0, 0, 1) * np.clip((DUR - t) / 1.0, 0, 1))[:, None]
    out = out / (np.abs(out).max() + 1e-9) * 0.85
    return (out * 32767).astype(np.int16)


def main():
    out = next((a for a in sys.argv[1:] if not a.startswith('--')), 'the_buried_vault.mp4')
    print('building assets...', flush=True)
    build_assets()
    if '--preview' in sys.argv:
        os.makedirs('preview', exist_ok=True)
        times = [0.8, 1.5, 3.5, 6.5, 8.5, 10.5, 11.8, 13.2, 14.5, 15.4, 16.0, 16.6, 17.2, 18.0, 18.6, 19.5]
        with Pool(4) as pool:
            res = pool.map(render_frame, [int(tt * FPS) for tt in times])
        sheet = Image.new('RGB', (480 * 4, 270 * 4))
        for i, b in enumerate(res):
            sheet.paste(Image.frombytes('RGB', (OW, OH), b).resize((480, 270)), ((i % 4) * 480, (i // 4) * 270))
        sheet.save('preview/sheet.png')
        for tt, b in zip(times, res):
            Image.frombytes('RGB', (OW, OH), b).save(f'preview/t{tt:05.2f}.png')
        return
    wav = 'audio.wav'
    with wave.open(wav, 'wb') as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes(audio().tobytes())
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-',
                             '-i', wav, '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
                             '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for i, b in enumerate(pool.imap(render_frame, range(NF), chunksize=4)):
            proc.stdin.write(b)
            if i % 48 == 0: print(f'frame {i}/{NF}', flush=True)
    proc.stdin.close(); proc.wait(); os.remove(wav)
    print('wrote', out)


if __name__ == '__main__':
    main()
