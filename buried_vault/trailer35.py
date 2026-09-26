"""The Buried Vault — 35 s channel-page trailer (1920x1080), paper-cut diorama in true perspective.

One continuous dolly: vault door opens -> corridor of four glowing alcoves (Lost Treasure, Ancient Ruins,
Unsolved Mysteries, Legendary Artifacts), each with a carved label -> central chamber where the channel
wordmark (lettering taken from assets/logo.jpg) carves into the wall above an open chest, followed by
"New episodes every week" and a SUBSCRIBE sigil.   Usage: python trailer35.py [out.mp4] [--preview]
"""
import math, os, subprocess, sys, wave
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.interpolate import PchipInterpolator
import imageio_ffmpeg

import render as base
from render import (Painter, torn, circ, arch, fbm, ss, seg, ease_out, glyph, flame_sprite, paste, radial,
                    glow_blob, keyhole_poly, spiral_pts, coin, chest_sprite, BRONZE, BRONZE_D, BRONZE_L, GOLD)

HERE = os.path.dirname(os.path.abspath(__file__))
OW, OH = 1920, 1080
FPS, DUR = 24, 35.0
NF = int(FPS * DUR)
SR = 48000
F = 1000.0                      # focal length: a card at distance F is drawn 1 world unit = 1 px
NEAR = 45.0
SX, SY = OW / 2, OH / 2
FONT = os.path.join(HERE, 'assets', 'fonts', 'Cinzel.ttf')

# ------------------------------------------------------------------ layout (x right, y down, z forward)
RO, RI = 380.0, 327.0                       # entrance door frame radii
HX = base.HINGE_X - base.DC[0]              # hinge x relative to door centre (-357)
Z_RIBS = [1300, 2700, 4100, 5500]
SIDES = [1, -1, 1, -1]                      # which side of the corridor each alcove is on
NX = 650                                    # alcove centre |x|
OPEN_HW, FLOOR, SPRING = 380, 380, -120     # corridor arch opening
Z_ENT, Z_PILLAR, Z_CHEST, Z_BACK = 6600, 7500, 7900, 8100
HOLD = [(9.4, 11.0), (12.6, 14.2), (15.8, 17.4), (19.0, 20.6)]
LABELS = ['LOST TREASURE', 'ANCIENT RUINS', 'UNSOLVED MYSTERIES', 'LEGENDARY ARTIFACTS']
GLOW = [(1.0, 0.8, 0.42), (1.0, 0.62, 0.34), (1.0, 0.42, 0.3), (0.95, 0.9, 0.7)]
T_WORD, T_TAG, T_SIGIL, T_PULSE = (25.3, 27.4), (27.7, 28.5), (28.7, 29.4), 30.6

# ------------------------------------------------------------------ camera (monotone dolly + smooth lateral path)
_T = [0, 2.6, 4.8, 6.6, 7.8, 8.8, 9.4, 11.0, 12.6, 14.2, 15.8, 17.4, 19.0, 20.6, 23.2, 25.0, 26.2, 35.0]
_Z = [-760, -735, -640, -300, 0, 300, 400, 490, 1800, 1890, 3200, 3290, 4600, 4690, 6560, 7040, 7100, 7130]
CZ = PchipInterpolator(_T, _Z)
_PZ = [-760, 0, 400, 490, 1300, 1800, 1890, 2700, 3200, 3290, 4100, 4600, 4690, 5500, 7200]
_PX = [0, 0, 400, 380, 0, -400, -380, 0, 400, 380, 0, -400, -380, 0, 0]
_PY = [0, -10, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -20, 0]
CXZ, CYZ = PchipInterpolator(_PZ, _PX), PchipInterpolator(_PZ, _PY)


def camera(t):
    z = float(CZ(min(max(t, 0), DUR)))
    return float(CXZ(z)), float(CYZ(z)), z


class Cam:
    def __init__(self, t):
        self.x, self.y, self.z = camera(t)

    def proj(self, p):
        dz = p[2] - self.z
        if dz < NEAR: return None, 0
        s = F / dz
        return (SX + (p[0] - self.x) * s, SY + (p[1] - self.y) * s), s

    def fog(self, z):
        return math.exp(-max(0.0, z - self.z - 1100) / 2600)

    def card(self, frame, L, z, fog=True, alpha=1.0, erase=None):
        """draw a flat card (img, world x0, y0, rs) facing the camera at depth z."""
        img, x0, y0, rs = L
        dz = z - self.z
        if dz < NEAR: return
        s = F / dz
        sx0, sy0 = SX + (x0 - self.x) * s, SY + (y0 - self.y) * s
        sx1, sy1 = sx0 + img.width / rs * s, sy0 + img.height / rs * s
        cx0, cy0, cx1, cy1 = max(0, math.floor(sx0)), max(0, math.floor(sy0)), min(OW, math.ceil(sx1)), min(OH, math.ceil(sy1))
        if cx1 - cx0 < 1 or cy1 - cy0 < 1: return
        k = rs / s
        box = ((cx0 - sx0) * k, (cy0 - sy0) * k, (cx1 - sx0) * k, (cy1 - sy0) * k)
        reg = img.transform((cx1 - cx0, cy1 - cy0), Image.EXTENT, box, Image.BILINEAR)
        f = self.fog(z) if fog else 1.0
        if f < 0.995 or alpha < 0.999:
            a = np.asarray(reg).astype(np.float32)
            if f < 0.995: a[..., :3] = a[..., :3] * f + np.array([6, 6, 16], np.float32) * (1 - f)
            if alpha < 0.999: a[..., 3] *= alpha
            reg = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGBA')
        frame.alpha_composite(reg, dest=(cx0, cy0))
        if erase is not None:   # nearer opaque card hides emissive light behind it
            erase.paste((0, 0, 0, 0), (cx0, cy0), mask=reg.getchannel('A'))


def find_coeffs(dst, src):
    """PIL perspective coefficients mapping output (dst) points to input (src) points."""
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        B += [u, v]
    return np.linalg.solve(np.array(A, float), np.array(B, float)).tolist()


# ------------------------------------------------------------------ carved lettering
def carved_from_mask(m, rs, stone=(118, 104, 100)):
    """m: float mask (0..1) at rs px/unit -> carved RGBA (lip highlight, dark cut, gold inlay) + gold alpha."""
    h, w = m.shape
    sh = lambda a, dx, dy: np.roll(np.roll(a, dy, 0), dx, 1)
    d = max(1, int(1.6 * rs))
    hi = np.clip(sh(m, d, d + 1) - m, 0, 1) * 0.85
    lo = np.clip(sh(m, -d, -d) - m, 0, 1) * 0.6
    er = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3 if rs < 2.5 else 5)), np.float32) / 255
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    n = fbm(h, w, int(12 * rs), 55, 3)
    gold = np.stack([np.full((h, w), 250.0), np.broadcast_to(214 - 60 * yy, (h, w)), np.broadcast_to(130 - 70 * yy, (h, w))], -1) * (0.82 + 0.3 * n)[..., None]
    out = np.zeros((h, w, 4), np.float32)
    def over(col, a):
        col = np.broadcast_to(np.asarray(col, np.float32), (h, w, 3))
        out[..., :3] = out[..., :3] * (1 - a[..., None]) + col * a[..., None]
        out[..., 3] = out[..., 3] * (1 - a) + a
    over((226, 212, 196), hi)
    over((10, 8, 14), lo)
    over((22, 16, 20), m)
    over(gold, er)
    out[..., 3] *= 255
    return np.clip(out, 0, 255).astype(np.uint8), er


def text_mask(lines, width_u, rs, spacing=0.0):
    """lines: [(text, size_units, y_units)] -> mask array (height from last line)."""
    H = max(y + s for _, s, y in lines) * 1.0 + 0.8 * max(s for _, s, _ in lines)
    im = Image.new('L', (int(width_u * rs), int(H * rs)), 0)
    d = ImageDraw.Draw(im)
    for txt, size, y in lines:
        f = ImageFont.truetype(FONT, int(size * rs))
        try: f.set_variation_by_axes([700])
        except Exception: pass
        if spacing:
            ws = [d.textlength(c, font=f) + spacing * rs for c in txt]
            x = (im.width - sum(ws)) / 2
            for c, wc in zip(txt, ws):
                d.text((x, (y + size * 0.9) * rs), c, font=f, fill=255, anchor='lm'); x += wc
        else:
            d.text((im.width / 2, (y + size * 0.9) * rs), txt, font=f, fill=255, anchor='mm')
    return np.asarray(im, np.float32) / 255


class Carving:
    """carved sprite that reveals along a sweep (chisel front) with sparks."""
    def __init__(self, mask, rs, x0, y0, z, sweep_y=0.0, gold_glow=0.35):
        self.rgba, self.gold = carved_from_mask(mask, rs)
        h, w = mask.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        sw = xx / w + sweep_y * yy / h
        self.sweep = (sw - sw.min()) / (sw.max() - sw.min())
        self.rs, self.x0, self.y0, self.z, self.gold_glow = rs, x0, y0, z, gold_glow
        ys, xs = np.nonzero(mask > 0.5)
        o = np.argsort(self.sweep[ys, xs])
        self.pts = np.stack([xs[o], ys[o]], 1); self.pts_sw = self.sweep[ys[o], xs[o]]
        self.full = Image.fromarray(self.rgba, 'RGBA')
        g = np.zeros((h, w, 4), np.uint8); g[..., 0] = 255; g[..., 1] = 200; g[..., 2] = 110; g[..., 3] = (self.gold * 255).astype(np.uint8)
        self.glow_img = Image.fromarray(g, 'RGBA').filter(ImageFilter.GaussianBlur(3 * rs))
        self.gold_img = Image.fromarray(g, 'RGBA')

    def image(self, p):
        if p >= 1: return self.full
        a = self.rgba.copy()
        a[..., 3] = (a[..., 3] * np.clip((p * 1.08 - self.sweep) / 0.04, 0, 1)).astype(np.uint8)
        return Image.fromarray(a, 'RGBA')

    def front_points(self, p, n, rng):
        f = p * 1.08 - 0.02
        i = np.searchsorted(self.pts_sw, f)
        lo, hi = max(0, i - 400), min(len(self.pts), i + 20)
        if hi <= lo: return []
        sel = self.pts[rng.integers(lo, hi, n)]
        return [(self.x0 + x / self.rs, self.y0 + y / self.rs) for x, y in sel]


# ------------------------------------------------------------------ static cards
STONE = (104, 94, 92)


def stone_blocks(p, x0, y0, x1, y1, seed, bw=150, bh=70, col=STONE):
    r = np.random.default_rng(seed)
    for row, y in enumerate(np.arange(y0, y1, bh)):
        off = (row % 2) * bw / 2
        for x in np.arange(x0 - bw + off, x1 + bw, bw):
            c = np.array(col) * r.uniform(0.78, 1.1)
            p.paper([(x + 3, y + 3), (x + bw - 3, y + 3), (x + bw - 3, y + bh - 3), (x + 3, y + bh - 3)], c, seed + row * 131 + int(x), amp=1.8, edge=(160, 146, 136), edge_w=1.1)


def torch_bracket(p, x, y, seed):
    p.paper([(x - 10, y + 10), (x + 10, y + 10), (x + 6, y + 110), (x - 6, y + 110)], (46, 32, 22), seed, amp=1, edge=(110, 84, 56))
    p.paper([(x - 24, y - 6), (x + 24, y - 6), (x + 14, y + 24), (x - 14, y + 24)], (86, 64, 38), seed + 1, amp=1, edge=(170, 130, 80))
    p.paper([(x - 34, y + 90), (x + 34, y + 90), (x + 34, y + 104), (x - 34, y + 104)], (40, 36, 40), seed + 2, amp=1, edge=(110, 100, 100))


def cut(L, polys):
    img, x0, y0, rs = L
    m = Image.new('L', img.size, 0); d = ImageDraw.Draw(m)
    for poly in polys:
        d.polygon([((x - x0) * rs, (y - y0) * rs) for x, y in poly], fill=255)
    a = np.asarray(img).copy(); a[..., 3] = np.where(np.asarray(m) > 0, 0, a[..., 3])
    return (Image.fromarray(a, 'RGBA'), x0, y0, rs)


def build_door_wall():
    p = Painter(-1500, -1150, 3000, 2450, rs=1.4)
    p.d.rectangle((0, 0, p.im.width, p.im.height), fill=(60, 54, 56, 255))
    stone_blocks(p, -1500, -1150, 1500, 1300, 5, 170, 80)
    for i, (gx, gy) in enumerate([(-800, -420), (800, -420), (-820, 250), (820, 250), (-1150, -100), (1150, -100)]):
        glyph(p, gx, gy, 34, i, (30, 24, 26), 6)
    p.paper(circ(8, 14, RO + 22, 200), (16, 12, 12), 90, amp=4, edge=None)
    p.paper(circ(0, 0, RO, 200), BRONZE, 91, amp=1.2, edge=(210, 170, 110), edge_w=2)
    p.d.ellipse(p.P([(-RO + 9, -RO + 9), (RO - 9, RO - 9)]), outline=BRONZE_L + (255,), width=p.W(3))
    p.d.ellipse(p.P([(-RI - 8, -RI - 8), (RI + 8, RI + 8)]), outline=BRONZE_D + (255,), width=p.W(5))
    for k in range(12):
        a = -math.pi / 2 + k * 2 * math.pi / 12
        p.rivet((RO + RI) / 2 * math.cos(a), (RO + RI) / 2 * math.sin(a), 11)
    for k, (a, b) in enumerate([(-0.72 * RO, -0.72 * RO + 90), (-0.72 * RO + 94, 0.76 * RO - 110), (0.76 * RO - 106, 0.76 * RO)]):
        x = HX
        p.paper([(x - 30, a), (x + 30, a), (x + 30, b), (x - 30, b)], BRONZE, 500 + k, amp=1, edge=(200, 160, 100), edge_w=1.4)
        p.d.rectangle(p.P([(x - 30, a + 2), (x - 18, b - 2)]), fill=BRONZE_D + (255,))
        p.d.rectangle(p.P([(x - 6, a + 2), (x + 8, b - 2)]), fill=BRONZE_L + (255,))
    for sx in (-1, 1):
        torch_bracket(p, sx * 560, -170, 60 + sx)
    # sand drift at the foot of the door
    xs = np.arange(-1500, 1520, 20)
    p.paper([(-1500, 1300)] + [(x, 300 + 60 * math.exp(-((x + 700) / 400) ** 2) * -1 + 40 * math.sin(x / 130) + 0.00008 * x * x * 60) for x in xs] + [(1500, 1300)], (84, 64, 44), 70, amp=3, edge=(170, 140, 100))
    L = p.finish(6, shadow=None, tex=0.3)
    return cut(L, [torn(circ(0, 0, RI, 220), 1.0, 999)])


def rib_polys(side):
    opening = arch(-OPEN_HW, OPEN_HW, FLOOR + 2, SPRING)
    niche = arch(side * NX - 180, side * NX + 180, 300, -60)
    return opening, niche


def build_rib(i):
    side = SIDES[i]
    p = Painter(-1450, -1300, 2900, 2700, rs=1.3)
    p.d.rectangle((0, 0, p.im.width, p.im.height), fill=(56, 50, 54, 255))
    stone_blocks(p, -1450, -1300, 1450, FLOOR, 100 + i * 7)
    # voussoirs round the corridor arch
    r0 = OPEN_HW
    for k in range(13):
        a0 = math.pi + math.pi * k / 13; a1 = math.pi + math.pi * (k + 1) / 13
        pts = [(r0 * math.cos(a0), SPRING + r0 * math.sin(a0)), ((r0 + 80) * math.cos(a0), SPRING + (r0 + 80) * math.sin(a0)),
               ((r0 + 80) * math.cos(a1), SPRING + (r0 + 80) * math.sin(a1)), (r0 * math.cos(a1), SPRING + r0 * math.sin(a1))]
        p.paper(pts, np.array((128, 116, 110)) * (1.1 if k == 6 else 1), 200 + i * 20 + k, amp=1.6, edge=(180, 164, 150))
    for sx in (-1, 1):
        for k, y in enumerate(range(SPRING, FLOOR, 100)):
            x = sx * OPEN_HW + (0 if sx > 0 else -80)
            p.paper([(x, y), (x + 80, y), (x + 80, y + 98), (x, y + 98)], (120, 108, 104), 300 + i * 30 + k * 2 + sx, amp=1.5, edge=(176, 160, 146))
    # niche frame + blank plaque above it
    nx = side * NX
    fr = arch(nx - 214, nx + 214, 330, -60)
    p.paper(fr, (122, 110, 106), 400 + i, amp=2, edge=(180, 164, 150))
    p.paper([(nx - 250, 300), (nx + 250, 300), (nx + 262, 340), (nx - 262, 340)], (132, 120, 114), 410 + i, amp=1.6, edge=(190, 170, 150))
    p.paper([(nx - 270, -350), (nx + 270, -350), (nx + 270, -268), (nx - 270, -268)], (40, 34, 38), 420 + i, amp=2, edge=None)
    p.paper([(nx - 262, -344), (nx + 262, -344), (nx + 262, -276), (nx - 262, -276)], (84, 76, 80), 421 + i, amp=2, edge=(170, 156, 146))
    torch_bracket(p, -side * 590, -170, 500 + i)
    # corridor floor (ground row)
    p.paper([(-1450, FLOOR), (1450, FLOOR), (1450, 1400), (-1450, 1400)], (70, 60, 58), 600 + i, amp=2.5, edge=(150, 136, 126))
    r = np.random.default_rng(i)
    for k, y in enumerate((FLOOR + 28, FLOOR + 70, FLOOR + 130, FLOOR + 220, FLOOR + 360)):
        p.line([(-1450, y), (1450, y)], (48, 42, 42), 2.5)
    for x in range(-1400, 1450, 120):
        p.line([(x * 0.55, FLOOR), (x * 1.4, 1400)], (52, 46, 46), 2)
    L = p.finish(700 + i, shadow=None, tex=0.3)
    op, ni = rib_polys(side)
    return cut(L, [torn(op, 1.5, 710 + i), torn(ni, 1.2, 720 + i)])


def alcove_back(i):
    side = SIDES[i]; nx = side * NX
    p = Painter(nx - 290, -330, 580, 720, rs=2)
    p.d.rectangle((0, 0, p.im.width, p.im.height), fill=(30, 26, 32, 255))
    stone_blocks(p, nx - 290, -330, nx + 290, 390, 800 + i, 110, 55, col=(66, 58, 62))
    p.paper([(nx - 290, 250), (nx + 290, 250), (nx + 290, 390), (nx - 290, 390)], (60, 52, 50), 810 + i, amp=2, edge=(120, 106, 96))
    getattr(sys.modules[__name__], f'alcove_{i}')(p, nx)
    return p.finish(830 + i, shadow=(4, 6, 8, 150), tex=0.28)


def alcove_0(p, nx):   # LOST TREASURE: coins + jeweled crown
    r = np.random.default_rng(1)
    for cx, w, h in ((nx - 60, 220, 60), (nx + 90, 150, 36)):
        pts = [(cx - w / 2 + w * k / 30, 262 - h * math.sin(math.pi * k / 30) ** 0.8) for k in range(31)] + [(cx + w / 2, 268), (cx - w / 2, 268)]
        p.paper(pts, (184, 134, 46), int(cx), amp=2, edge=(240, 200, 110))
        for _ in range(int(w * h / 50)):
            u = r.uniform(-0.5, 0.5)
            coin(p, cx + u * w, 262 - h * math.sin(math.pi * (u + 0.5)) ** 0.8 * r.uniform(0.1, 0.95), r.uniform(6, 9), r)
    for _ in range(26):
        coin(p, nx + r.uniform(-160, 160), r.uniform(270, 300), r.uniform(6, 9), r)
    # crown resting on the pile
    cx, by = nx - 40, 200
    band = [(cx - 70, by), (cx + 70, by), (cx + 66, by - 36), (cx - 66, by - 36)]
    pts = [(cx - 66, by - 36)]
    for k in range(5):
        x = cx - 66 + k * 33
        pts += [(x + 8, by - 36), (x + 16.5, by - 92 + (14 if k % 2 else 0)), (x + 25, by - 36)]
    pts += [(cx + 66, by - 36)]
    p.paper(pts + [(cx + 66, by - 30), (cx - 66, by - 30)], (222, 176, 70), 30, amp=0.8, edge=(255, 230, 150))
    p.paper(band, (206, 156, 56), 31, amp=0.8, edge=(255, 226, 140))
    for k, (dx, c) in enumerate([(-44, (180, 30, 40)), (0, (40, 120, 190)), (44, (40, 150, 80))]):
        p.d.ellipse(p.P([(cx + dx - 11, by - 26), (cx + dx + 11, by - 8)]), fill=c + (255,), outline=(255, 230, 160, 255), width=p.W(2))
    for k in range(5):
        x = cx - 66 + k * 33 + 16.5; y = by - 92 + (14 if k % 2 else 0)
        p.d.ellipse(p.P([(x - 6, y - 6), (x + 6, y + 6)]), fill=(250, 220, 140, 255))


def alcove_1(p, nx):   # ANCIENT RUINS: collapsed archway, roots, sand
    p.d.rectangle(p.P([(nx - 290, -330), (nx + 290, 250)]), fill=(58, 44, 50, 255))
    for k in range(4):
        y = -200 + k * 60
        p.paper([(nx - 290, y), (nx + 290, y + 20), (nx + 290, y + 60), (nx - 290, y + 40)], np.array((86, 62, 60)) * (0.8 + 0.08 * k), 900 + k, amp=3, edge=None)
    col = (150, 132, 108)
    p.paper([(nx - 130, 240), (nx - 80, 240), (nx - 84, -80), (nx - 126, -80)], col, 910, amp=2, edge=(210, 190, 160))
    p.paper([(nx - 140, -80), (nx - 70, -80), (nx - 76, -104), (nx - 134, -104)], (170, 150, 122), 911, amp=1.5, edge=(220, 200, 170))
    p.paper([(nx + 80, 240), (nx + 128, 240), (nx + 124, 60), (nx + 112, 40), (nx + 96, 70), (nx + 84, 50)], col, 912, amp=2, edge=(210, 190, 160))
    # fallen lintel, tilted
    p.paper([(nx - 150, -130), (nx + 60, -60), (nx + 50, -20), (nx - 160, -90)], (160, 140, 114), 913, amp=2, edge=(220, 200, 170))
    for k in range(4):
        p.line([(nx - 120 + k * 45, -110 + k * 16), (nx - 118 + k * 45, -76 + k * 16)], (80, 66, 54), 3)
    p.paper([(nx + 60, 210), (nx + 170, 200), (nx + 180, 250), (nx + 50, 252)], (140, 122, 100), 914, amp=2, edge=(200, 180, 150))
    # roots swallowing the stones
    r = np.random.default_rng(3)
    for k in range(9):
        x, y = nx + r.uniform(-200, 200), -330
        pts = [(x, y)]
        for _ in range(12):
            x += r.uniform(-22, 22); y += r.uniform(20, 42); pts.append((x, y))
        p.line(pts, (44, 30, 20), r.uniform(4, 9))
        p.line([(a + 2, b - 1) for a, b in pts], (104, 76, 50), 1.5)
    # sand dunes swallowing the base
    xs = np.arange(nx - 290, nx + 300, 12)
    p.paper([(nx - 290, 390)] + [(x, 170 + 50 * math.sin((x - nx) / 70) + 30 * math.cos((x - nx) / 33)) for x in xs] + [(nx + 290, 390)], (176, 140, 88), 915, amp=2.5, edge=(236, 206, 150))


def alcove_2(p, nx):   # UNSOLVED MYSTERIES: torn map + red string into shadow
    m = [(nx - 190, -200), (nx + 150, -220), (nx + 170, 150), (nx - 175, 175)]
    p.paper(m, (206, 180, 128), 950, amp=5, edge=(244, 228, 190), edge_w=2.5)
    p.paper([(nx + 150, -220), (nx + 170, 150), (nx + 110, 60), (nx + 130, -40)], (30, 26, 32), 951, amp=6, edge=(244, 228, 190))   # torn-away corner
    r = np.random.default_rng(5)
    coast = [(nx - 150 + k * 12, -60 + 40 * math.sin(k * 0.5) + r.uniform(-8, 8)) for k in range(22)]
    p.line(coast, (112, 84, 52), 3)
    for k in range(10):
        x, y = nx + r.uniform(-160, 100), r.uniform(-180, 130)
        p.line([(x - 8, y), (x + 8, y)], (120, 96, 66), 2)
    for k in range(14):
        p.line([(nx - 170 + k * 22, -200 + k * 1.2), (nx - 168 + k * 22, 160 - k * 1.5)], (170, 146, 104), 1)
    route = [(nx - 130, 100), (nx - 60, 40), (nx - 90, -40), (nx + 10, -110), (nx + 60, -30)]
    for (x0, y0), (x1, y1) in zip(route[:-1], route[1:]):
        for f in np.linspace(0, 1, 6)[:-1]:
            x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
            p.d.ellipse(p.P([(x - 2.5, y - 2.5), (x + 2.5, y + 2.5)]), fill=(120, 40, 30, 255))
    p.line([(nx + 50, -40), (nx + 70, -20)], (160, 30, 25), 5); p.line([(nx + 70, -40), (nx + 50, -20)], (160, 30, 25), 5)
    pins = [(nx - 130, 100), (nx + 10, -110), (nx + 60, -30)]
    s = [pins[0], pins[1], pins[2], (nx + 140, 60), (nx + 230, 20), (nx + 300, 90)]
    p.line(s, (70, 10, 10), 5); p.line(s, (190, 30, 30), 3)
    for x, y in pins:
        p.d.ellipse(p.P([(x - 7, y - 7), (x + 7, y + 7)]), fill=(210, 50, 40, 255), outline=(60, 10, 10, 255), width=p.W(2))
    p.line([(nx - 60, 200), (nx - 10, 190), (nx + 30, 210)], (60, 40, 30), 3)   # magnifier handle
    p.d.ellipse(p.P([(nx - 110, 170), (nx - 50, 230)]), outline=(170, 130, 70, 255), width=p.W(6))
    # shadow swallowing the string to the right
    a = np.asarray(p.im).astype(np.float32)
    xx = np.arange(a.shape[1], dtype=np.float32)[None, :] / p.rs + p.x0
    shade = np.clip(1 - (xx - (nx + 150)) / 150, 0.08, 1)
    a[..., :3] *= shade[..., None]
    p.im = Image.fromarray(a.astype(np.uint8), 'RGBA'); p.d = ImageDraw.Draw(p.im)


def alcove_3(p, nx):   # LEGENDARY ARTIFACTS: pedestal (relic is drawn per frame)
    p.paper([(nx - 70, 262), (nx + 70, 262), (nx + 60, 236), (nx - 60, 236)], (140, 126, 116), 970, amp=1, edge=(200, 184, 170))
    p.paper([(nx - 44, 236), (nx + 44, 236), (nx + 38, 90), (nx - 38, 90)], (126, 112, 104), 971, amp=1, edge=(196, 180, 166))
    for k in range(3):
        p.line([(nx - 30 + k * 30, 100), (nx - 30 + k * 30, 228)], (92, 82, 78), 3)
    p.paper([(nx - 64, 90), (nx + 64, 90), (nx + 56, 66), (nx - 56, 66)], (150, 136, 124), 972, amp=1, edge=(210, 194, 180))


def relic_faces():
    """golden sun-disc relic: front + back images at 2x."""
    S = 4; R = 62
    out = []
    for back in (False, True):
        im = Image.new('RGBA', (int(2 * R * S + 8), int(2 * R * S + 8)), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        c = im.width / 2
        for k in range(16):
            a = k * math.pi / 8
            d.polygon([(c + R * S * math.cos(a), c + R * S * math.sin(a)), (c + 0.7 * R * S * math.cos(a + 0.2), c + 0.7 * R * S * math.sin(a + 0.2)), (c + 0.7 * R * S * math.cos(a - 0.2), c + 0.7 * R * S * math.sin(a - 0.2))], fill=(214, 164, 60, 255) if not back else (150, 110, 40, 255))
        d.ellipse((c - 0.72 * R * S, c - 0.72 * R * S, c + 0.72 * R * S, c + 0.72 * R * S), fill=(236, 190, 84, 255) if not back else (170, 124, 48, 255))
        d.ellipse((c - 0.6 * R * S, c - 0.6 * R * S, c + 0.6 * R * S, c + 0.6 * R * S), outline=(150, 100, 30, 255), width=3 * S)
        if not back:
            d.ellipse((c - 0.2 * R * S, c - 0.2 * R * S, c + 0.2 * R * S, c + 0.2 * R * S), fill=(40, 120, 150, 255), outline=(255, 230, 160, 255), width=2 * S)
            for k in range(8):
                a = k * math.pi / 4
                d.line((c + 0.28 * R * S * math.cos(a), c + 0.28 * R * S * math.sin(a), c + 0.52 * R * S * math.cos(a), c + 0.52 * R * S * math.sin(a)), fill=(160, 110, 36, 255), width=3 * S)
        else:
            for k in range(3):
                d.ellipse((c - (0.15 + 0.13 * k) * R * S, c - (0.15 + 0.13 * k) * R * S, c + (0.15 + 0.13 * k) * R * S, c + (0.15 + 0.13 * k) * R * S), outline=(120, 86, 30, 255), width=2 * S)
        out.append(im)
    return out, S


def relic_image(t):
    (front, back), S = A['relic']
    ang = t * 1.1
    c = math.cos(ang)
    img = front if c >= 0 else back
    w = max(3, int(img.width * abs(c)))
    face = img.resize((w, img.height), Image.LANCZOS)
    band = int(10 * S * abs(math.sin(ang)))
    out = Image.new('RGBA', (img.width + 40, img.height), (0, 0, 0, 0))
    ox = (out.width - w) // 2
    if band:
        e = Image.new('RGBA', face.size, (130, 90, 30, 0)); e.putalpha(face.getchannel('A'))
        sgn = 1 if math.sin(ang) > 0 else -1
        for k in range(band, 0, -max(1, band // 5)):
            out.alpha_composite(e, (ox + sgn * k // 2, 0))
    out.alpha_composite(face, (ox, 0))
    shade = 0.75 + 0.35 * abs(c)
    a = np.asarray(out).astype(np.float32); a[..., :3] *= shade
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGBA'), S


def build_entrance():
    p = Painter(-1600, -1400, 3200, 2900, rs=1.2)
    p.d.rectangle((0, 0, p.im.width, p.im.height), fill=(50, 44, 50, 255))
    stone_blocks(p, -1600, -1400, 1600, 400, 1100, 180, 84)
    for k in range(17):
        a0 = math.pi + math.pi * k / 17; a1 = math.pi + math.pi * (k + 1) / 17
        r0 = 720
        pts = [(r0 * math.cos(a0), -150 + r0 * math.sin(a0)), ((r0 + 110) * math.cos(a0), -150 + (r0 + 110) * math.sin(a0)),
               ((r0 + 110) * math.cos(a1), -150 + (r0 + 110) * math.sin(a1)), (r0 * math.cos(a1), -150 + r0 * math.sin(a1))]
        p.paper(pts, (122, 110, 106), 1200 + k, amp=2, edge=(180, 164, 150))
        glyph(p, (r0 + 55) * math.cos((a0 + a1) / 2), -150 + (r0 + 55) * math.sin((a0 + a1) / 2), 16, k, (40, 32, 34), 4)
    p.paper([(-1600, FLOOR), (1600, FLOOR), (1600, 1500), (-1600, 1500)], (70, 60, 58), 1250, amp=2.5, edge=(150, 136, 126))
    L = p.finish(1260, shadow=None, tex=0.3)
    return cut(L, [torn(arch(-720, 720, FLOOR + 2, -150), 2, 1270)])


def build_pillars():
    p = Painter(-1500, -1300, 3000, 2800, rs=1.3)
    for sx in (-1, 1):
        x = sx * 880
        p.paper([(x - 110, -1300), (x + 110, -1300), (x + 110, 1500), (x - 110, 1500)], (98, 88, 86), 1300 + sx, amp=2, edge=(170, 154, 144))
        for k in range(12):
            p.line([(x - 110, -1300 + k * 200), (x + 110, -1300 + k * 200)], (60, 54, 54), 3)
        p.paper([(x - 130, -380), (x + 130, -380), (x + 120, -330), (x - 120, -330)], (120, 108, 102), 1310 + sx, amp=1.5, edge=(190, 170, 156))
        torch_bracket(p, x, -290, 1320 + sx)
    return p.finish(1330, shadow=(8, 10, 12, 160), tex=0.3)


def build_back():
    p = Painter(-1900, -1500, 3800, 3100, rs=1.6)
    p.d.rectangle((0, 0, p.im.width, p.im.height), fill=(58, 52, 56, 255))
    stone_blocks(p, -1900, -1500, 1900, 320, 1400, 200, 90)
    # smooth carving panel for the wordmark + tagline + sigil
    p.paper([(-420, -520), (420, -520), (430, 40), (-430, 40)], (40, 34, 38), 1410, amp=3, edge=None)
    p.paper([(-410, -510), (410, -510), (418, 30), (-418, 30)], (70, 64, 72), 1411, amp=3, edge=(150, 136, 130))
    for sx in (-1, 1):
        torch_bracket(p, sx * 560, -290, 1420 + sx)
        p.paper([(sx * 1100 - 100, 320), (sx * 1100 + 100, 320), (sx * 1100 + 80, -60), (sx * 1100 - 80, -60)], (26, 22, 28), 1430 + sx, amp=2, edge=(110, 96, 90))
    p.paper([(-1900, 320), (1900, 320), (1900, 1600), (-1900, 1600)], (72, 62, 58), 1440, amp=2.5, edge=(150, 136, 126))
    r = np.random.default_rng(14)
    for sx in (-1, 1):   # treasure heaps either side
        cx = sx * 640
        pts = [(cx - 260 + 520 * k / 40, 360 - 90 * math.sin(math.pi * k / 40) ** 0.8) for k in range(41)] + [(cx + 260, 370), (cx - 260, 370)]
        p.paper(pts, (184, 134, 46), 1450 + sx, amp=2, edge=(240, 200, 110))
        for _ in range(170):
            u = r.uniform(-0.5, 0.5)
            coin(p, cx + u * 520, 360 - 90 * math.sin(math.pi * (u + 0.5)) ** 0.8 * r.uniform(0.1, 0.95), r.uniform(8, 12), r)
    return p.finish(1460, shadow=None, tex=0.3)


def build_front_row():
    p = Painter(-1500, 250, 3000, 900, rs=1.4)
    r = np.random.default_rng(21)
    for sx in (-1, 1):
        cx = sx * 520
        pts = [(cx - 200 + 400 * k / 30, 420 - 60 * math.sin(math.pi * k / 30) ** 0.8) for k in range(31)] + [(cx + 200, 430), (cx - 200, 430)]
        p.paper(pts, (184, 134, 46), 1500 + sx, amp=2, edge=(240, 200, 110))
        for _ in range(90):
            u = r.uniform(-0.5, 0.5)
            coin(p, cx + u * 400, 420 - 60 * math.sin(math.pi * (u + 0.5)) ** 0.8 * r.uniform(0.1, 0.95), r.uniform(8, 12), r)
    return p.finish(1510, shadow=(4, 6, 8, 150))


def build_sigil_mask(rs):
    """cartouche with keyhole seals + SUBSCRIBE."""
    W_, H_ = 440, 96
    im = Image.new('L', (int(W_ * rs), int(H_ * rs)), 0); d = ImageDraw.Draw(im)
    P = lambda pts: [(x * rs, y * rs) for x, y in pts]
    d.rounded_rectangle((6 * rs, 10 * rs, (W_ - 6) * rs, (H_ - 10) * rs), radius=38 * rs, outline=255, width=int(5 * rs))
    d.rounded_rectangle((16 * rs, 20 * rs, (W_ - 16) * rs, (H_ - 20) * rs), radius=30 * rs, outline=255, width=int(2 * rs))
    for cx in (52, W_ - 52):
        d.ellipse(P([(cx - 26, H_ / 2 - 26), (cx + 26, H_ / 2 + 26)]), outline=255, width=int(3 * rs))
        d.polygon(P(keyhole_poly(cx, H_ / 2 + 2, 70)), fill=255)
    f = ImageFont.truetype(FONT, int(40 * rs))
    try: f.set_variation_by_axes([800])
    except Exception: pass
    x = W_ / 2 * rs; txt = 'SUBSCRIBE'
    ws = [d.textlength(c, font=f) + 4 * rs for c in txt]; x -= sum(ws) / 2
    for c, wc in zip(txt, ws):
        d.text((x, H_ / 2 * rs + 2 * rs), c, font=f, fill=255, anchor='lm'); x += wc
    return np.asarray(im, np.float32) / 255, W_, H_


def logo_word_mask(rs, height_u):
    lg = Image.open(os.path.join(HERE, 'assets', 'logo.jpg')).convert('RGB').crop((330, 1110, 1670, 1995))
    a = np.asarray(lg).astype(np.float32)
    lum = a.mean(axis=2)
    m = np.clip((lum - 70) / 90, 0, 1)
    w_u = height_u * lg.width / lg.height
    im = Image.fromarray((m * 255).astype(np.uint8)).resize((int(w_u * rs), int(height_u * rs)), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255, w_u


# ------------------------------------------------------------------ timeline helpers
def door_angle(t):
    return math.radians(80) * ss(seg(t, 2.6, 6.2))


def reveal(i, t):
    a, b = HOLD[i]
    return ss(seg(t, a - 1.3, a - 0.2))


def carve_prog(i, t):
    a, _ = HOLD[i]
    return min(1.0, max(0.0, seg(t, a - 0.35, a + 0.45)))


def ignite(k, t):
    return ss(seg(t, 22.8 + 0.45 * k, 23.3 + 0.45 * k))


def chest_angle(t):
    return math.radians(100) * ease_out(seg(t, 21.2, 21.9)) if t > 21.2 else 0.0


def sigil_level(t):
    lv = ss(seg(t, T_SIGIL[1] - 0.1, T_SIGIL[1] + 0.5)) * 0.75
    lv += 1.6 * math.exp(-((t - T_PULSE) / 0.18) ** 2)
    return lv


# ------------------------------------------------------------------ assets
A = {}


def build_assets():
    A['door'] = build_door_wall(); A['leaf'] = base.build_leaf()
    A['ribs'] = [build_rib(i) for i in range(4)]
    A['alc'] = [alcove_back(i) for i in range(4)]
    A['ent'] = build_entrance(); A['pillars'] = build_pillars(); A['back'] = build_back(); A['front'] = build_front_row()
    A['relic'] = relic_faces()
    rs = 3.0
    A['labels'] = []
    for i, txt in enumerate(LABELS):
        m = text_mask([(txt, 38, 0)], 520, rs, spacing=2)
        A['labels'].append(Carving(m, rs, SIDES[i] * NX - 260, -310 - m.shape[0] / rs / 2, Z_RIBS[i] - 1))
    wm, wu = logo_word_mask(2.0, 318)
    A['word'] = Carving(wm, 2.0, -wu / 2, -470, Z_BACK - 1, sweep_y=0.45)
    tm = text_mask([('NEW EPISODES EVERY WEEK', 32, 0)], 760, 2.5, spacing=5)
    A['tag'] = Carving(tm, 2.5, -380, -106 - tm.shape[0] / 2.5 / 2, Z_BACK - 1)
    sm, sw, sh = build_sigil_mask(2.5)
    A['sigil'] = Carving(sm, 2.5, -sw / 2, -70, Z_BACK - 1)
    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    e = np.sqrt(((xx - OW / 2) / (OW * 0.55)) ** 2 + ((yy - OH / 2) / (OH * 0.62)) ** 2) + (fbm(OH, OW, 60, 7, 4) - 0.5) * 0.3
    A['vig'] = np.clip(1 - 0.8 * np.clip((e - 0.74) / 0.5, 0, 1) ** 1.4, 0.06, 1)[..., None].astype(np.float32)
    r = np.random.default_rng(77)
    A['dust'] = np.stack([r.uniform(-900, 900, 700), r.uniform(-600, 420, 700), r.uniform(-800, 8200, 700), r.uniform(0, 1, 700)], 1)
    shaft = np.zeros((600, 300), np.float32)
    yy, xx = np.mgrid[0:600, 0:300].astype(np.float32)
    half = 0.16 + 0.3 * yy / 600
    shaft = np.clip(1 - np.abs(xx / 300 - 0.5) / half * 2, 0, 1) ** 1.6 * (0.35 + 0.65 * yy / 600)
    A['shaft'] = shaft_img = Image.fromarray(np.dstack([np.full_like(shaft, 255), np.full_like(shaft, 214), np.full_like(shaft, 140), shaft * 120]).astype(np.uint8), 'RGBA')
    A['cshaft'] = shaft_img.resize((300, 190), Image.BILINEAR)


# ------------------------------------------------------------------ frame render
def draw_leaf(frame, cam, t, em=None):
    th = door_angle(t)
    img, x0, y0, rs = A['leaf']
    W_ = img.width / rs
    xa = x0
    if th > 0.001:
        zmin = cam.z + NEAR + 5
        xa = max(x0, HX + zmin / math.sin(th))   # clip the part of the leaf that is behind the camera
    xb = x0 + W_
    if xb - xa < 2: return
    def P(xl, yl):
        d = xl - HX
        return cam.proj((HX + d * math.cos(th), yl, d * math.sin(th)))[0]
    corners = [P(xa, y0), P(xb, y0), P(xb, y0 + img.height / rs), P(xa, y0 + img.height / rs)]
    if any(c is None for c in corners): return
    xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
    bx0, by0 = max(0, int(min(xs))), max(0, int(min(ys))); bx1, by1 = min(OW, int(max(xs)) + 1), min(OH, int(max(ys)) + 1)
    if bx1 - bx0 < 2 or by1 - by0 < 2: return
    src = [((xa - x0) * rs, 0), (img.width, 0), (img.width, img.height), ((xa - x0) * rs, img.height)]
    coeffs = find_coeffs([(x - bx0, y - by0) for x, y in corners], src)
    reg = img.transform((bx1 - bx0, by1 - by0), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    a = np.asarray(reg).astype(np.float32); a[..., :3] *= (1 - 0.45 * math.sin(th)) * cam.fog(300)
    reg = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGBA')
    frame.alpha_composite(reg, dest=(bx0, by0))
    if em is not None: em.paste((0, 0, 0, 0), (bx0, by0), mask=reg.getchannel('A'))


def lights(t, cam):
    L = []   # (world pos, radius, rgb, intensity)
    for i, z in enumerate([0] + Z_RIBS):
        xs = (-560, 560) if i == 0 else (-SIDES[i - 1] * 590,)
        for k, x in enumerate(xs):
            fl = 1 + 0.12 * math.sin(t * 8.3 + i * 2 + k) + 0.07 * math.sin(t * 17.9 + i) + 0.05 * math.sin(t * 3.1 + i * 5)
            L.append(((x, -215, z - 10), 520, (1.05, 0.6, 0.28), 1.0 * fl))
    for i, z in enumerate(Z_RIBS):
        L.append(((SIDES[i] * NX, 60, z + 60), 430, GLOW[i], 0.45 + 0.85 * reveal(i, t)))
    conv = ss(seg(t, 21.0, 24.5))
    L.append(((0, -100, Z_BACK), 900, (1.0, 0.7, 0.36), 0.2 + 0.25 * conv))
    for k, (x, y, z) in enumerate([(-880, -335, Z_PILLAR), (880, -335, Z_PILLAR), (-560, -335, Z_BACK), (560, -335, Z_BACK)]):
        g = ignite(k, t)
        if g > 0: L.append(((x, y, z - 10), 480, (1.05, 0.62, 0.3), 0.85 * g * (1 + 0.08 * math.sin(t * 9 + k))))
    ca = chest_angle(t)
    if ca > 0: L.append(((0, 150, Z_CHEST), 360, (1.1, 0.85, 0.45), 0.8 * min(1, ca / 0.8)))
    wg = ss(seg(t, T_WORD[0], T_WORD[1]))
    if wg > 0: L.append(((0, -300, Z_BACK - 20), 520, (1.0, 0.78, 0.45), 0.25 * wg))
    lv = sigil_level(t)
    if lv > 0: L.append(((0, -20, Z_BACK - 20), 320, (1.1, 0.8, 0.4), 0.35 * lv))
    return L


def render_frame(fi):
    t = fi / FPS
    cam = Cam(t)
    frame = Image.new('RGBA', (OW, OH), (4, 4, 10, 255))
    em = Image.new('RGBA', (OW, OH), (0, 0, 0, 0))
    items = []
    def C(L, z, **kw): items.append((z, lambda: cam.card(frame, L, z, erase=em, **kw)))
    def E(L, z, **kw): items.append((z - 0.5, lambda: cam.card(em, L, z, **kw)))
    def FN(z, fn): items.append((z, fn))
    C(A['back'], Z_BACK)
    for key, prog, gl in ((A['word'], seg(t, *T_WORD), 0.85), (A['tag'], seg(t, *T_TAG), 0.8), (A['sigil'], seg(t, *T_SIGIL), 0.0)):
        if prog > 0:
            C((key.image(min(1, prog)), key.x0, key.y0, key.rs), key.z)
            if gl: E((key.gold_img if prog >= 1 else key.image(prog), key.x0, key.y0, key.rs), key.z - 0.2, alpha=gl * min(1, prog) * 0.55)
            if gl: E((key.glow_img, key.x0, key.y0, key.rs), key.z - 0.3, alpha=min(1, prog) * 0.35)
    ch, anc = chest_sprite(chest_angle(t))
    cs = 1.0
    C((ch, -anc[0] * cs, 300 - anc[1] * cs, base.RS / cs), Z_CHEST)
    C(A['front'], Z_CHEST - 150)
    C(A['pillars'], Z_PILLAR)
    C(A['ent'], Z_ENT)
    for i in range(4):
        z = Z_RIBS[i]
        if z - cam.z < NEAR: continue       # arch already passed: its alcove is inside the wall behind us
        C(A['alc'][i], z + 120)
        if i == 3:
            img, S = relic_image(t)
            bob = 6 * math.sin(t * 1.4)
            C((img, SIDES[i] * NX - img.width / S / 2, -10 + bob - img.height / S / 2, S), z + 110)
        C(A['ribs'][i], z)
        lb = A['labels'][i]; p = carve_prog(i, t)
        if p > 0:
            C((lb.image(p), lb.x0, lb.y0, lb.rs), lb.z)
            E((lb.glow_img, lb.x0, lb.y0, lb.rs), lb.z - 0.2, alpha=0.55 * min(1, p))
            E((lb.gold_img if p >= 1 else lb.image(p), lb.x0, lb.y0, lb.rs), lb.z - 0.3, alpha=0.5)
    FN(300, lambda: draw_leaf(frame, cam, t, em))
    C(A['door'], 0)
    # torch flames (depth-sorted so walls hide them)
    flames = [((x, -176, 0), 1.0) for x in (-560, 560)] + [((-SIDES[i] * 590, -176, Z_RIBS[i]), 1.0) for i in range(4)]
    flames += [((x, -296, z), ignite(k, t)) for k, (x, z) in enumerate([(-880, Z_PILLAR), (880, Z_PILLAR), (-560, Z_BACK), (560, Z_BACK)])]
    for k, ((x, y, z), g) in enumerate(flames):
        if g <= 0.01: continue
        f = flame_sprite(t, k * 3.7, 0.9 * g)
        E((f, x - f.width / base.RS / 2, y - f.height / base.RS + 8, base.RS), z - 3)
    for i in (0, 3):   # light shafts inside alcoves 1 and 4
        sh = A['shaft']
        if Z_RIBS[i] - cam.z >= NEAR:
            E((sh, SIDES[i] * NX - 90, -300, sh.width / 180), Z_RIBS[i] + 100, alpha=0.4 + 0.6 * reveal(i, t))
    def crown_glints():
        dd = ImageDraw.Draw(em)
        for k, (dx, dy) in enumerate([(-84, 112), (-40, 146), (4, 112), (48, 146), (-40, 180), (0, 170), (44, 180)]):
            v = max(0.0, math.sin(t * 2.6 + k * 1.7)) ** 10 * (0.3 + 0.7 * reveal(0, t))
            s, sc = cam.proj((SIDES[0] * NX - 40 + dx + 16, -300 + dy + 60, Z_RIBS[0] + 110))
            if s is None or v < 0.05: continue
            rr = 7 * sc * v
            dd.line((s[0] - rr, s[1], s[0] + rr, s[1]), fill=(255, 245, 210, int(240 * v)), width=2)
            dd.line((s[0], s[1] - rr, s[0], s[1] + rr), fill=(255, 245, 210, int(240 * v)), width=2)
    if Z_RIBS[0] - cam.z >= NEAR: FN(Z_RIBS[0] + 105, crown_glints)
    ca = chest_angle(t)
    if ca > 0:
        sh = A['shaft']; k = min(1, ca / 0.8)
        cshaft = A['cshaft']   # short glow rising out of the chest, kept below the sigil
        E((cshaft, -110, 215 - cshaft.height * 220 / cshaft.width, cshaft.width / 220), Z_CHEST - 1, alpha=0.5 * k)
    lv = sigil_level(t)
    if lv > 0:
        sg = A['sigil']
        E((sg.glow_img, sg.x0, sg.y0, sg.rs), sg.z - 0.4, alpha=min(1.0, 0.8 * lv))
        E((sg.gold_img, sg.x0, sg.y0, sg.rs), sg.z - 0.5, alpha=min(1.0, 0.6 * lv))
        def sigil_bloom():
            s, sc = cam.proj((0, -22, Z_BACK - 5))
            if s is None: return
            rr = int(150 * sc * (0.8 + 0.25 * min(lv, 2)))
            paste(em, glow_blob(rr, (255, 190, 90), round(min(0.45, 0.16 * lv), 2)), s[0] - 2 * rr, s[1] - 2 * rr)
        FN(Z_BACK - 3, sigil_bloom)
    items.sort(key=lambda it: -it[0])
    for _, fn in items:
        fn()

    # -------- lighting
    q = 4; lw, lh = OW // q, OH // q
    light = np.broadcast_to(np.array([0.15, 0.15, 0.27], np.float32), (lh, lw, 3)).copy()
    for pos, R, col, inten in lights(t, cam):
        s, sc = cam.proj(pos)
        if s is None or inten <= 0: continue
        rr = R * sc / q
        if s[0] / q + rr * 2 < 0 or s[0] / q - rr * 2 > lw or s[1] / q + rr * 2 < 0 or s[1] / q - rr * 2 > lh: continue
        light += radial(lw, lh, s[0] / q, s[1] / q, rr)[..., None] * np.array(col, np.float32) * inten * cam.fog(pos[2]) ** 0.5
    light = np.minimum(light, 1.7)
    light = np.asarray(Image.fromarray(np.clip(light * 80, 0, 255).astype(np.uint8)).resize((OW, OH), Image.BILINEAR), np.float32) / 80
    rgb = np.asarray(frame.convert('RGB'), np.float32) * light

    # -------- screen-space emissive overlays
    dd = ImageDraw.Draw(em)
    # carving sparks + grit
    rng = np.random.default_rng(fi)
    active = [(A['labels'][i], carve_prog(i, t)) for i in range(4)] + [(A['word'], seg(t, *T_WORD)), (A['tag'], seg(t, *T_TAG)), (A['sigil'], seg(t, *T_SIGIL))]
    for cv, p in active:
        if 0 < p < 1:
            for (x, y) in cv.front_points(p, 10, rng):
                s, sc = cam.proj((x, y, cv.z - 2))
                if s is None: continue
                for j in range(2):
                    vx, vy = rng.uniform(-30, 30) * sc, rng.uniform(-40, 10) * sc
                    rr = rng.uniform(1.0, 2.4)
                    dd.ellipse((s[0] + vx * 0.2 - rr, s[1] + vy * 0.2 - rr, s[0] + vx * 0.2 + rr, s[1] + vy * 0.2 + rr), fill=(255, 214, 140, 230))
    # convergence: four streams of alcove light flowing into the chamber
    tgt, _ = cam.proj((0, 205, Z_CHEST))
    if tgt is not None:
        starts = [(-120, 250), (OW + 120, 330), (-120, 880), (OW + 120, 820)]
        for sidx, (sx, sy) in enumerate(starts):
            col = tuple(int(255 * c) for c in GLOW[sidx])
            for k in range(46):
                t0 = 21.0 + sidx * 0.3 + k * 0.045
                u = (t - t0) / 1.7
                if u <= 0 or u >= 1: continue
                cx_, cy_ = (sx + tgt[0]) / 2 + (-1 if sx < 0 else 1) * -200, min(sy, tgt[1]) - 260
                for tr in range(4):
                    uu = max(0.0, u - tr * 0.025); e = ss(uu)
                    x = (1 - e) ** 2 * sx + 2 * (1 - e) * e * cx_ + e * e * tgt[0] + 14 * math.sin(k * 1.3 + t * 3)
                    y = (1 - e) ** 2 * sy + 2 * (1 - e) * e * cy_ + e * e * tgt[1] + 10 * math.cos(k * 0.9 + t * 3)
                    rr = (4.6 - tr * 0.9) * (1 - 0.4 * e)
                    a = int(230 * math.sin(math.pi * u) * (1 - tr * 0.22))
                    dd.ellipse((x - rr, y - rr, x + rr, y + rr), fill=col + (a,))
    # sparkles rising from the open chest
    if ca > 0:
        r2 = np.random.default_rng(4)
        for j in range(40):
            born = 21.4 + r2.uniform(0, 13); age = (t - born)
            if age < 0 or age > 2.5: continue
            s, sc = cam.proj((r2.uniform(-90, 90) + 10 * math.sin(age * 2 + j), 170 - age * r2.uniform(40, 90), Z_CHEST - 10))
            if s is None: continue
            al = int(230 * math.sin(math.pi * age / 2.5)); rr = 2.2 * sc
            dd.ellipse((s[0] - rr, s[1] - rr, s[0] + rr, s[1] + rr), fill=(255, 236, 180, al))
    # sigil flare sparks
    if abs(t - T_PULSE) < 1.2:
        s, sc = cam.proj((0, -22, Z_BACK - 5))
        r3 = np.random.default_rng(9)
        for j in range(60):
            ang = r3.uniform(0, 2 * math.pi); sp = r3.uniform(120, 320)
            age = t - T_PULSE
            if age < 0: break
            x = s[0] + math.cos(ang) * (230 * sc + sp * age); y = s[1] + math.sin(ang) * (60 * sc + sp * age * 0.5) + 60 * age * age
            a = int(220 * max(0, 1 - age / 1.2)); rr = 2
            dd.ellipse((x - rr, y - rr, x + rr, y + rr), fill=(255, 220, 150, a))
    # floating dust in the torchlight
    D = A['dust']
    for x, y, z, ph in D:
        yy_ = y + 20 * math.sin(t * 0.4 + ph * 6); xx_ = x + 15 * math.sin(t * 0.3 + ph * 9)
        s, sc = cam.proj((xx_, yy_, z))
        if s is None or sc < 0.25 or not (0 <= s[0] < OW and 0 <= s[1] < OH): continue
        rr = min(4.0, 1.2 * sc); a = int(min(160, 90 * sc) * (0.5 + 0.5 * math.sin(t * 1.3 + ph * 20)))
        dd.ellipse((s[0] - rr, s[1] - rr, s[0] + rr, s[1] + rr), fill=(255, 214, 160, a))
    # door grinding: grit from the frame
    if 2.4 < t < 7.5:
        r4 = np.random.default_rng(31)
        for j in range(220):
            born = r4.uniform(2.4, 6.4); age = t - born
            if age < 0 or age > 1.8: continue
            ang = r4.uniform(math.pi * 1.1, math.pi * 1.9)
            x0_, y0_ = RO * math.cos(ang), RO * math.sin(ang)
            s, sc = cam.proj((x0_ + 5 * math.sin(age * 4 + j), y0_ + 0.5 * 320 * age * age, -5))
            if s is None: continue
            a = int(200 * min(1, (1.8 - age) / 0.5)); rr = r4.uniform(0.8, 2.0) * sc
            dd.ellipse((s[0] - rr, s[1] - rr, s[0] + rr, s[1] + rr), fill=(190, 160, 120, a))
    ea = np.asarray(em, np.float32); al = ea[..., 3:4] / 255
    rgb = rgb * (1 - al) + ea[..., :3] * al

    # -------- grade: indigo shadows, warm highlights
    lum = rgb.mean(axis=2, keepdims=True) / 255
    rgb = rgb * (1 - lum) * np.array([0.88, 0.9, 1.1], np.float32) + rgb * lum * np.array([1.08, 0.99, 0.84], np.float32)
    rgb *= A['vig']
    fade = min(1.0, ss(t / 1.0), 1 - ss(seg(t, 33.9, 34.95)))
    return np.clip(rgb * fade, 0, 255).astype(np.uint8).tobytes()


# ------------------------------------------------------------------ audio
def audio():
    from scipy.signal import butter, sosfilt, fftconvolve
    n = int(SR * DUR); t = np.arange(n) / SR
    r = np.random.default_rng(3)
    bp = lambda x, lo, hi, o=2: sosfilt(butter(o, [lo, hi], 'bandpass', fs=SR, output='sos'), x)
    lp = lambda x, f, o=2: sosfilt(butter(o, f, 'lowpass', fs=SR, output='sos'), x)
    dry = np.zeros((n, 2)); send = np.zeros(n)

    def add(t0, sig, pan=(1, 1), wet=0.3):
        i0 = int(t0 * SR); L = min(len(sig), n - i0)
        if L <= 0: return
        dry[i0:i0 + L] += sig[:L, None] * np.array(pan); send[i0:i0 + L] += sig[:L] * wet

    wind = lp(np.cumsum(r.normal(0, 1, n)) * 0.002, 380); wind -= lp(wind, 20)
    dry += (wind * (0.6 + 0.4 * np.sin(2 * np.pi * t / 7.3) ** 2) * 0.8)[:, None] * np.array([1, 0.85])
    cr = np.zeros(n); idx = r.choice(n, 520, replace=False); cr[idx] = r.uniform(-1, 1, 520)
    dry += (bp(cr, 1500, 6000) * 0.4)[:, None] * np.array([0.8, 1.0])
    # stone grinding door
    L = int(3.8 * SR); tt = np.arange(L) / SR; u = tt / tt[-1]
    g = bp(r.normal(0, 1, L), 50, 380, 3) * 1.6 * (0.55 + 0.45 * np.abs(np.sin(2 * np.pi * 5.1 * tt + 2 * np.sin(2 * np.pi * 0.8 * tt))))
    g += bp(r.normal(0, 1, L), 700, 2200) * 0.2 * np.abs(np.sin(2 * np.pi * 11 * tt))
    g += np.sin(2 * np.pi * 40 * tt + 2 * np.sin(2 * np.pi * 1.3 * tt)) * 0.3
    add(2.6, g * np.sin(np.pi * u) ** 0.5, (1, 0.9), 0.6)
    L = int(1.6 * SR); tt = np.arange(L) / SR
    add(6.25, np.sin(2 * np.pi * (50 * np.exp(-tt * 3) + 34) * tt) * np.exp(-tt * 4) * 0.8 + lp(r.normal(0, 1, L), 900) * np.exp(-tt * 10) * 0.4, (1, 1), 0.7)
    trickle = bp(r.normal(0, 1, n), 3000, 9000) * (0.5 + 0.5 * (r.random(n) > 0.97))
    dry += (trickle * 0.12 * np.clip((t - 2.4) / 0.4, 0, 1) * np.clip((7.2 - t) / 1.0, 0, 1))[:, None]
    # air whoosh as each arch passes
    for z in [0] + Z_RIBS + [Z_ENT]:
        tp = float(np.interp(z, _Z, _T))
        L = int(1.4 * SR); tt = np.arange(L) / SR
        w = bp(r.normal(0, 1, L), 250, 1500) * np.exp(-((tt - 0.7) / 0.28) ** 2) * 0.18
        add(tp - 0.7, w, (1, 1), 0.4)
    # chisel taps + grit for every carving
    def chisel(t0, dur, n_taps, loud=1.0):
        for k in range(n_taps):
            tk = t0 + dur * k / n_taps + r.uniform(-0.02, 0.02)
            L = int(0.12 * SR); tt = np.arange(L) / SR
            tap = bp(r.normal(0, 1, L), 1800, 7000) * np.exp(-tt * 70) * 0.5 + np.sin(2 * np.pi * r.uniform(2400, 3200) * tt) * np.exp(-tt * 45) * 0.18
            add(tk, tap * loud, (r.uniform(0.7, 1), r.uniform(0.7, 1)), 0.4)
        L = int((dur + 0.4) * SR); tt = np.arange(L) / SR
        add(t0, bp(r.normal(0, 1, L), 2500, 8000) * 0.05 * np.clip(tt / 0.1, 0, 1) * np.clip((dur + 0.4 - tt) / 0.4, 0, 1) * loud, (1, 1), 0.3)
    for i in range(4):
        chisel(HOLD[i][0] - 0.35, 0.8, 7)
    chisel(T_WORD[0], T_WORD[1] - T_WORD[0], 16, 1.2)
    chisel(T_TAG[0], T_TAG[1] - T_TAG[0], 7, 0.8)
    chisel(T_SIGIL[0], T_SIGIL[1] - T_SIGIL[0], 6, 0.8)
    # convergence swell + torch ignitions
    L = int(4.5 * SR); tt = np.arange(L) / SR
    swell = bp(r.normal(0, 1, L), 1500, 7000) * np.clip(tt / 3.0, 0, 1) ** 2 * np.exp(-np.clip(tt - 3.3, 0, None) * 3) * 0.12
    drone = sum(np.sin(2 * np.pi * f * tt) * a_ for f, a_ in ((110, 1), (164.8, 0.6), (220, 0.4))) * np.clip(tt / 3.0, 0, 1) * np.exp(-np.clip(tt - 3.3, 0, None) * 1.5) * 0.05
    add(20.8, swell + drone, (1, 1), 0.6)
    for k in range(4):
        L = int(0.9 * SR); tt = np.arange(L) / SR
        whoosh = bp(r.normal(0, 1, L), 300, 3000) * np.exp(-((tt - 0.18) / 0.12) ** 2) * 0.35 + lp(r.normal(0, 1, L), 500) * np.exp(-tt * 6) * 0.25
        add(22.8 + 0.45 * k, whoosh, (1.0, 0.5) if k % 2 == 0 else (0.5, 1.0), 0.5)
    # chest creak
    L = int(0.7 * SR); tt = np.arange(L) / SR
    ph = 2 * np.pi * np.cumsum(140 + 90 * tt + 40 * np.sin(2 * np.pi * 3 * tt)) / SR
    saw = 2 * ((ph / (2 * np.pi)) % 1) - 1
    add(21.1, bp(saw * (np.sin(2 * np.pi * 26 * tt) > 0.2), 300, 2400) * np.sin(np.pi * tt / tt[-1]) * 0.3)
    # soft chime as the sigil lights, brighter at the flare pulse
    def chime(t0, gain, base_f=1318.5):
        L = int(4.0 * SR); tt = np.arange(L) / SR; c = np.zeros(L)
        for k, (m, amp, dec) in enumerate([(1, 1, 1.3), (1.5, 0.7, 1.1), (2, 0.5, 0.9), (2.67, 0.3, 0.8), (1.19, 0.5, 1.5)]):
            st = int(k * 0.05 * SR)
            c[st:] += np.sin(2 * np.pi * base_f * m * tt[:L - st]) * np.exp(-tt[:L - st] * dec) * amp
        add(t0, c * gain + bp(r.normal(0, 1, L), 6000, 12000) * np.exp(-tt * 1.2) * 0.04 * gain * 5, (0.9, 1), 0.6)
    chime(T_SIGIL[1] - 0.05, 0.05, 987.8)
    chime(T_PULSE - 0.05, 0.11)
    L = int(1.2 * SR); tt = np.arange(L) / SR
    add(T_PULSE - 0.1, bp(r.normal(0, 1, L), 200, 2500) * np.exp(-((tt - 0.15) / 0.1) ** 2) * 0.3 + np.sin(2 * np.pi * (60 * np.exp(-tt * 5) + 40) * tt) * np.exp(-tt * 7) * 0.5, (1, 1), 0.5)
    # reverb
    L = int(1.8 * SR); tt = np.arange(L) / SR
    ir = lp(r.normal(0, 1, L) * np.exp(-tt * 2.8), 3000); ir /= np.sqrt((ir ** 2).sum()) * 6
    wet = fftconvolve(send, ir)[:n]
    out = dry + np.stack([wet, np.roll(wet, 331)], 1)
    out *= (np.clip(t / 1.0, 0, 1) * np.clip((DUR - t) / 1.2, 0, 1))[:, None]
    out = out / (np.abs(out).max() + 1e-9) * 0.85
    return (out * 32767).astype(np.int16)


def main():
    out = next((a for a in sys.argv[1:] if not a.startswith('--')), 'the_buried_vault_trailer_35s.mp4')
    print('building assets...', flush=True)
    build_assets()
    if '--preview' in sys.argv:
        os.makedirs('preview', exist_ok=True)
        times = [float(a) for a in sys.argv if a.replace('.', '').isdigit()] or [0.5, 2.0, 4.5, 6.5, 8.2, 9.6, 10.6, 11.8, 13.4, 16.6, 19.8, 21.8, 23.5, 25.2, 26.5, 28.2, 29.6, 30.6, 32.5, 34.5]
        with Pool(4) as pool:
            res = pool.map(render_frame, [int(tt * FPS) for tt in times])
        cols = 5; rows = (len(res) + cols - 1) // cols
        sheet = Image.new('RGB', (384 * cols, 216 * rows))
        for i, b in enumerate(res):
            sheet.paste(Image.frombytes('RGB', (OW, OH), b).resize((384, 216)), ((i % cols) * 384, (i // cols) * 216))
        sheet.save('preview/sheet35.png')
        for tt, b in zip(times, res):
            Image.frombytes('RGB', (OW, OH), b).save(f'preview/s{tt:05.2f}.png')
        return
    wav = 'audio35.wav'
    with wave.open(wav, 'wb') as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes(audio().tobytes())
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-',
                             '-i', wav, '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
                             '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for i, b in enumerate(pool.imap(render_frame, range(NF), chunksize=4)):
            proc.stdin.write(b)
            if i % 96 == 0: print(f'frame {i}/{NF}', flush=True)
    proc.stdin.close(); proc.wait(); os.remove(wav)
    print('wrote', out)


if __name__ == '__main__':
    main()
