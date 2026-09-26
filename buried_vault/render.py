"""The Buried Vault — 20 s channel trailer, procedural paper-cut diorama.

Renders 1920x1080 (16:9 landscape) video + synthesized ambient audio with numpy/Pillow,
then encodes with ffmpeg.   Usage: python render.py [out.mp4] [--preview]
"""
import math, os, subprocess, sys, wave
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
OW, OH = 1920, 1080            # output frame (16:9 landscape)
W, H = 1080, 1920               # world design space (layout coordinates)
XL, XR = -520, 1600             # world canvas x-extent (wider than the design space for landscape)
CW = XR - XL
FPS, DUR = 24, 20.0
NF = int(FPS * DUR)
RS = 2                        # static layers are painted at 2x world resolution
C0 = np.array([540.0, 960.0])  # world parallax pivot
SC = np.array([OW / 2, OH / 2])
SR = 48000

# ------------------------------------------------------------------ helpers
def hexc(h):
    h = h.lstrip('#'); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def ss(x):  # smootherstep, clamped
    x = min(1.0, max(0.0, x)); return x * x * x * (x * (x * 6 - 15) + 10)


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


def torn(pts, amp, seed, step=5.0, closed=True):
    """subdivide polygon and jitter along normals -> torn-paper edge."""
    seed = int(seed) % 1000003
    pts = [tuple(p) for p in pts]
    if closed:
        pts = pts + [pts[0]]
    out = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        L = math.hypot(x1 - x0, y1 - y0); n = max(1, int(L / step))
        for i in range(n):
            f = i / n; out.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
    N = len(out)
    nz = noise1d(N, seed, 3) * amp + np.random.default_rng(seed + 1).normal(0, amp * 0.35, N)
    res = []
    for i in range(N):
        xa, ya = out[i - 1]; xb, yb = out[(i + 1) % N]
        tx, ty = xb - xa, yb - ya; l = math.hypot(tx, ty) + 1e-6
        res.append((out[i][0] - ty / l * nz[i], out[i][1] + tx / l * nz[i]))
    return res


def arch(x0, x1, ybot, yspring, n=40):
    cx, r = (x0 + x1) / 2, (x1 - x0) / 2
    pts = [(x0, ybot), (x0, yspring)]
    for i in range(1, n):
        a = math.pi + math.pi * i / n
        pts.append((cx + r * math.cos(a), yspring + r * math.sin(a)))
    pts += [(x1, yspring), (x1, ybot)]
    return pts


class Painter:
    """paper-cut painter on a world-sized RGBA layer at RS scale."""
    def __init__(self, w=None, h=H, ox=None):
        self.ox = -XL if ox is None and w is None else (ox or 0)
        w = CW if w is None else w
        self.w, self.h = w, h
        self.im = Image.new('RGBA', (int(w * RS), int(h * RS)), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def P(self, pts):
        return [((x + self.ox) * RS, y * RS) for x, y in pts]

    def X(self, x):
        return (x + self.ox) * RS

    def paper(self, pts, col, seed, amp=2.2, edge=(236, 214, 170), edge_w=1.6):
        tp = torn(pts, amp, seed)
        if edge:  # pale torn fibre rim
            ep = torn(pts, amp * 1.2, seed + 7)
            cx = sum(p[0] for p in ep) / len(ep); cy = sum(p[1] for p in ep) / len(ep)
            ep2 = []
            for x, y in ep:
                dx, dy = x - cx, y - cy; l = math.hypot(dx, dy) + 1e-6
                ep2.append((x + dx / l * edge_w, y + dy / l * edge_w))
            self.d.polygon(self.P(ep2), fill=edge + (255,))
        self.d.polygon(self.P(tp), fill=tuple(col) + (255,))

    def finish(self, texture_seed, shadow=(8, 12, 14, 150), tex_amt=0.22):
        a = np.asarray(self.im).astype(np.float32)
        h, w = a.shape[:2]
        n = fbm(h, w, 64, texture_seed, 5) - 0.5
        fib = fbm(h, w, 3, texture_seed + 3, 2) - 0.5
        a[..., :3] *= (1 + tex_amt * n + 0.12 * fib)[..., None]
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGBA')
        if shadow:
            dx, dy, blur, alpha = shadow
            al = im.getchannel('A').filter(ImageFilter.GaussianBlur(blur * RS / 2))
            sh = Image.new('RGBA', im.size, (6, 4, 12, 0))
            sh.putalpha(al.point(lambda v: int(v * alpha / 255)))
            base = Image.new('RGBA', im.size, (0, 0, 0, 0))
            base.alpha_composite(sh, (int(dx * RS), int(dy * RS)))
            base.alpha_composite(im)
            im = base
        return im


# ------------------------------------------------------------------ scene constants (world coords)
DX0, DX1, DBOT, DSPR = 300, 780, 1420, 900       # door opening (arch)
PLQ = (190, 452, 890, 628)                        # plaque box
TORCHES = [(168, 930), (912, 930), (-300, 930), (1380, 930)]                # flame base positions
CHEST = (610, 1250)                               # chest front-bottom center (chamber layer)
FLOOR_Y = 1225
EXS = 1.3
DEPTH = dict(sky=0.12, dunes=0.35, cback=0.72, cprops=0.8, explorer=0.84, leaves=0.98, wall=1.0, sand=1.06)


def build_sky():
    p = Painter()
    y = np.linspace(0, 1, H * RS, dtype=np.float32)[:, None, None]
    top, bot = np.array(hexc('#05040f'), np.float32), np.array(hexc('#1b1740'), np.float32)
    g = top * (1 - y) + bot * y
    g = np.broadcast_to(g, (H * RS, CW * RS, 3)).copy()
    n = fbm(H * RS, CW * RS, 300, 3, 4)
    g += ((n - 0.5) * 30)[..., None] * np.array([0.6, 0.5, 1.0])
    im = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8)).convert('RGBA')
    d = ImageDraw.Draw(im)
    r = np.random.default_rng(5)
    for _ in range(480):
        x, yy = r.random() * CW * RS, r.random() * 700 * RS
        s = r.choice([1, 1, 2, 3]); a = int(r.uniform(70, 200))
        d.ellipse((x - s, yy - s, x + s, yy + s), fill=(240, 225, 190, a))
    # paper moon
    mx, my, mr = (1180 - XL) * RS, 330 * RS, 60 * RS
    d.ellipse((mx - mr - 4, my - mr - 4, mx + mr + 4, my + mr + 4), fill=(236, 214, 170, 255))
    d.ellipse((mx - mr, my - mr, mx + mr, my + mr), fill=(214, 196, 150, 255))
    d.ellipse((mx - mr + 26 * RS, my - mr - 10 * RS, mx + mr + 26 * RS, my + mr - 10 * RS), fill=(10, 9, 26, 255))
    return im


def build_dunes():
    p = Painter()
    for i, (base, col) in enumerate([(430, '#2a2146'), (500, '#3a2a4a')]):
        pts = [(XL - 50, H + 50), (XL - 50, base)]
        for x in range(XL - 50, XR + 100, 40):
            pts.append((x, base + 45 * math.sin(x / (170 + 60 * i) + i * 2) + 20 * math.sin(x / 57 + i)))
        pts.append((XR + 100, H + 50))
        p.paper(pts, hexc(col), 20 + i, amp=2.5, edge=(120, 95, 110))
    return p.finish(21, shadow=(4, 8, 10, 120))


def build_chamber_back():
    p = Painter()
    # back wall
    p.d.rectangle((0, 0, CW * RS, H * RS), fill=hexc('#2a2338') + (255,))
    r = np.random.default_rng(31)
    for row, y in enumerate(range(540, FLOOR_Y, 62)):
        off = (row % 2) * 60
        for x in range(XL - 120 + off, XR + 120, 120):
            c = np.array(hexc('#3b3048')) * r.uniform(0.85, 1.12)
            p.paper([(x + 3, y + 3), (x + 117, y + 3), (x + 117, y + 59), (x + 3, y + 59)], c.astype(int), 1000 + row * 50 + x, amp=1.6, edge=(92, 78, 90), edge_w=1.0)
    # carved glyph frieze
    fy = 830
    p.paper([(180, fy - 38), (900, fy - 38), (900, fy + 38), (180, fy + 38)], hexc('#4a3b50'), 40, amp=2, edge=(120, 100, 100))
    for i, x in enumerate(range(220, 880, 70)):
        glyph(p.d, p.X(x), fy * RS, 22 * RS, i, (26, 20, 32, 255), 3 * RS)
    # alcoves with urns
    for ax in (330, 760):
        p.paper(arch(ax - 55, ax + 55, 1100, 960), hexc('#16121f'), 50 + ax, amp=2, edge=(90, 70, 80))
        urn(p, ax, 1090, 40 + ax)
    # pillars
    for px in (235, 845):
        p.paper([(px - 42, 560), (px + 42, 560), (px + 46, FLOOR_Y + 20), (px - 46, FLOOR_Y + 20)], hexc('#453852'), 60 + px, amp=2, edge=(120, 100, 105))
        p.paper([(px - 58, 540), (px + 58, 540), (px + 50, 580), (px - 50, 580)], hexc('#51425c'), 61 + px, amp=2, edge=(130, 110, 110))
        for k in range(4):
            yy = 640 + k * 160
            p.d.line(p.P([(px - 30, yy), (px + 30, yy)]), fill=(30, 24, 38, 255), width=2 * RS)
    # floor
    p.paper([(XL - 40, FLOOR_Y), (XR + 40, FLOOR_Y), (XR + 40, H + 40), (XL - 40, H + 40)], hexc('#4a3a3a'), 70, amp=2.5, edge=(150, 120, 100))
    for k in range(-14, 15):
        x0 = 540 + k * 70; x1 = 540 + k * 170
        p.d.line(p.P([(x0, FLOOR_Y), (x1, H)]), fill=(58, 44, 44, 255), width=2 * RS)
    for yy in (1262, 1318, 1400, 1520):
        p.d.line(p.P([(XL, yy), (XR, yy)]), fill=(58, 44, 44, 255), width=2 * RS)
    return p.finish(71, shadow=None, tex_amt=0.3)


def glyph(d, x, y, s, kind, col, w):
    k = kind % 6
    if k == 0:   # eye
        d.arc((x - s, y - s * 0.6, x + s, y + s * 0.6), 200, 340, fill=col, width=w)
        d.arc((x - s, y - s * 0.6, x + s, y + s * 0.6), 20, 160, fill=col, width=w)
        d.ellipse((x - s * 0.25, y - s * 0.25, x + s * 0.25, y + s * 0.25), fill=col)
    elif k == 1:  # spiral
        pts = [(x + s * i / 60 * math.cos(i / 60 * 13), y + s * i / 60 * math.sin(i / 60 * 13)) for i in range(61)]
        d.line(pts, fill=col, width=w)
    elif k == 2:  # sun
        d.ellipse((x - s * 0.4, y - s * 0.4, x + s * 0.4, y + s * 0.4), outline=col, width=w)
        for a in range(8):
            ang = a * math.pi / 4
            d.line((x + s * 0.55 * math.cos(ang), y + s * 0.55 * math.sin(ang), x + s * math.cos(ang), y + s * math.sin(ang)), fill=col, width=w)
    elif k == 3:  # triangle / pyramid
        d.polygon([(x, y - s), (x + s, y + s * 0.8), (x - s, y + s * 0.8)], outline=col, width=w)
        d.line((x - s * 0.5, y, x + s * 0.5, y), fill=col, width=w)
    elif k == 4:  # key
        d.ellipse((x - s, y - s * 0.4, x - s * 0.2, y + s * 0.4), outline=col, width=w)
        d.line((x - s * 0.2, y, x + s, y), fill=col, width=w)
        d.line((x + s * 0.6, y, x + s * 0.6, y + s * 0.4), fill=col, width=w)
        d.line((x + s * 0.9, y, x + s * 0.9, y + s * 0.4), fill=col, width=w)
    else:  # wave
        pts = [(x - s + i * 2 * s / 30, y + s * 0.4 * math.sin(i / 30 * 4 * math.pi)) for i in range(31)]
        d.line(pts, fill=col, width=w)
        d.line([(px, py + s * 0.5) for px, py in pts], fill=col, width=w)


def urn(p, x, base, seed):
    pts = []
    for i in range(21):
        f = i / 20; yy = base - f * 90
        rr = 16 + 20 * math.sin(f * math.pi * 0.95) ** 1.2 - (6 if f > 0.85 else 0)
        pts.append((x + rr, yy))
    pts += [(x - px + x, py) for px, py in reversed(pts)]
    p.paper(pts, hexc('#7a4a2e'), seed, amp=1.2, edge=(190, 150, 110))
    p.d.line(p.P([(x - 30, base - 45), (x + 30, base - 45)]), fill=(170, 130, 60, 255), width=3 * RS)


def build_chamber_props():
    p = Painter()
    r = np.random.default_rng(81)
    # coin piles
    for cx, base, wdt, hgt in [(380, 1275, 190, 62), (800, 1265, 170, 52), (610, 1277, 300, 26)]:
        pts = [(cx - wdt / 2 + wdt * i / 30, base - hgt * math.sin(math.pi * i / 30) ** 0.8) for i in range(31)] + [(cx + wdt / 2, base + 6), (cx - wdt / 2, base + 6)]
        p.paper(pts, hexc('#b8862e'), int(cx), amp=2.5, edge=(240, 200, 110))
        for _ in range(int(wdt * hgt / 60)):
            u = r.uniform(-0.5, 0.5); yy = base - hgt * math.sin(math.pi * (u + 0.5)) ** 0.8 * r.uniform(0.1, 0.95)
            coin(p, cx + u * wdt, yy, r.uniform(7, 11), r)
    # scattered coins on floor
    for _ in range(46):
        coin(p, r.uniform(260, 880), r.uniform(1282, 1370), r.uniform(7, 11), r)
    # treasure map + coiled roll
    mp = [(420, 1318), (560, 1304), (575, 1362), (430, 1380)]
    p.paper(mp, hexc('#c9ad7a'), 90, amp=2.5, edge=(240, 225, 190))
    route = [(440, 1360), (470, 1340), (500, 1350), (530, 1330), (553, 1325)]
    for (x0, y0), (x1, y1) in zip(route[:-1], route[1:]):
        for f in np.linspace(0, 1, 5)[:-1]:
            xx, yy = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
            p.d.ellipse(p.P([(xx - 1.5, yy - 1.5), (xx + 1.5, yy + 1.5)]), fill=(120, 40, 30, 255))
    p.d.line(p.P([(548, 1319), (560, 1331)]), fill=(150, 30, 25, 255), width=3 * RS)
    p.d.line(p.P([(560, 1319), (548, 1331)]), fill=(150, 30, 25, 255), width=3 * RS)
    for k in range(5):  # coiled roll at the end
        rr = 16 - k * 2.8
        p.d.arc(p.P([(413 - rr, 1348 - rr), (413 + rr, 1348 + rr)]), 0, 330, fill=(150, 118, 70, 255), width=int(2.5 * RS))
    p.d.ellipse(p.P([(399, 1334), (428, 1363)]), outline=(236, 214, 170, 255), width=RS)
    return p.finish(82, shadow=(5, 8, 8, 140))


def coin(p, x, y, s, r):
    p.d.ellipse(p.P([(x - s, y - s * 0.55), (x + s, y + s * 0.55)]), fill=(120, 80, 20, 255))
    p.d.ellipse(p.P([(x - s + 1.3, y - s * 0.55 + 0.6), (x + s - 1.3, y + s * 0.55 - 1.2)]), fill=hexc(['#e0b24a', '#d4a03c', '#f0c860'][r.integers(3)]) + (255,))


def build_wall():
    p = Painter()
    # ruined facade silhouette
    top = [(XL - 40, H + 40), (XL - 40, 360)]
    for x in range(XL - 40, XR + 80, 30):
        step = 330 + (40 if (x // 180) % 2 else 0) + 25 * math.sin(x * 0.05)
        top.append((x, step))
    top.append((XR + 80, H + 40))
    hole = arch(DX0, DX1, DBOT, DSPR)
    p.paper(top, hexc('#4b3e5e'), 100, amp=3.5, edge=(170, 150, 150))
    # stone blocks
    r = np.random.default_rng(101)
    for row, y in enumerate(range(380, 1500, 74)):
        off = (row % 2) * 80
        for x in range(XL - 160 + off, XR + 160, 160):
            c = np.array(hexc('#5a4a6c')) * r.uniform(0.8, 1.1)
            p.paper([(x + 4, y + 4), (x + 156, y + 4), (x + 156, y + 70), (x + 4, y + 70)], c.astype(int), 2000 + row * 97 + x, amp=2.0, edge=(120, 104, 120), edge_w=1.2)
    # carved symbols near door (torchlit)
    for i, (gx, gy) in enumerate([(110, 760), (970, 760), (110, 1110), (970, 1110), (225, 700), (855, 700), (-160, 700), (1240, 700), (-400, 900), (1480, 900), (-160, 1120), (1240, 1120)]):
        glyph(p.d, p.X(gx), gy * RS, 26 * RS, i + 2, (34, 26, 44, 255), 4 * RS)
        glyph(p.d, p.X(gx) + 3, gy * RS + 3, 26 * RS, i + 2, (130, 110, 120, 255), 2 * RS)
    # arch voussoirs
    cx, rad = (DX0 + DX1) / 2, (DX1 - DX0) / 2
    for i in range(11):
        a0 = math.pi + math.pi * i / 11; a1 = math.pi + math.pi * (i + 1) / 11
        pts = [(cx + rad * math.cos(a0), DSPR + rad * math.sin(a0)), (cx + (rad + 70) * math.cos(a0), DSPR + (rad + 70) * math.sin(a0)),
               (cx + (rad + 70) * math.cos(a1), DSPR + (rad + 70) * math.sin(a1)), (cx + rad * math.cos(a1), DSPR + rad * math.sin(a1))]
        p.paper(pts, (np.array(hexc('#6b5a78')) * (1.08 if i == 5 else 1)).astype(int), 300 + i, amp=2, edge=(170, 150, 150))
    for side in (-1, 1):  # jambs
        x = DX0 - 70 if side < 0 else DX1
        for k, yy in enumerate(range(int(DSPR), DBOT + 40, 90)):
            p.paper([(x, yy), (x + 70, yy), (x + 70, yy + 88), (x, yy + 88)], hexc('#665573'), 400 + k * 3 + side, amp=2, edge=(170, 150, 150))
    # keystone glyph
    glyph(p.d, p.X(cx), (DSPR - rad - 36) * RS, 18 * RS, 0, (30, 22, 40, 255), 3 * RS)
    # plaque (blank; lettering is stamped in later)
    x0, y0, x1, y1 = PLQ
    p.paper([(x0 - 14, y0 - 14), (x1 + 14, y0 - 14), (x1 + 14, y1 + 14), (x0 - 14, y1 + 14)], hexc('#3a2f48'), 500, amp=2.5, edge=(150, 130, 130))
    p.paper([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], hexc('#7d6a84'), 501, amp=2.5, edge=(200, 180, 170))
    for (bx, by) in [(x0 + 18, y0 + 18), (x1 - 18, y0 + 18), (x0 + 18, y1 - 18), (x1 - 18, y1 - 18)]:
        p.d.ellipse(p.P([(bx - 6, by - 6), (bx + 6, by + 6)]), fill=(60, 48, 60, 255))
        p.d.ellipse(p.P([(bx - 4, by - 5), (bx + 3, by + 2)]), fill=(150, 120, 90, 255))
    # torch brackets
    for tx, ty in TORCHES:
        p.paper([(tx - 12, ty), (tx + 12, ty), (tx + 7, ty + 90), (tx - 7, ty + 90)], hexc('#3a2616'), tx, amp=1.4, edge=(120, 90, 60))
        p.paper([(tx - 20, ty - 6), (tx + 20, ty - 6), (tx + 14, ty + 14), (tx - 14, ty + 14)], hexc('#5c4a30'), tx + 1, amp=1.2, edge=(160, 130, 80))
        p.paper([(tx - 30, ty + 70), (tx + 30, ty + 70), (tx + 30, ty + 86), (tx - 30, ty + 86)], hexc('#2d2a30'), tx + 2, amp=1.2, edge=(110, 100, 100))
    im = p.finish(102, shadow=(6, 12, 14, 170))
    # cut the doorway
    m = Image.new('L', im.size, 0)
    ImageDraw.Draw(m).polygon([((x - XL) * RS, y * RS) for x, y in torn(hole, 2.0, 999)], fill=255)
    a = np.asarray(im).copy(); a[..., 3] = np.where(np.asarray(m) > 0, 0, a[..., 3])
    return Image.fromarray(a, 'RGBA')


def build_leaf(side):
    """one stone door leaf as a sprite (world coords local), side -1 left / +1 right."""
    wdt, hgt = 262, DBOT - (DSPR - 260) + 20
    p = Painter(wdt + 40, hgt + 40, ox=0)
    pts = [(20, 20), (wdt + 20, 20), (wdt + 20, hgt + 20), (20, hgt + 20)]
    p.paper(pts, hexc('#5f4f6c'), 600 + side, amp=2.5, edge=(170, 150, 150))
    # cracks, carvings
    r = np.random.default_rng(610 + side)
    for k in range(3):
        x, y = r.uniform(40, wdt), r.uniform(80, hgt)
        pts = [(x, y)]
        for _ in range(7):
            x += r.uniform(-18, 18); y += r.uniform(10, 40); pts.append((x, y))
        p.d.line(p.P(pts), fill=(28, 20, 34, 255), width=2 * RS)
    inner = 30 if side < 0 else 0
    p.paper([(40 + inner, 300), (wdt - 20 + inner, 300), (wdt - 20 + inner, hgt - 60), (40 + inner, hgt - 60)], hexc('#6d5c7a'), 620 + side, amp=2, edge=(180, 160, 150))
    cxg = 20 + wdt / 2 + (14 if side < 0 else -14)
    for i, yy in enumerate(range(380, hgt - 100, 150)):
        glyph(p.d, p.X(cxg), yy * RS, 34 * RS, i * 2 + (0 if side < 0 else 1), (34, 26, 44, 255), 4 * RS)
    # half of a central ring handle
    hx = wdt + 20 if side < 0 else 20
    p.d.arc(p.P([(hx - 40, 760), (hx + 40, 840)]), 0, 360, fill=(150, 120, 60, 255), width=5 * RS)
    im = p.finish(630 + side, shadow=(-8 * side, 6, 10, 160))
    return im


def build_sand():
    p = Painter()
    pts = [(XL - 60, H + 60), (XL - 60, 1150)]
    for x in range(XL - 60, XR + 120, 30):
        y = 1412 - 60 * math.exp(-((x + 380) / 200) ** 2) - 70 * math.exp(-((x - 1420) / 220) ** 2) - 260 * math.exp(-((x - 20) / 260) ** 2) - 170 * math.exp(-((x - 1060) / 210) ** 2) + 8 * math.sin(x / 45) - 40 * (1 - math.exp(-((x - 540) / 330) ** 2))
        pts.append((x, y))
    pts.append((XR + 120, H + 60))
    p.paper(pts, hexc('#8a6a44'), 700, amp=3, edge=(230, 200, 150))
    pts2 = [(XL - 60, H + 60), (XL - 60, 1600)] + [(x, 1650 - 60 * math.sin(x / 260 + 1) - 12 * math.sin(x / 33)) for x in range(XL - 60, XR + 120, 30)] + [(XR + 120, H + 60)]
    p.paper(pts2, hexc('#6e5236'), 701, amp=3, edge=(200, 170, 120))
    # half-buried debris: broken column + skull-ish rock
    p.paper([(850, 1290), (1010, 1260), (1030, 1320), (870, 1350)], hexc('#5a4a60'), 702, amp=2, edge=(170, 150, 150))
    for k in range(4):
        p.d.line(p.P([(880 + k * 35, 1285 - k * 7), (895 + k * 35, 1342 - k * 7)]), fill=(40, 30, 46, 255), width=2 * RS)
    return p.finish(703, shadow=(0, -10, 16, 170))


# ------------------------------------------------------------------ dynamic sprites
def chest_sprite(a):
    """3D-projected paper chest. a = lid angle in radians. returns (sprite at 2x, anchor offset)."""
    S = RS * 1.0
    Wc, Hb, D, Hl = 170, 86, 92, 34
    tau = math.radians(22)
    size = (int(260 * S), int(320 * S))
    ox, oy = size[0] / 2, size[1] - 40 * S
    im = Image.new('RGBA', size, (0, 0, 0, 0)); d = ImageDraw.Draw(im)

    def proj(x, y, z):
        sy = y * math.cos(tau) - z * math.sin(tau)
        return (ox + x * S, oy - sy * S)

    def depth(x, y, z):
        return z * math.cos(tau) + y * math.sin(tau)

    faces = []
    NORM = {'front': (0, 0, 1), 'back': (0, 0, -1), 'top': (0, 1, 0), 'bottom': (0, -1, 0), 'left': (-1, 0, 0), 'right': (1, 0, 0)}
    def box(xr, yr, zr, cols, xf=None):
        (x0, x1), (y0, y1), (z0, z1) = xr, yr, zr
        V = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
        if xf: V = [xf(*v) for v in V]
        idx = {'front': (1, 3, 7, 5), 'back': (0, 4, 6, 2), 'top': (2, 6, 7, 3), 'bottom': (0, 1, 5, 4), 'left': (0, 2, 3, 1), 'right': (4, 5, 7, 6)}
        for name, q in idx.items():
            if name not in cols: continue
            pts3 = [V[i] for i in q]
            n = NORM[name]
            if xf: n = np.subtract(xf(*n), xf(0, 0, 0))
            nz = n[2] * math.cos(tau) + n[1] * math.sin(tau)
            if nz <= 0: continue
            faces.append((np.mean([depth(*v) for v in pts3]), [proj(*v) for v in pts3], cols[name], name))

    wood, wood_d, gold = (104, 62, 34), (72, 42, 24), (214, 168, 70)
    inner = (255, 196, 90) if a > 0.05 else (40, 20, 18)
    box((-Wc / 2, Wc / 2), (0, Hb), (-D / 2, D / 2), {'front': wood, 'top': inner, 'right': wood_d, 'left': wood_d})
    hz = -D / 2; hy = Hb
    def lid_xf(x, y, z):
        y2, z2 = y - hy, z - hz
        c, s = math.cos(a), math.sin(a)
        return (x, hy + y2 * c + z2 * s, hz - y2 * s + z2 * c)
    box((-Wc / 2 - 3, Wc / 2 + 3), (Hb, Hb + Hl), (-D / 2 - 3, D / 2 + 3), {'front': wood, 'top': (120, 74, 40), 'bottom': (96, 58, 32), 'back': (86, 52, 30), 'left': wood_d, 'right': wood_d}, lid_xf)
    faces.sort(key=lambda f: f[0])
    for _, pts, col, name in faces:
        d.polygon(pts, fill=col + (255,), outline=(30, 18, 12, 255))
        (qa, qb, qc, qd) = [np.array(p_) for p_ in pts]
        quad = lambda u, v: tuple(qa + (qb - qa) * u + (qd - qa) * v + (qa - qb + qc - qd) * u * v)
        if name == 'top' and col[0] > 200:      # heaped coins inside the open chest
            cr = np.random.default_rng(5)
            for _ in range(70):
                cx_, cy_ = quad(cr.uniform(0.06, 0.94), cr.uniform(0.1, 0.9)); rr_ = cr.uniform(3.5, 6) * S
                d.ellipse((cx_ - rr_, cy_ - rr_ * 0.6, cx_ + rr_, cy_ + rr_ * 0.6), fill=(150, 100, 30, 255))
                d.ellipse((cx_ - rr_ + 1, cy_ - rr_ * 0.6 + 1, cx_ + rr_ - 1, cy_ + rr_ * 0.6 - 2), fill=(255, 214, 110, 255))
            continue
        if name == 'bottom':                    # velvet-lined inside of the lid with gold trim
            inset = [quad(0.08, 0.1), quad(0.92, 0.1), quad(0.92, 0.9), quad(0.08, 0.9)]
            d.polygon(inset, fill=(60, 16, 26, 255), outline=gold + (255,), width=int(3 * S))
            continue
        if col in (wood, (120, 74, 40)) or name == 'front':
            # gold bands on front-ish faces
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts
            for f in (0.18, 0.82):
                pa = (x0 + (x3 - x0) * f, y0 + (y3 - y0) * f); pb = (x1 + (x2 - x1) * f, y1 + (y2 - y1) * f)
                d.line([pa, pb], fill=gold + (255,), width=int(7 * S))
    # lock plate on body front
    fx, fy = proj(0, Hb * 0.72, D / 2)
    d.rectangle((fx - 12 * S, fy - 14 * S, fx + 12 * S, fy + 14 * S), fill=gold + (255,), outline=(90, 60, 20, 255))
    d.ellipse((fx - 4 * S, fy - 6 * S, fx + 4 * S, fy + 2 * S), fill=(30, 18, 12, 255))
    return im, (ox / RS, oy / RS)


def explorer_sprite(t, walk_phase, arm_ang, walking):
    """hooded explorer, returns sprite (2x) and anchor (feet) plus lantern local pos (world units)."""
    S = RS
    size = (int(260 * S), int(300 * S))
    ox, oy = size[0] / 2, size[1] - 20 * S
    im = Image.new('RGBA', size, (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    bob = 3.0 * abs(math.sin(walk_phase)) * walking
    sway = 1.2 * math.sin(walk_phase) * walking
    P = lambda x, y: (ox + (x + sway * (1 + y / 200)) * S, oy + (y - bob * (y < -20)) * S)
    # boots
    for k, side in enumerate((-1, 1)):
        lift = max(0.0, math.sin(walk_phase + k * math.pi)) * 6 * walking
        fx = side * 16 + (math.cos(walk_phase + k * math.pi) * 8 * walking)
        d.polygon([P(fx - 12, -lift), P(fx + 16, -lift), P(fx + 14, -14 - lift), P(fx - 10, -16 - lift)], fill=(30, 22, 20, 255))
    # cloak with torn hem
    hem = [P(-46 + i * 92 / 12, -10 - (6 if i % 2 else 0) - 4 * math.sin(i + walk_phase)) for i in range(13)]
    cloak = [P(-20, -150), P(20, -150), P(40, -60)] + [hem[-1]] + list(reversed(hem)) + [P(-40, -60)]
    d.polygon([(x - 2, y) for x, y in cloak], fill=(200, 170, 130, 255))  # paper rim
    d.polygon(cloak, fill=(44, 50, 70, 255))
    d.line([P(-6, -140), P(-10, -20)], fill=(30, 34, 50, 255), width=2 * S)
    # hood
    hood = [P(-30, -140), P(-26, -178), P(-6, -204), P(8, -214), P(24, -190), P(30, -150), P(20, -128), P(-22, -128)]
    d.polygon([(x - 2, y - 1) for x, y in hood], fill=(200, 170, 130, 255))
    d.polygon(hood, fill=(52, 58, 80, 255))
    d.ellipse([P(-12, -186), P(16, -148)], fill=(8, 6, 10, 255))
    # faint lantern-lit face glints
    d.ellipse([P(4, -170), P(8, -166)], fill=(255, 190, 110, 200))
    d.ellipse([P(-5, -170), P(-1, -166)], fill=(255, 190, 110, 150))
    # arm + lantern
    sh = (22, -128)
    L = 64
    hx, hy = sh[0] + L * math.cos(arm_ang), sh[1] + L * math.sin(arm_ang)
    d.line([P(*sh), P(hx, hy)], fill=(44, 50, 70, 255), width=14 * S)
    d.ellipse([P(hx - 7, hy - 7), P(hx + 7, hy + 7)], fill=(44, 50, 70, 255))
    swing = 0.12 * math.sin(t * 2.1)
    lx, ly = hx + 26 * math.sin(swing), hy + 26 * math.cos(swing)
    d.line([P(hx, hy), P(lx, ly - 10)], fill=(90, 70, 40, 255), width=2 * S)
    d.polygon([P(lx - 11, ly - 10), P(lx + 11, ly - 10), P(lx + 9, ly + 16), P(lx - 9, ly + 16)], fill=(255, 216, 130, 255), outline=(80, 56, 24, 255))
    d.rectangle([P(lx - 13, ly - 14), P(lx + 13, ly - 9)], fill=(120, 86, 36, 255))
    d.rectangle([P(lx - 11, ly + 16), P(lx + 11, ly + 20)], fill=(120, 86, 36, 255))
    d.line([P(lx, ly - 9), P(lx, ly + 16)], fill=(120, 86, 36, 255), width=S)
    lantern = (lx + sway, ly + 3 - bob)
    return im, (ox / S, oy / S), lantern


def flame_sprite(t, seed, scale=1.0):
    S = RS
    w, h = int(80 * S * scale), int(140 * S * scale)
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    cx, by = w / 2, h - 8 * S * scale
    for k, (col, sc) in enumerate([((214, 90, 30), 1.0), ((244, 160, 50), 0.72), ((255, 230, 150), 0.42)]):
        pts = []
        n = 28
        for i in range(n + 1):
            f = i / n; ang = math.pi * f
            wob = math.sin(t * 7.3 + seed + k + f * 5) * 0.12 + math.sin(t * 11.1 + seed * 2 + f * 9) * 0.06
            rx = 26 * sc * math.sin(ang) ** 0.9 * (1 + wob)
            yy = -f * 110 * sc * (1 + 0.1 * math.sin(t * 5 + seed + k))
            pts.append((cx - rx * S * scale + wob * 20 * S * scale * f, by + yy * S * scale))
        pts += [(cx + (cx - x), y) for x, y in reversed(pts)]
        d.polygon(pts, fill=col + (235,))
    return im


# ------------------------------------------------------------------ camera
KEYS = [  # t, cx, cy, zoom  (landscape framing)
    (0.0, 540, 930, 0.96),
    (2.5, 540, 936, 0.98),
    (9.2, 548, 1100, 1.55),
    (12.6, 560, 1106, 1.60),
    (13.4, 560, 1106, 1.62),
    (15.4, 540, 890, 0.94),
    (17.6, 540, 893, 0.95),
    (19.4, 540, 905, 0.98),
    (20.0, 540, 906, 0.982),
]


def camera(t):
    for (t0, x0, y0, z0), (t1, x1, y1, z1) in zip(KEYS[:-1], KEYS[1:]):
        if t <= t1:
            f = ss((t - t0) / (t1 - t0))
            return np.array([x0 + (x1 - x0) * f, y0 + (y1 - y0) * f]), z0 + (z1 - z0) * f
    return np.array(KEYS[-1][1:3], float), KEYS[-1][3]


def shake(t):
    s = 0.0
    if 15.1 < t < 15.9:
        k = t - 15.1; s = 5.0 * math.exp(-k * 7) * math.sin(k * 60)
    if 1.6 < t < 5.8:
        s += 0.8 * math.sin(t * 43) * math.sin(math.pi * seg(t, 1.6, 5.8))
    return s


class Cam:
    def __init__(self, t):
        self.c, self.z = camera(t)
        self.sh = shake(t)

    def of(self, d):
        zd = 1 + (self.z - 1) * d
        cd = C0 + (self.c - C0) * d + np.array([self.sh * 0.6, self.sh])
        return cd, zd

    def to_screen(self, p, d):
        cd, zd = self.of(d)
        return (np.asarray(p, float) - cd) * zd + SC, zd

    def draw_layer(self, frame, layer, d):
        cd, zd = self.of(d)
        x0 = (cd[0] - XL - OW / 2 / zd) * RS; y0 = (cd[1] - OH / 2 / zd) * RS
        x1 = (cd[0] - XL + OW / 2 / zd) * RS; y1 = (cd[1] + OH / 2 / zd) * RS
        frame.alpha_composite(layer.transform((OW, OH), Image.EXTENT, (x0, y0, x1, y1), Image.BILINEAR))

    def draw_sprite(self, frame, spr, anchor_px, world_pos, d, rot=0.0):
        """spr painted at RS; anchor_px = anchor in sprite world units."""
        s, zd = self.to_screen(world_pos, d)
        sc = zd / RS
        w, h = max(1, int(spr.width * sc)), max(1, int(spr.height * sc))
        im = spr.resize((w, h), Image.LANCZOS)
        x = s[0] - anchor_px[0] * zd; y = s[1] - anchor_px[1] * zd
        paste(frame, im, x, y)


def paste(canvas, sprite, x0, y0):
    x0, y0 = int(round(x0)), int(round(y0))
    sw, sh = sprite.size; cw, ch = canvas.size
    l, t = max(0, -x0), max(0, -y0); r, b = min(sw, cw - x0), min(sh, ch - y0)
    if r > l and b > t:
        canvas.alpha_composite(sprite.crop((l, t, r, b)), dest=(x0 + l, y0 + t))


# ------------------------------------------------------------------ timeline
def door_open(t):
    return ss(seg(t, 1.6, 5.8))


def explorer_state(t):
    walk = ss(seg(t, 7.2, 9.9))
    x = 180 + (408 - 180) * walk
    walking = min(ss(seg(t, 7.2, 7.7)), 1 - ss(seg(t, 9.4, 9.9)))
    phase = (t - 7.2) * 2 * math.pi * 0.85
    low = math.radians(80); up = math.radians(-58)
    raise_ = ss(seg(t, 10.0, 10.9))
    arm = low + (up - low) * raise_
    # sweep: arm swings across the room, settles toward the chest
    k = min(1.0, max(0.0, seg(t, 10.9, 12.6)))
    sw = math.radians(-24) * math.sin(1.5 * math.pi * ss(k))
    arm += sw * raise_
    return x, phase, arm, walking, raise_


def lantern_intensity(t):
    x = 0.25 + 0.75 * ss(seg(t, 9.9, 10.9))
    x *= 1 + 0.05 * math.sin(t * 9.1) + 0.03 * math.sin(t * 23.7)
    beat = math.exp(-((t - 18.55) / 0.12) ** 2) + 0.6 * math.exp(-((t - 18.85) / 0.12) ** 2)
    return x * (1 + 0.9 * beat)


def lid_angle(t):
    k = seg(t, 12.7, 13.25)
    creak = 0.06 * ss(seg(t, 12.45, 12.7))   # an initial hesitant crack
    return creak + math.radians(100) * ease_out(k) if k > 0 else creak


# ------------------------------------------------------------------ assets (global so workers inherit via fork)
A = {}


def build_assets():
    A['sky'] = build_sky(); A['dunes'] = build_dunes()
    A['cback'] = build_chamber_back(); A['cprops'] = build_chamber_props()
    A['wall'] = build_wall(); A['sand'] = build_sand()
    A['leafL'] = build_leaf(-1); A['leafR'] = build_leaf(1)
    A['logo'], A['logo_gold'] = build_logo()
    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    A['vig'] = (1 - 0.6 * np.clip(((xx - OW / 2) / (OW * 0.62)) ** 2 + ((yy - OH / 2) / (OH * 0.72)) ** 2 - 0.25, 0, 1))[..., None]
    A['grain'] = [np.asarray(Image.fromarray(np.random.default_rng(900 + i).normal(0, 6, (OH // 2, OW // 2)).astype(np.float32), 'F').resize((OW, OH), Image.BILINEAR))[..., None] for i in range(6)]
    A['lw'], A['lh'] = OW // 4, OH // 4
    r = np.random.default_rng(77)
    A['motes'] = r.random((70, 4))
    A['coinglints'] = [(r.uniform(300, 880), r.uniform(1230, 1370), r.uniform(0, 6.28)) for _ in range(26)]


def build_logo():
    """engraved wordmark sized to the plaque, at RS."""
    x0, y0, x1, y1 = PLQ
    w, h = (x1 - x0) * RS, (y1 - y0) * RS
    font_path = os.path.join(HERE, 'fonts', 'Cinzel.ttf')
    def font(sz):
        f = ImageFont.truetype(font_path, sz)
        try: f.set_variation_by_axes([800])
        except Exception: pass
        return f
    big, small = font(int(70 * RS)), font(int(40 * RS))
    layers = {}
    for name, off, col in [('hi', (1.5, 2.5), (214, 190, 170, 200)), ('cut', (0, 0), (26, 18, 30, 255)), ('gold', (-0.6, -1.0), (255, 204, 96, 255))]:
        im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        ox, oy = off[0] * RS, off[1] * RS
        d.text((w / 2 + ox, 48 * RS + oy), 'THE', font=small, fill=col, anchor='mm')
        for sgn in (-1, 1):
            d.line((w / 2 + sgn * 46 * RS + ox, 48 * RS + oy, w / 2 + sgn * 200 * RS + ox, 48 * RS + oy), fill=col, width=int(2 * RS))
            d.polygon([(w / 2 + sgn * 205 * RS + ox, 48 * RS + oy - 5 * RS), (w / 2 + sgn * 215 * RS + ox, 48 * RS + oy), (w / 2 + sgn * 205 * RS + ox, 48 * RS + oy + 5 * RS), (w / 2 + sgn * 195 * RS + ox, 48 * RS + oy)], fill=col)
        d.text((w / 2 + ox, 118 * RS + oy), 'BURIED VAULT', font=big, fill=col, anchor='mm')
        layers[name] = im
    # gold inlay only inside the carved area, slightly inset
    gold = layers['gold']
    inset = layers['cut'].getchannel('A').filter(ImageFilter.MinFilter(5))
    ga = np.minimum(np.asarray(gold.getchannel('A')), np.asarray(inset))
    g = np.asarray(gold).copy(); g[..., 3] = ga
    n = fbm(h, w, 20, 55, 3)
    g[..., :3] = np.clip(g[..., :3] * (0.8 + 0.4 * n[..., None]), 0, 255)
    logo = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    logo.alpha_composite(layers['hi']); logo.alpha_composite(layers['cut']); logo.alpha_composite(Image.fromarray(g, 'RGBA'))
    glow = Image.fromarray(g, 'RGBA').filter(ImageFilter.GaussianBlur(6 * RS))
    return (logo, glow), Image.fromarray(g, 'RGBA')


# ------------------------------------------------------------------ frame render
def radial(lw, lh, cx, cy, rad, sx=1.0, sy=1.0):
    yy, xx = np.mgrid[0:lh, 0:lw].astype(np.float32)
    d2 = ((xx - cx) / (rad * sx)) ** 2 + ((yy - cy) / (rad * sy)) ** 2
    return 1 / (1 + d2 * 3.0) * np.clip(1.4 - d2 * 0.35, 0, 1)


def render_frame(fi):
    t = fi / FPS
    cam = Cam(t)
    frame = Image.new('RGBA', (OW, OH), (0, 0, 0, 255))
    op = door_open(t)
    for name in ('sky', 'dunes', 'cback'):
        cam.draw_layer(frame, A[name], DEPTH[name])
    # chest (before props so coins in front overlap its base)
    a = lid_angle(t)
    chest, anc = chest_sprite(a)
    cam.draw_sprite(frame, chest, anc, CHEST, DEPTH['cprops'])
    cam.draw_layer(frame, A['cprops'], DEPTH['cprops'])
    # explorer
    ex, ph, arm, walking, raised = explorer_state(t)
    lantern_s = None
    if t > 7.0:
        spr, anc, lan = explorer_sprite(t, ph, arm, walking)
        feet = (ex, 1318)
        spr = spr.resize((int(spr.width * EXS), int(spr.height * EXS)), Image.LANCZOS)
        cam.draw_sprite(frame, spr, (anc[0] * EXS, anc[1] * EXS), feet, DEPTH['explorer'])
        lantern_w = (feet[0] + lan[0] * EXS, feet[1] + lan[1] * EXS)
        lantern_s, _ = cam.to_screen(lantern_w, DEPTH['explorer'])
    # door leaves slide into the wall
    jit = 1.2 * math.sin(t * 37) * (0 < op < 1)
    for side, key in ((-1, 'leafL'), (1, 'leafR')):
        base_x = (540 - 262 - 20) if side < 0 else (540 - 20)
        lx = base_x + side * 262 * op + jit
        cam.draw_sprite(frame, A[key], (0, 0), (lx, DSPR - 260 - 20), DEPTH['leaves'])
    cam.draw_layer(frame, A['wall'], DEPTH['wall'])
    # logo stamp
    stamp = ss(seg(t, 15.05, 15.3))
    if stamp > 0:
        logo, glow = A['logo']
        sc = 1 + 0.06 * (1 - stamp)
        lg = logo if stamp >= 1 else logo.resize((int(logo.width * sc), int(logo.height * sc)), Image.LANCZOS)
        if stamp < 1:
            arr = np.asarray(lg).copy(); arr[..., 3] = (arr[..., 3] * stamp).astype(np.uint8); lg = Image.fromarray(arr, 'RGBA')
        pc = ((PLQ[0] + PLQ[2]) / 2, (PLQ[1] + PLQ[3]) / 2)
        cam.draw_sprite(frame, lg, (lg.width / RS / 2, lg.height / RS / 2), pc, 1.0)
    cam.draw_layer(frame, A['sand'], DEPTH['sand'])

    # -------- lighting (quarter-res light map)
    lw, lh = A['lw'], A['lh']
    q = 4.0
    amb = np.array([0.30, 0.28, 0.46], np.float32)
    light = np.broadcast_to(amb, (lh, lw, 3)).copy()
    for i, (tx, ty) in enumerate(TORCHES):
        s, zd = cam.to_screen((tx, ty - 30), 1.0)
        fl = 1 + 0.12 * math.sin(t * 8.3 + i * 2) + 0.07 * math.sin(t * 17.9 + i) + 0.05 * math.sin(t * 3.1 + i * 5)
        light += radial(lw, lh, s[0] / q, s[1] / q, 280 * zd / q)[..., None] * np.array([1.05, 0.58, 0.26]) * 1.2 * fl
    # chamber glow spilling out of the doorway
    s, zd = cam.to_screen((540, 1110), DEPTH['cprops'])
    g = op * (0.9 + 0.1 * math.sin(t * 2.2))
    light += radial(lw, lh, s[0] / q, s[1] / q, 330 * zd / q, 1.0, 1.1)[..., None] * np.array([1.0, 0.72, 0.34]) * 0.95 * g
    # chest burst
    burst = ss(seg(t, 12.75, 13.3))
    if burst > 0:
        s, zd = cam.to_screen((CHEST[0], CHEST[1] - 110), DEPTH['cprops'])
        k = burst * (1 + 0.8 * math.exp(-max(0, t - 13.1) * 3))
        light += radial(lw, lh, s[0] / q, s[1] / q, 200 * zd / q)[..., None] * np.array([1.1, 0.85, 0.45]) * 0.9 * k
    if t > 15.0:
        s, zd = cam.to_screen(((PLQ[0] + PLQ[2]) / 2, (PLQ[1] + PLQ[3]) / 2), 1.0)
        k = ss(seg(t, 15.05, 15.3)) * (0.75 + 0.6 * math.exp(-max(0, t - 15.3) * 1.2))
        light += radial(lw, lh, s[0] / q, s[1] / q, 380 * zd / q, 1.0, 0.5)[..., None] * np.array([1.0, 0.75, 0.4]) * k
    # lantern with shadow casting
    if lantern_s is not None:
        li = lantern_intensity(t)
        zd = cam.of(DEPTH['explorer'])[1]
        lx, ly = lantern_s / q
        ll = radial(lw, lh, lx, ly, 200 * zd / q) * 1.4
        # beam pool follows arm direction
        _, _, arm, _, raised = explorer_state(t)
        tgt = (lx + math.cos(arm + math.radians(58)) * 150 * zd / q * raised + 60 * zd / q * raised, ly + 70 * zd / q * raised)
        ll = ll + radial(lw, lh, tgt[0], tgt[1], 190 * zd / q, 1.2, 0.8) * 0.8 * raised
        # long shadows from chest + coin piles, stretched away from the lantern along the floor
        smask = Image.new('L', (lw, lh), 0); sd = ImageDraw.Draw(smask)
        Lw = np.array(lantern_w)
        occluders = [((CHEST[0], CHEST[1] - 6), 90, 150 + 60 * (lid_angle(t) > 0.5)), ((380, 1275), 95, 62), ((800, 1265), 85, 52)]
        for (ox, base), half, hgt in occluders:
            Lh = max(40.0, base - Lw[1])
            poly = []
            for (px, ph_) in [(ox - half, 0), (ox - half * 0.8, hgt * 0.8), (ox, hgt), (ox + half * 0.8, hgt * 0.8), (ox + half, 0)]:
                shx = (px - Lw[0]) / Lh * 1.6
                poly.append((px + ph_ * shx * 1.3, base - ph_ * 0.22 - 4))
            base_pts = [(ox + half, base), (ox - half, base)]
            pts = [cam.to_screen(p, DEPTH['cprops'])[0] / q for p in poly + base_pts]
            sd.polygon([tuple(p) for p in pts], fill=200)
        sm = np.asarray(smask.filter(ImageFilter.GaussianBlur(3)), np.float32) / 255 * raised
        ll = ll * (1 - 0.85 * sm)
        light += ll[..., None] * np.array([1.05, 0.78, 0.45]) * li * 1.2
    light = np.asarray(Image.fromarray(np.clip(light * 80, 0, 255).astype(np.uint8)).resize((OW, OH), Image.BILINEAR), np.float32) / 80

    rgb = np.asarray(frame.convert('RGB'), np.float32) * light

    # -------- emissive overlays
    em = Image.new('RGBA', (OW, OH), (0, 0, 0, 0))
    for i, (tx, ty) in enumerate(TORCHES):
        f = flame_sprite(t, i * 3.7)
        cam.draw_sprite(em, f, (f.width / RS / 2, f.height / RS - 8), (tx, ty), 1.0)
    if lantern_s is not None:
        li = lantern_intensity(t)
        paste(em, glow_blob(int(30 * cam.of(DEPTH['explorer'])[1] * min(1.6, li)), (255, 200, 110), 0.55), lantern_s[0] - 60 * cam.of(DEPTH['explorer'])[1] * min(1.6, li), lantern_s[1] - 60 * cam.of(DEPTH['explorer'])[1] * min(1.6, li))
    if burst > 0:   # shaft of gold light + rising sparkles
        s, zd = cam.to_screen((CHEST[0], CHEST[1] - 90), DEPTH['cprops'])
        shaft = Image.new('RGBA', (int(260 * zd), int(420 * zd)), (0, 0, 0, 0))
        sa = np.zeros((shaft.height, shaft.width), np.float32)
        yy, xx = np.mgrid[0:shaft.height, 0:shaft.width].astype(np.float32)
        yf = yy / shaft.height; half = 0.18 + 0.32 * (1 - yf)
        sa = np.clip(1 - np.abs(xx / shaft.width - 0.5) / half * 2, 0, 1) ** 1.5 * yf ** 1.2
        sarr = np.dstack([np.full_like(sa, 255), np.full_like(sa, 205), np.full_like(sa, 110), sa * 150 * burst * (0.85 + 0.15 * math.sin(t * 5))]).astype(np.uint8)
        paste(em, Image.fromarray(sarr, 'RGBA'), s[0] - shaft.width / 2, s[1] - shaft.height)
        r = np.random.default_rng(4)
        dd = ImageDraw.Draw(em)
        for k in range(40):
            born = 12.8 + r.uniform(0, 6.5)
            age = t - born
            if age < 0 or age > 2.5: continue
            px = CHEST[0] + r.uniform(-70, 70) + 12 * math.sin(age * 2 + k)
            py = CHEST[1] - 95 - age * r.uniform(40, 80)
            sp, zd2 = cam.to_screen((px, py), DEPTH['cprops'])
            al = int(255 * math.sin(math.pi * age / 2.5))
            rr = 2.2 * zd2
            dd.line((sp[0] - rr * 2.5, sp[1], sp[0] + rr * 2.5, sp[1]), fill=(255, 230, 160, al // 2))
            dd.line((sp[0], sp[1] - rr * 2.5, sp[0], sp[1] + rr * 2.5), fill=(255, 230, 160, al // 2))
            dd.ellipse((sp[0] - rr, sp[1] - rr, sp[0] + rr, sp[1] + rr), fill=(255, 236, 180, al))
    # coin glints under the lantern
    if lantern_s is not None and raised > 0:
        dd = ImageDraw.Draw(em)
        for (gx, gy, ph) in A['coinglints']:
            v = max(0.0, math.sin(t * 3 + ph)) ** 8 * raised
            if v < 0.05: continue
            sp, zd2 = cam.to_screen((gx, gy), DEPTH['cprops'])
            rr = 5 * zd2 * v
            dd.line((sp[0] - rr, sp[1], sp[0] + rr, sp[1]), fill=(255, 240, 190, int(230 * v)), width=2)
            dd.line((sp[0], sp[1] - rr, sp[0], sp[1] + rr), fill=(255, 240, 190, int(230 * v)), width=2)
    # logo glow (warm pulse on stamp)
    if stamp > 0:
        logo, glow = A['logo']
        k = 0.35 + 0.9 * math.exp(-max(0, t - 15.3) * 1.5)
        garr = np.asarray(glow).copy(); garr[..., 3] = np.clip(garr[..., 3] * k * stamp, 0, 255).astype(np.uint8)
        pc = ((PLQ[0] + PLQ[2]) / 2, (PLQ[1] + PLQ[3]) / 2)
        cam.draw_sprite(em, Image.fromarray(garr, 'RGBA'), (glow.width / RS / 2, glow.height / RS / 2), pc, 1.0)
        la = np.asarray(A['logo_gold']).copy(); la[..., 3] = (la[..., 3] * 0.8 * stamp).astype(np.uint8)
        cam.draw_sprite(em, Image.fromarray(la, 'RGBA'), (glow.width / RS / 2, glow.height / RS / 2), pc, 1.0)
    dust(em, cam, t, op)
    ea = np.asarray(em, np.float32)
    al = ea[..., 3:4] / 255
    rgb = rgb * (1 - al) + ea[..., :3] * al

    # -------- grade: indigo shadows, warm highlights, vignette, grain, fades
    lum = rgb.mean(axis=2, keepdims=True) / 255
    rgb = rgb * (1 - lum) * np.array([0.9, 0.92, 1.1], np.float32) + rgb * lum * np.array([1.07, 1.0, 0.88], np.float32)
    rgb *= A['vig']
    rgb += A['grain'][fi % 6]
    fade = min(1.0, ss(t / 1.0), 1 - ss(seg(t, 18.9, 19.9)))
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


def dust(em, cam, t, op):
    dd = ImageDraw.Draw(em)
    r = np.random.default_rng(12)
    # sand pouring off the lintel while the door grinds
    for k in range(260):
        born = r.uniform(1.4, 6.4); x0 = r.uniform(DX0 - 40, DX1 + 40)
        speed = r.uniform(160, 320)
        age = t - born
        if age < 0 or age > 2.2: continue
        near = 1 - abs((x0 - 540) - (op * 262 if x0 > 540 else -op * 262)) / 300
        if r.random() > 0.35 + 0.65 * max(0, near): continue
        y = 640 + 0.5 * 300 * age * age + speed * age * 0.3
        x = x0 + 8 * math.sin(age * 3 + k)
        sp, zd = cam.to_screen((x, y), 1.0)
        a = int(200 * min(1, (2.2 - age) / 0.6))
        rr = r.uniform(0.8, 2.0) * zd
        dd.ellipse((sp[0] - rr, sp[1] - rr, sp[0] + rr, sp[1] + rr), fill=(214, 180, 120, a))
    # puff of grit from the stamped plaque
    for k in range(90):
        age = t - 15.1
        if age < 0 or age > 2.6: break
        ang = r.uniform(0, 2 * math.pi)
        x = r.uniform(PLQ[0], PLQ[2]); y = r.choice([PLQ[1], PLQ[3]])
        v = r.uniform(20, 70)
        px = x + math.cos(ang) * v * age; py = y + math.sin(ang) * v * age * 0.4 + 60 * age * age
        sp, zd = cam.to_screen((px, py), 1.0)
        a = int(180 * max(0, 1 - age / 2.6)); rr = r.uniform(0.8, 2.2) * zd
        dd.ellipse((sp[0] - rr, sp[1] - rr, sp[0] + rr, sp[1] + rr), fill=(200, 176, 150, a))
    # slow motes drifting in the chamber light
    for i, (mx, my, ms, mp) in enumerate(A['motes']):
        x = 280 + mx * 520 + 30 * math.sin(t * 0.3 + mp * 6)
        y = 700 + ((my * 700 - t * (8 + ms * 12)) % 700)
        sp, zd = cam.to_screen((x, y), DEPTH['cprops'])
        a = int((60 + 120 * ms) * max(op, 0.0) * (0.5 + 0.5 * math.sin(t * 1.3 + mp * 9)))
        rr = (0.8 + ms * 1.6) * zd
        dd.ellipse((sp[0] - rr, sp[1] - rr, sp[0] + rr, sp[1] + rr), fill=(255, 220, 160, a))


# ------------------------------------------------------------------ audio
def audio():
    from scipy.signal import butter, sosfilt
    n = int(SR * DUR); t = np.arange(n) / SR
    r = np.random.default_rng(3)
    def bp(x, lo, hi, order=2):
        return sosfilt(butter(order, [lo, hi], 'bandpass', fs=SR, output='sos'), x)
    def lp(x, f, order=2):
        return sosfilt(butter(order, f, 'lowpass', fs=SR, output='sos'), x)
    def env(a, b, att=0.05, rel=0.3):
        e = np.clip((t - a) / att, 0, 1) * np.clip((b - t) / rel, 0, 1); return e
    out = np.zeros((n, 2))
    # room tone / desert wind
    wind = lp(np.cumsum(r.normal(0, 1, n)) * 0.002, 400)
    wind -= lp(wind, 20)
    wmod = 0.6 + 0.4 * np.sin(2 * np.pi * t / 7.3) ** 2
    out += (wind * wmod * 0.9)[:, None] * np.array([1, 0.85])
    # torch crackle
    cr = np.zeros(n); idx = r.choice(n, 260, replace=False); cr[idx] = r.uniform(-1, 1, 260)
    cr = bp(cr, 1500, 6000) * 0.5
    out += cr[:, None] * np.array([0.7, 1.0]) * (1 - 0.6 * env(8, 13.5, 1.5, 1.5))[:, None]
    # stone grinding 1.6–5.8
    g = r.normal(0, 1, n)
    grind = bp(g, 60, 420, 3) * (0.55 + 0.45 * np.abs(np.sin(2 * np.pi * 5.3 * t + 3 * np.sin(2 * np.pi * 0.7 * t))))
    grind += bp(g, 900, 2200) * 0.15 * np.abs(np.sin(2 * np.pi * 11 * t))
    rumble = np.sin(2 * np.pi * 38 * t + 2 * np.sin(2 * np.pi * 1.3 * t)) * 0.25
    out += ((grind * 1.6 + rumble) * env(1.6, 5.9, 0.5, 0.6))[:, None]
    # falling dust / sand trickle
    tr = bp(r.normal(0, 1, n), 3000, 9000) * (0.5 + 0.5 * (r.random(n) > 0.97))
    out += (tr * 0.12 * (env(1.6, 6.6, 0.4, 1.0) + 0.8 * env(15.1, 17.3, 0.02, 1.8)))[:, None] * np.array([1, 0.8])
    # soft footsteps on sand 7.3–9.9
    for k, ts in enumerate(np.arange(7.45, 9.9, 1 / 1.7)):
        i0 = int(ts * SR); L = int(0.18 * SR)
        e = np.exp(-np.arange(L) / (0.04 * SR))
        s = lp(r.normal(0, 1, L), 700) * e * 0.9
        pan = np.array([0.9, 0.6]) if k % 2 else np.array([0.7, 0.8])
        out[i0:i0 + L] += s[:, None] * pan
    # lantern raise: faint metal clink
    i0 = int(10.4 * SR); L = int(0.6 * SR); tt = np.arange(L) / SR
    out[i0:i0 + L] += ((np.sin(2 * np.pi * 2350 * tt) + 0.6 * np.sin(2 * np.pi * 3710 * tt)) * np.exp(-tt * 9) * 0.05)[:, None]
    # chest creak 12.45–13.25
    i0 = int(12.45 * SR); L = int(0.85 * SR); tt = np.arange(L) / SR
    f = 140 + 90 * tt + 40 * np.sin(2 * np.pi * 3 * tt)
    ph = 2 * np.pi * np.cumsum(f) / SR
    saw = 2 * ((ph / (2 * np.pi)) % 1) - 1
    stick = (np.sin(2 * np.pi * 26 * tt) > 0.2).astype(float)
    creak = bp(saw * stick, 300, 2400) * np.sin(np.pi * tt / tt[-1]) * 0.45
    out[i0:i0 + L] += creak[:, None]
    # magical chime 12.95
    i0 = int(12.95 * SR); L = int(4.5 * SR); tt = np.arange(L) / SR
    chime = np.zeros(L)
    for k, (fr, amp, dec) in enumerate([(1318.5, 1, 1.4), (1975.5, 0.7, 1.1), (2637, 0.5, 0.9), (3520, 0.3, 0.7), (1567.98, 0.5, 1.6)]):
        st = int(k * 0.06 * SR)
        seg_ = np.zeros(L); seg_[st:] = np.sin(2 * np.pi * fr * tt[:L - st] * (1 + 0.0008 * np.sin(2 * np.pi * 5 * tt[:L - st]))) * np.exp(-tt[:L - st] * dec) * amp
        chime += seg_
    shimmer = bp(r.normal(0, 1, L), 6000, 12000) * np.exp(-tt * 1.2) * 0.25
    out[i0:i0 + L] += ((chime * 0.12 + shimmer * 0.2)[:, None]) * np.array([0.9, 1.0])
    # stamp thud 15.1
    i0 = int(15.1 * SR); L = int(1.6 * SR); tt = np.arange(L) / SR
    thud = np.sin(2 * np.pi * (70 * np.exp(-tt * 3) + 38) * tt) * np.exp(-tt * 4.5) * 0.9 + lp(r.normal(0, 1, L), 900) * np.exp(-tt * 14) * 0.6
    out[i0:i0 + L] += thud[:, None]
    # heartbeat pulse 18.55 / 18.85
    for ts, a in ((18.55, 0.8), (18.85, 0.55)):
        i0 = int(ts * SR); L = int(0.5 * SR); tt = np.arange(L) / SR
        hb = np.sin(2 * np.pi * (55 * np.exp(-tt * 6) + 42) * tt) * np.exp(-tt * 12) * a
        out[i0:i0 + L] += hb[:, None]
    # fades
    fade = np.clip(t / 1.0, 0, 1) * np.clip((DUR - t) / 1.2, 0, 1)
    out *= fade[:, None]
    out /= np.abs(out).max() + 1e-9
    out *= 0.85
    return (out * 32767).astype(np.int16)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'the_buried_vault.mp4'
    preview = '--preview' in sys.argv
    print('building assets...', flush=True)
    build_assets()
    if preview:
        os.makedirs('preview', exist_ok=True)
        times = [0.8, 3.5, 6.5, 8.5, 10.5, 11.8, 13.2, 14.3, 15.2, 16.5, 18.6, 19.4]
        with Pool(4) as pool:
            res = pool.map(render_frame, [int(tt * FPS) for tt in times])
        ims = [Image.frombytes('RGB', (OW, OH), b).resize((480, 270)) for b in res]
        sheet = Image.new('RGB', (480 * 4, 270 * 3))
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % 4) * 480, (i // 4) * 270))
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
