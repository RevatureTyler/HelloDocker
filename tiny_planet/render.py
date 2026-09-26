"""Tiny planet orbiting the sun — procedural stop-motion (Coraline-inspired) animation.

Renders 20 s of 1080x1920 (9:16) video with numpy + Pillow and encodes it with ffmpeg.
Usage: python render.py [out.mp4]
"""
import math, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import imageio_ffmpeg

W, H = 1080, 1920
FPS, SECONDS = 24, 20
NFRAMES = FPS * SECONDS
HOLD = 2                                  # animate "on twos" like stop-motion
TILT = math.radians(-17)                  # orbital plane tilt on screen
K = 0.36                                  # orbit ellipse squash (camera elevation)
CX, CY = 540, 1000                        # sun position
FOCAL = 2400.0
MARGIN = 60
rng0 = np.random.default_rng(7)


def hexc(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32)


def vnoise(h, w, cell, seed, wrap=False):
    r = np.random.default_rng(seed)
    gh, gw = h // cell + 3, w // cell + 3
    g = r.random((gh, gw)).astype(np.float32)
    if wrap:
        gw = max(2, w // cell)
        g = r.random((gh, gw)).astype(np.float32)
        g = np.concatenate([g, g[:, :3]], 1)
        im = Image.fromarray(g, 'F').resize((int(w * (gw + 3) / gw), gh * cell), Image.BICUBIC)
        return np.asarray(im)[:h, :w]
    im = Image.fromarray(g, 'F').resize((gw * cell, gh * cell), Image.BICUBIC)
    return np.asarray(im)[:h, :w]


def fbm(h, w, cell, seed, octaves=4, wrap=False):
    out = np.zeros((h, w), np.float32); amp = 1.0; tot = 0
    for o in range(octaves):
        c = max(2, cell >> o)
        out += amp * vnoise(h, w, c, seed + o * 31, wrap); tot += amp; amp *= 0.5
    return out / tot


def lerp(a, b, t):
    t = np.asarray(t, np.float32)[..., None]
    return a * (1 - t) + b * t


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def paste(canvas, sprite, cx, cy):
    """alpha-composite sprite centered at (cx, cy), clipped to canvas."""
    sw, sh = sprite.size
    x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
    cw, ch = canvas.size
    l, t = max(0, -x0), max(0, -y0)
    r, b = min(sw, cw - x0), min(sh, ch - y0)
    if r <= l or b <= t:
        return
    canvas.alpha_composite(sprite.crop((l, t, r, b)), dest=(x0 + l, y0 + t))


def spiral(draw, x, y, size, turns, col, width):
    pts = []
    n = int(40 * turns)
    for i in range(n + 1):
        a = i / n * turns * 2 * math.pi
        rr = size * i / n
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    draw.line(pts, fill=col, width=width, joint='curve')


# ---------------------------------------------------------------- background
def make_background():
    S = 2
    bw, bh = W + 2 * MARGIN, H + 2 * MARGIN
    y = np.linspace(0, 1, bh, dtype=np.float32)[:, None]
    top, mid, bot = hexc('#070720'), hexc('#18214d'), hexc('#2a1440')
    col = np.where((y < 0.5)[..., None], lerp(top, mid, y / 0.5), lerp(mid, bot, (y - 0.5) / 0.5))
    col = np.broadcast_to(col, (bh, bw, 3)).copy()
    # painted nebula band along the orbital plane
    yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
    xr = (xx - CX - MARGIN); yr = (yy - CY - MARGIN)
    band = -xr * math.sin(TILT) + yr * math.cos(TILT)
    n1 = fbm(bh, bw, 256, 11, 5); n2 = fbm(bh, bw, 180, 23, 5)
    bandmask = np.exp(-(band / 520) ** 2)
    teal = smooth(0.45, 0.75, n1) * bandmask * 0.55
    plum = smooth(0.5, 0.8, n2) * (0.35 + 0.65 * bandmask) * 0.5
    col = lerp(col, hexc('#2c6d78'), teal)
    col = lerp(col, hexc('#7a2d63'), plum)
    swirl = smooth(0.6, 0.9, fbm(bh, bw, 90, 41, 4)) * 0.18
    col = lerp(col, hexc('#c77a4a'), swirl * bandmask)
    # felt / paper fibre texture
    col += (fbm(bh, bw, 4, 5, 2) - 0.5)[..., None] * 22
    col += (rng0.random((bh, bw)) - 0.5)[..., None].astype(np.float32) * 8
    img = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8)).convert('RGBA')

    # hand-drawn stars at 2x for clean edges
    ov = Image.new('RGBA', (bw * S, bh * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    r = np.random.default_rng(3)
    for _ in range(420):
        x, y = r.random() * bw * S, r.random() * bh * S
        s = r.choice([1, 1, 1.5, 2, 3]) * S * 0.8
        a = int(r.uniform(60, 190))
        c = [(245, 230, 190, a), (190, 215, 235, a), (240, 190, 170, a)][r.integers(3)]
        d.ellipse((x - s, y - s, x + s, y + s), fill=c)
    for _ in range(34):  # Coraline-style curly spiral stars
        x, y = r.random() * bw * S, r.random() * bh * S
        spiral(d, x, y, r.uniform(8, 20) * S, r.uniform(1.6, 2.6), (240, 220, 170, int(r.uniform(90, 170))), 2 * S)
    ov = ov.resize((bw, bh), Image.LANCZOS)
    img.alpha_composite(ov)
    return img


# ---------------------------------------------------------------- planet textures (equirect)
TW, TH = 512, 256


def tex_grid():
    lon = np.linspace(0, 1, TW, endpoint=False, dtype=np.float32)[None, :]
    lat = np.linspace(-1, 1, TH, dtype=np.float32)[:, None]
    return lon, lat


def banded(colors, seed, freq=6, warp=0.12, spot=None):
    lon, lat = tex_grid()
    n = fbm(TH, TW, 64, seed, 4, wrap=True)
    v = (lat + warp * (n - 0.5)) * freq
    idx = (np.sin(v * math.pi) * 0.5 + 0.5) * (len(colors) - 1)
    lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, len(colors) - 1)
    f = smooth(0.25, 0.75, idx - lo)
    C = np.stack([hexc(c) for c in colors])
    tex = lerp(C[lo], C[hi], f)
    if spot:
        sx, sy, sr, sc = spot
        dx = np.minimum(abs(lon - sx), 1 - abs(lon - sx)) * 2
        dist = np.sqrt((dx / sr) ** 2 + ((lat - sy) / (sr * 0.6)) ** 2)
        tex = lerp(tex, hexc(sc), smooth(1.0, 0.7, dist))
        tex = lerp(tex, hexc(sc) * 0.7, smooth(0.5, 0.2, dist) * 0.6)
    return tex


def blotchy(base, dark, seed, light=None, cap=None, thr=(0.45, 0.6)):
    lon, lat = tex_grid()
    n = fbm(TH, TW, 48, seed, 5, wrap=True)
    tex = lerp(np.broadcast_to(hexc(base), (TH, TW, 3)), hexc(dark), smooth(*thr, n))
    if light:
        tex = lerp(tex, hexc(light), smooth(0.35, 0.2, n) * 0.8)
    if cap:
        tex = lerp(tex, hexc(cap), smooth(0.78, 0.86, np.abs(lat) + 0.05 * (n - 0.5)))
    return tex


def hero_tex():
    lon, lat = tex_grid()
    n = fbm(TH, TW, 56, 101, 5, wrap=True)
    ocean = lerp(np.broadcast_to(hexc('#1f5f73'), (TH, TW, 3)), hexc('#3a93a0'), smooth(0.3, 0.6, n))
    land = smooth(0.52, 0.56, n)
    grass = lerp(np.broadcast_to(hexc('#5f8b3e'), (TH, TW, 3)), hexc('#a9b957'), smooth(0.6, 0.75, n))
    tex = lerp(ocean, grass, land)
    shore = smooth(0.5, 0.52, n) * (1 - land)
    tex = lerp(tex, hexc('#e8d6a0'), shore * 0.9)
    tex = lerp(tex, hexc('#f1ead6'), smooth(0.8, 0.86, np.abs(lat) + 0.06 * (n - 0.5)))
    return tex


def clayify(tex, seed):
    """fingerprint-y clay grain."""
    g = fbm(TH, TW, 6, seed, 2, wrap=True) - 0.5
    return np.clip(tex * (1 + 0.16 * g[..., None]), 0, 255)


# ---------------------------------------------------------------- sphere render
def render_sphere(tex, r, spin, light, axis_tilt=0.0, rim=(120, 200, 230), ambient=0.22):
    """tex equirect, r px radius, light: 3-vector pointing toward light (screen x right, y down, z toward viewer)."""
    size = int(math.ceil(r)) * 2 + 4
    c = size / 2
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    u = (xx + 0.5 - c) / r; v = (yy + 0.5 - c) / r
    d2 = u * u + v * v
    alpha = np.clip((1 - np.sqrt(d2)) * r + 0.5, 0, 1)
    w = np.sqrt(np.clip(1 - d2, 0, 1))
    ct, st = math.cos(axis_tilt), math.sin(axis_tilt)
    ua, va = u * ct + v * st, -u * st + v * ct
    lat = np.arcsin(np.clip(va, -1, 1))
    lon = np.arctan2(ua, w) / (2 * math.pi) + spin
    ti = np.clip(((lat / math.pi + 0.5) * (TH - 1)).astype(int), 0, TH - 1)
    tj = ((lon % 1.0) * TW).astype(int) % TW
    col = tex[ti, tj]
    L = np.asarray(light, np.float32); L = L / (np.linalg.norm(L) + 1e-6)
    ndl = u * L[0] + v * L[1] + w * L[2]
    diff = np.clip((ndl + 0.25) / 1.25, 0, 1) ** 1.3
    shade = ambient + (1 - ambient) * diff
    cool = np.array([0.55, 0.62, 0.95], np.float32)
    warm = np.array([1.08, 0.98, 0.86], np.float32)
    tint = cool * (1 - diff[..., None]) + warm * diff[..., None]
    out = col * shade[..., None] * tint
    rimf = np.clip(1 - w, 0, 1) ** 3 * 0.55
    out = out * (1 - rimf[..., None]) + np.array(rim, np.float32) * rimf[..., None] * 0.9
    spec = np.clip(ndl, 0, 1) ** 30 * 0.35
    out += spec[..., None] * 255
    rgba = np.dstack([np.clip(out, 0, 255), alpha * 255]).astype(np.uint8)
    return Image.fromarray(rgba, 'RGBA'), diff


# ---------------------------------------------------------------- sun
def make_sun_parts(R):
    S = 3
    size = int(R * 3.6)
    parts = {}
    for name, n, r0, amp, colr, rot in [('outer', 13, 1.06, 0.62, (206, 84, 38, 255), 0),
                                        ('inner', 17, 1.02, 0.36, (240, 160, 58, 255), 0.5)]:
        im = Image.new('RGBA', (size * S, size * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        c = size * S / 2
        pts = []
        for i in range(720):
            a = i / 720 * 2 * math.pi
            k = abs(math.sin(n * a / 2 + rot))
            flick = 0.08 * math.sin(3 * n * a)
            rr = R * S * (r0 + amp * (1 - k) ** 2.2 + flick * (1 - k))
            # curl the tips
            a2 = a + 0.06 * (1 - k) ** 2
            pts.append((c + rr * math.cos(a2), c + rr * math.sin(a2)))
        d.polygon(pts, fill=colr)
        # stitched cream outline
        for i in range(0, 720, 6):
            x, y = pts[i]
            d.ellipse((x - 2.2 * S, y - 2.2 * S, x + 2.2 * S, y + 2.2 * S), fill=(255, 226, 170, 120))
        parts[name] = im.resize((size, size), Image.LANCZOS)
    # core: clay ball with engraved swirl
    cs = int(R * 2) + 4
    yy, xx = np.mgrid[0:cs, 0:cs].astype(np.float32)
    u = (xx - cs / 2) / R; v = (yy - cs / 2) / R
    rr = np.sqrt(u * u + v * v); ang = np.arctan2(v, u)
    base = lerp(hexc('#fff0b0'), hexc('#f5a23c'), smooth(0.0, 0.85, rr))
    base = lerp(base, hexc('#d9582a'), smooth(0.8, 1.0, rr))
    sw = np.sin(ang * 1 + rr * 16) * 0.5 + 0.5
    base = lerp(base, hexc('#e9782f'), smooth(0.75, 0.95, sw) * smooth(0.15, 0.5, rr) * 0.55)
    g = fbm(cs, cs, 5, 77, 2) - 0.5
    base *= (1 + 0.12 * g)[..., None]
    alpha = np.clip((1 - rr) * R + 0.5, 0, 1)
    parts['core'] = Image.fromarray(np.dstack([np.clip(base, 0, 255), alpha * 255]).astype(np.uint8), 'RGBA')
    # glow
    gs = int(R * 7)
    yy, xx = np.mgrid[0:gs, 0:gs].astype(np.float32)
    rr = np.sqrt((xx - gs / 2) ** 2 + (yy - gs / 2) ** 2) / R
    a = np.clip(0.55 * np.exp(-((rr - 0.8) / 1.1) ** 2) + 0.25 * np.exp(-rr / 2.2), 0, 1)
    glow = np.dstack([np.full_like(a, 255), np.full_like(a, 150), np.full_like(a, 60), a * 170])
    parts['glow'] = Image.fromarray(glow.astype(np.uint8), 'RGBA')
    return parts


# ---------------------------------------------------------------- props on hero planet
def make_props(r):
    """Crooked pink house + curly tree, drawn upright; base line at sprite center."""
    S = 4
    size = int(r * 3.2)
    im = Image.new('RGBA', (size * S, size * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = size * S / 2
    u = r * S / 78.0   # prop scale unit
    by = c - r * S * 0.93   # surface top
    # house (slightly crooked)
    hx = c - 8 * u
    body = [(hx - 17 * u, by + 6 * u), (hx + 17 * u, by + 6 * u), (hx + 16 * u, by - 28 * u), (hx - 18 * u, by - 26 * u)]
    d.polygon(body, fill=(214, 128, 150, 255), outline=(90, 40, 60, 255))
    roof = [(hx - 25 * u, by - 24 * u), (hx + 23 * u, by - 27 * u), (hx - 2 * u, by - 56 * u)]
    d.polygon(roof, fill=(78, 48, 96, 255), outline=(40, 22, 50, 255))
    # tower / chimney
    d.polygon([(hx + 7 * u, by - 40 * u), (hx + 13 * u, by - 41 * u), (hx + 13 * u, by - 52 * u), (hx + 7 * u, by - 51 * u)], fill=(150, 80, 110, 255))
    # glowing windows
    for wx, wy in [(-9, -18), (6, -19)]:
        d.rectangle((hx + (wx - 4) * u, by + (wy - 5) * u, hx + (wx + 4) * u, by + (wy + 4) * u), fill=(255, 214, 110, 255), outline=(90, 40, 60, 255))
    d.ellipse((hx - 5 * u, by - 44 * u, hx + 2 * u, by - 37 * u), fill=(255, 214, 110, 255))
    d.rectangle((hx - 3 * u, by - 6 * u, hx + 3 * u, by + 6 * u), fill=(70, 36, 50, 255))
    # curly tree
    tx = c + 26 * u
    trunk = [(tx + 2 * u * math.sin(i / 6), by + 4 * u - i * 1.4 * u) for i in range(26)]
    d.line(trunk, fill=(70, 42, 40, 255), width=int(4 * u))
    top = trunk[-1]
    d.ellipse((top[0] - 13 * u, top[1] - 13 * u, top[0] + 13 * u, top[1] + 13 * u), fill=(58, 110, 70, 255))
    spiral(d, top[0], top[1], 11 * u, 2.2, (150, 190, 90, 255), int(2.4 * u))
    im = im.resize((size, size), Image.LANCZOS)
    return im


def make_button(r):
    """Coraline's button: moon with four holes."""
    S = 4
    size = int(r * 2 + 4)
    im = Image.new('RGBA', (size * S, size * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = size * S / 2; R = r * S
    d.ellipse((c - R, c - R, c + R, c + R), fill=(38, 34, 52, 255))
    d.ellipse((c - R * 0.82, c - R * 0.82, c + R * 0.82, c + R * 0.82), fill=(58, 52, 76, 255))
    d.ellipse((c - R * 0.72, c - R * 0.9, c + R * 0.2, c - R * 0.1), fill=(90, 84, 112, 120))
    h = R * 0.3; hr = R * 0.14
    for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        d.ellipse((c + sx * h - hr, c + sy * h - hr, c + sx * h + hr, c + sy * h + hr), fill=(12, 10, 18, 255))
    d.line((c - h, c - h, c + h, c + h), fill=(220, 70, 60, 255), width=int(R * 0.12))
    d.line((c - h, c + h, c + h, c - h), fill=(220, 70, 60, 255), width=int(R * 0.12))
    return im.resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------- orbits
def orbit_point(R, th, ox=0, oy=0):
    x0, y0 = R * math.cos(th), R * K * math.sin(th)
    x = x0 * math.cos(TILT) - y0 * math.sin(TILT)
    y = x0 * math.sin(TILT) + y0 * math.cos(TILT)
    z = R * math.sin(th) * math.sqrt(1 - K * K)
    return x, y, z


def make_orbit_layers(radii):
    S = 2
    ow, oh = (W + 2 * MARGIN) * S, (H + 2 * MARGIN) * S
    back = Image.new('RGBA', (ow, oh), (0, 0, 0, 0)); front = back.copy()
    db, df = ImageDraw.Draw(back), ImageDraw.Draw(front)
    for R in radii:
        n = int(R * 0.9)
        for i in range(n):
            if i % 2:
                continue
            t0, t1 = i / n * 2 * math.pi, (i + 1.1) / n * 2 * math.pi
            x0, y0, _ = orbit_point(R, t0); x1, y1, _ = orbit_point(R, t1)
            p = [((CX + MARGIN + x0) * S, (CY + MARGIN + y0) * S), ((CX + MARGIN + x1) * S, (CY + MARGIN + y1) * S)]
            front_side = math.sin(t0) > 0
            (df if front_side else db).line(p, fill=(242, 222, 180, 110 if front_side else 70), width=int(2.2 * S))
    return back.resize((ow // S, oh // S), Image.LANCZOS), front.resize((ow // S, oh // S), Image.LANCZOS)


# ---------------------------------------------------------------- scene set-up
PLANETS = [
    # name, orbit R, radius, revs in 20s, phase, spin revs, tex, axis tilt, ring
    dict(name='lilac', R=215, r=17, revs=3, ph=2.2, spin=4, tex=clayify(blotchy('#a497c4', '#6e6190', 1, light='#d8cfe8'), 2)),
    dict(name='mustard', R=300, r=27, revs=2, ph=4.4, spin=3, tex=clayify(banded(['#efd08a', '#d9a441', '#b8742f', '#e3b862'], 3, freq=5), 4)),
    dict(name='hero', R=405, r=66, revs=1, ph=0.35, spin=2, tex=clayify(hero_tex(), 5)),
    dict(name='rust', R=510, r=25, revs=0.75, ph=3.3, spin=3, tex=clayify(blotchy('#b5522f', '#6f2d20', 6, light='#d27a4f', cap='#efe0c6'), 7)),
    dict(name='jupiter', R=640, r=74, revs=0.45, ph=5.2, spin=1.5, tex=clayify(banded(['#f0d3a0', '#d98c4a', '#8c4a2f', '#e6b27a', '#b86a3d'], 8, freq=9, warp=0.2, spot=(0.3, 0.3, 0.12, '#b0382a')), 9)),
    dict(name='saturn', R=790, r=52, revs=0.3, ph=1.2, spin=1.5, tex=clayify(banded(['#e3cf92', '#c9a865', '#a88b52', '#ecdcae'], 10, freq=7), 11), ring=True),
    dict(name='ice', R=950, r=40, revs=0.2, ph=2.6, spin=1, tex=clayify(blotchy('#6fb3b8', '#3b7f8c', 12, light='#c4e6e2', thr=(0.4, 0.7)), 13)),
]


def make_ring_sprites(r, tilt):
    """back and front halves of a striped ring, in the planet's frame, rotated to tilt."""
    S = 3
    size = int(r * 5)
    halves = []
    for half in ('back', 'front'):
        im = Image.new('RGBA', (size * S, size * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        c = size * S / 2
        for rr, col, wdt in [(2.25, (232, 217, 168, 230), 0.22), (1.95, (156, 132, 87, 230), 0.14), (1.7, (214, 190, 130, 220), 0.18), (1.45, (190, 160, 110, 160), 0.1)]:
            rx, ry = r * rr * S, r * rr * S * 0.28
            box = (c - rx, c - ry, c + rx, c + ry)
            if half == 'back':
                d.arc(box, 180, 360, fill=col, width=int(r * wdt * S))
            else:
                d.arc(box, 0, 180, fill=col, width=int(r * wdt * S))
        im = im.resize((size, size), Image.LANCZOS).rotate(math.degrees(-tilt) - 12, resample=Image.BICUBIC)
        halves.append(im)
    return halves


def main(out):
    print('building assets...', flush=True)
    bg = make_background()
    SUN_R = 118
    sun = make_sun_parts(SUN_R)
    orb_back, orb_front = make_orbit_layers([p['R'] for p in PLANETS])
    hero_r = [p for p in PLANETS if p['name'] == 'hero'][0]['r']
    props = make_props(hero_r)
    button = make_button(15)
    rings = make_ring_sprites(52, TILT)
    tex_button_back = None

    # twinkle stars + dust
    r = np.random.default_rng(9)
    twinkles = [(r.random() * W, r.random() * H, r.uniform(1.5, 3.2), r.random() * 6.28) for _ in range(70)]
    dust = [(r.random() * W, r.random() * H, r.uniform(2, 9), r.uniform(-8, 8), r.uniform(-20, -4), r.uniform(0.2, 0.6)) for _ in range(55)]
    blob = {}

    def blob_sprite(sz):
        k = int(round(sz))
        if k not in blob:
            s = k * 4 + 4
            yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
            d = np.sqrt((xx - s / 2) ** 2 + (yy - s / 2) ** 2) / k
            a = np.exp(-d * d * 1.2)
            blob[k] = Image.fromarray(np.dstack([np.full_like(a, 255), np.full_like(a, 225), np.full_like(a, 180), a * 255]).astype(np.uint8), 'RGBA')
        return blob[k]

    # grading / grain / vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vig = 1 - 0.55 * smooth(0.45, 1.25, np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.6)) ** 2))
    vig = vig[..., None].astype(np.float32)
    grains = [(np.random.default_rng(100 + i).normal(0, 7, (H // 2, W // 2)).astype(np.float32)) for i in range(8)]
    grains = [np.asarray(Image.fromarray(g, 'F').resize((W, H), Image.BILINEAR))[..., None] for g in grains]

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                             '-c:v', 'libx264', '-preset', 'slow', '-crf', '21', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
                             '-movflags', '+faststart', '-metadata', 'title=Tiny Planet', out], stdin=subprocess.PIPE)

    pose_img = None
    for f in range(NFRAMES):
        pose = f // HOLD
        if f % HOLD == 0:
            t = pose * HOLD / FPS
            pr = np.random.default_rng(1000 + pose)
            jit = lambda s=0.8: (pr.uniform(-s, s), pr.uniform(-s, s))
            # camera sway (loops over 20 s)
            camx = 14 * math.sin(2 * math.pi * t / 20); camy = 10 * math.sin(4 * math.pi * t / 20)
            ox, oy = CX + camx, CY + camy
            bx, by = int(MARGIN - camx * 0.35), int(MARGIN - camy * 0.35)
            canvas = bg.crop((bx, by, bx + W, by + H))
            d = ImageDraw.Draw(canvas)
            for (x, y, s, ph) in twinkles:
                b = 0.5 + 0.5 * math.sin(ph + t * 2.3 + pr.uniform(-0.4, 0.4))
                a = int(60 + 190 * b); ss = s * (0.7 + 0.5 * b)
                d.line((x - ss * 2.2, y, x + ss * 2.2, y), fill=(255, 238, 200, a // 2), width=1)
                d.line((x, y - ss * 2.2, x, y + ss * 2.2), fill=(255, 238, 200, a // 2), width=1)
                d.ellipse((x - ss * 0.6, y - ss * 0.6, x + ss * 0.6, y + ss * 0.6), fill=(255, 240, 210, a))
            # comet (6.5 s – 11.5 s)
            if 6.5 <= t <= 11.5:
                k = (t - 6.5) / 5
                hx, hy = 1180 - 1400 * k, 180 + 520 * k
                dx, dy = 1400, -520; nrm = math.hypot(dx, dy); dx, dy = dx / nrm, dy / nrm
                for i in range(40, 0, -1):
                    s = 7 * (1 - i / 44)
                    px = hx + dx * i * 7 + pr.uniform(-2, 2); py = hy + dy * i * 7 + math.sin(i * 0.35 + t * 3) * 1.2
                    a = int(210 * (1 - i / 40))
                    d.ellipse((px - s, py - s, px + s, py + s), fill=(250, 220, 170, a))
                paste(canvas, blob_sprite(9), hx, hy)
                d.ellipse((hx - 5, hy - 5, hx + 5, hy + 5), fill=(255, 250, 235, 255))

            # sun glow & back orbits
            pulse = 1 + 0.03 * math.sin(t * 2 * math.pi / 2.5) + pr.uniform(-0.01, 0.01)
            glow = sun['glow']
            gs = int(glow.size[0] * pulse)
            paste(canvas, glow.resize((gs, gs), Image.BILINEAR), ox, oy)
            j = jit(0.6)
            paste(canvas, orb_back, W / 2 + camx + j[0], H / 2 + camy + j[1])

            # planets
            items = []
            for p in PLANETS:
                th = p['ph'] + 2 * math.pi * p['revs'] * t / SECONDS
                x, y, z = orbit_point(p['R'], th)
                sc = FOCAL / (FOCAL - z)
                px, py = ox + x * sc, oy + y * sc
                items.append((z, p, th, px, py, sc))
            items.sort(key=lambda it: it[0])

            def draw_planet(it):
                z, p, th, px, py, sc = it
                rr = p['r'] * sc * (1 + pr.uniform(-0.006, 0.006))
                if px < -rr * 3 or px > W + rr * 3 or py < -rr * 3 or py > H + rr * 3:
                    return
                wx, wy, _ = orbit_point(p['R'], th)
                L = (-wx, -wy, -z * 0.45 + 0.4 * p['R'])  # soften backlight with a front fill
                spin = p['spin'] * t / SECONDS
                jx, jy = jit(0.9)
                if p['name'] == 'hero':
                    wob = math.radians(6 * math.sin(2 * math.pi * t / 5) + pr.uniform(-0.8, 0.8))
                    sph, diff = render_sphere(p['tex'], rr, spin, L, axis_tilt=wob, rim=(140, 220, 235), ambient=0.38)
                    # moon (button) orbit around hero
                    mth = 2 * math.pi * 3 * t / SECONDS + 1.0
                    mx, my = math.cos(mth) * rr * 1.95, math.sin(mth) * rr * 0.55 - math.cos(mth) * rr * 0.25
                    mz = math.sin(mth)
                    bs = button.resize((max(4, int(button.size[0] * sc * (1 + 0.12 * mz))),) * 2, Image.LANCZOS)
                    bs = bs.rotate(math.degrees(mth) * 0.5, resample=Image.BICUBIC)
                    if mz < 0:
                        paste(canvas, bs, px + mx + jx, py + my + jy)
                    # light factor for props: top of planet normal
                    Ln = np.array(L, np.float32); Ln /= np.linalg.norm(Ln) + 1e-6
                    top_n = np.array([math.sin(wob), -math.cos(wob), 0.35]); top_n /= np.linalg.norm(top_n)
                    lf = 0.45 + 0.6 * max(0.0, float(np.dot(Ln, top_n)) * 0.8 + 0.2)
                    pw = props.resize((int(props.size[0] * rr / p['r']),) * 2, Image.LANCZOS)
                    pa = np.asarray(pw).astype(np.float32)
                    pa[..., :3] *= np.array([lf, lf * 0.97, lf * 1.02 if lf < 0.8 else lf * 0.93])
                    # keep the window light glowing
                    win = (pa[..., 0] > 150 * lf) & (pa[..., 1] > 120 * lf) & (pa[..., 2] < 140 * lf)
                    pa[win, :3] = [255, 214, 110]
                    pw = Image.fromarray(np.clip(pa, 0, 255).astype(np.uint8), 'RGBA').rotate(-math.degrees(wob), resample=Image.BICUBIC)
                    paste(canvas, sph, px + jx, py + jy)
                    paste(canvas, pw, px + jx, py + jy)
                    if mz >= 0:
                        paste(canvas, bs, px + mx + jx, py + my + jy)
                    return
                if p.get('ring'):
                    rs = [im.resize((int(im.size[0] * rr / 52),) * 2, Image.LANCZOS) for im in rings]
                    paste(canvas, rs[0], px + jx, py + jy)
                sph, _ = render_sphere(p['tex'], rr, spin, L, axis_tilt=-TILT * 0.8 if p.get('ring') else 0.2)
                paste(canvas, sph, px + jx, py + jy)
                if p.get('ring'):
                    paste(canvas, rs[1], px + jx, py + jy)

            for it in items:
                if it[0] < 0:
                    draw_planet(it)
            # sun petals + core
            j = jit(0.7)
            outer = sun['outer'].rotate(t * 9 + pr.uniform(-0.6, 0.6), resample=Image.BICUBIC)
            inner = sun['inner'].rotate(-t * 14 + pr.uniform(-0.6, 0.6), resample=Image.BICUBIC)
            paste(canvas, outer, ox + j[0], oy + j[1])
            paste(canvas, inner, ox + j[0] * 0.5, oy + j[1] * 0.5)
            core = sun['core'].rotate(t * 20, resample=Image.BICUBIC)
            paste(canvas, core, ox, oy)
            j = jit(0.6)
            paste(canvas, orb_front, W / 2 + camx + j[0], H / 2 + camy + j[1])
            for it in items:
                if it[0] >= 0:
                    draw_planet(it)
            # floating dust motes (foreground)
            for (x, y, s, vx, vy, a) in dust:
                px = (x + vx * t + camx * 1.6) % W; py = (y + vy * t + camy * 1.6) % H
                b = blob_sprite(s)
                ba = np.asarray(b).copy(); ba[..., 3] = (ba[..., 3] * a * 0.5).astype(np.uint8)
                paste(canvas, Image.fromarray(ba, 'RGBA'), px, py)

            arr = np.asarray(canvas.convert('RGB')).astype(np.float32)
            # grade: cool shadows, warm highlights, slight lamp flicker
            lum = arr.mean(axis=2, keepdims=True) / 255
            arr = arr * (1 - lum) * np.array([0.92, 0.97, 1.08], np.float32) + arr * lum * np.array([1.06, 1.0, 0.9], np.float32)
            arr = arr * vig * (1 + pr.uniform(-0.018, 0.018))
            pose_img = arr
        frame = pose_img + grains[np.random.default_rng(f).integers(8)]
        tf = f / FPS
        fade = max(0.0, min(1.0, tf / 0.5, (SECONDS - 1 / FPS - tf) / 0.5))
        frame = np.clip(frame * fade, 0, 255).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
        if f % 48 == 0:
            print(f'frame {f}/{NFRAMES}', flush=True)
    proc.stdin.close(); proc.wait()
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'tiny_planet.mp4')
