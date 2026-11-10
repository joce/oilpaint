"""The compiled core of rendering (numba): colour and light, one pixel at a time.

:mod:`oilpaint.canvas` holds the rendering model and its constants and calls
these kernels; the reasoning is described there. Everything here works on
plain arrays: the wet layers ``top``/``und`` (H, W, PACKED) float32, their
coverage (H, W), the ground reflectance (H, W, N_BANDS) and float32 height
and colour maps.
"""

import math

import numpy as np
from numba import njit

from .spectral import N_BANDS

N = N_BANDS
IS, IV = N, N + 1  # packed indices of scattering and volume (as in _engine)

# tanh(B) / B, the Kubelka-Munk layer term, on an even grid of B up to
# _TAU_MAX (beyond it tanh(B) is 1 to float precision): looked up and
# interpolated it is several times cheaper than tanh for every band of every
# pixel, and good to about 1e-7.
_TAU_MAX = 10.0
_TAU_STEPS = 8000
_b = np.arange(_TAU_STEPS + 2) * (_TAU_MAX / _TAU_STEPS)
_b[0] = 1.0
TAU_TABLE = np.tanh(_b) / _b
TAU_TABLE[0] = 1.0
del _b


# ----------------------------------------------------------------------
# colour
# ----------------------------------------------------------------------
@njit(cache=True, error_model="numpy", nogil=True)
def _over(R, layer, alpha, min_film, tau_table):
    """Lay one wet sublayer (packed K, S, volume) over the reflectance ``R``
    (N bands, changed in place): ``canvas._composite`` for one pixel."""
    v = layer[IV]
    by_volume = v / min_film
    cov = min(alpha, by_volume) if alpha > 0.0 else by_volume
    cov = min(max(cov, 1e-6), 1.0)
    sig = max(layer[IS] / cov, 0.0)
    scale = _TAU_STEPS / _TAU_MAX
    for b in range(N):
        kap = max(layer[b] / cov, 0.0)
        B = math.sqrt(kap * kap + 2.0 * kap * sig)
        if B < _TAU_MAX:  # (false for NaN, which must not index the table)
            f = B * scale
            i = int(f)
            f -= i
            tau = tau_table[i] + f * (tau_table[i + 1] - tau_table[i])
        else:
            tau = 1.0 / B
        g = R[b]
        num = sig * tau - g * ((sig + kap) * tau - 1.0)
        den = (sig + kap - g * sig) * tau + 1.0
        R[b] = cov * (num / den) + (1.0 - cov) * g


@njit(cache=True, error_model="numpy", nogil=True)
def paint_rgb(top, und, top_a, und_a, ground, to_rgb, min_film, out):
    """Linear RGB of the painting (not reverse): the wet sublayers over the
    dry ground, Kubelka-Munk per band, then spectrum to RGB. Fills ``out``
    (H, W, 3). Where a sublayer holds no paint it leaves what is below
    unchanged, so it is skipped."""
    tau_table = TAU_TABLE
    H, W = top_a.shape
    R = np.empty(N)
    for y in range(H):
        for x in range(W):
            for b in range(N):
                R[b] = ground[y, x, b]
            if und[y, x, IV] > 0.0:
                _over(R, und[y, x], und_a[y, x], min_film, tau_table)
            if top[y, x, IV] > 0.0:
                _over(R, top[y, x], top_a[y, x], min_film, tau_table)
            for c in range(3):
                acc = 0.0
                for b in range(N):
                    acc += R[b] * to_rgb[c, b]
                out[y, x, c] = acc


# ----------------------------------------------------------------------
# height
# ----------------------------------------------------------------------
@njit(cache=True, error_model="numpy", nogil=True)
def box_blur(a, r, out):
    """Separable box blur of radius ``r`` (edges clamped) of the float32 map
    ``a`` into ``out`` (not ``a``): ``_filters._box_blur`` to float32 rounding."""
    H, W = a.shape
    if r <= 0:
        out[:] = a
        return
    inv = 1.0 / (2 * r + 1)
    tmp = np.empty((H, W), np.float32)
    for y in range(H):
        s = (r + 1) * np.float64(a[y, 0])
        for k in range(1, r + 1):
            s += a[y, min(k, W - 1)]
        for x in range(W):
            tmp[y, x] = s * inv
            s += np.float64(a[y, min(x + r + 1, W - 1)]) - a[y, max(x - r, 0)]
    col = np.zeros(W)
    for x in range(W):
        col[x] = (r + 1) * np.float64(tmp[0, x])
    for k in range(1, r + 1):
        yk = min(k, H - 1)
        for x in range(W):
            col[x] += tmp[yk, x]
    for y in range(H):
        y1 = min(y + r + 1, H - 1)
        y0 = max(y - r, 0)
        for x in range(W):
            out[y, x] = col[x] * inv
            col[x] += np.float64(tmp[y1, x]) - tmp[y0, x]


# ----------------------------------------------------------------------
# light: cast shadows
# ----------------------------------------------------------------------
#: Size of the tiles whose highest point bounds the shadow search.
TILE = 16
#: The lamp is seen as this many points across its width (fewer show as
#: bands in the side of a long shadow).
N_RAYS = 5
#: Lowest elevation (radians) of the lamp's lower edge: keeps the tangent
#: positive for a lamp at the horizon.
ELEV_MIN = 0.01
#: Highest elevation of its upper edge: short of 90 degrees, where the
#: tangent has no limit.
ELEV_MAX = 1.5
#: Over this last share of a capped search, shadows fade out rather than stop.
FADE = 0.25
#: Stretches of ray (pixels) whose highest point is bounded (_ray_bounds),
#: skipped at once where nothing on them could hide the lamp. Powers of two.
SKIPS = (8, 32, 128)


@njit(cache=True, error_model="numpy", nogil=True)
def _tile_max(h):
    """Highest point of ``h`` in each TILE-square tile and its 8 neighbours."""
    H, W = h.shape
    th, tw = (H + TILE - 1) // TILE, (W + TILE - 1) // TILE
    own = np.full((th, tw), -np.inf, np.float32)
    for y in range(H):
        ty = y // TILE
        for x in range(W):
            tx = x // TILE
            if h[y, x] > own[ty, tx]:
                own[ty, tx] = h[y, x]
    near = np.empty((th, tw), np.float32)
    for ty in range(th):
        for tx in range(tw):
            m = own[ty, tx]
            for j in range(max(ty - 1, 0), min(ty + 2, th)):
                for i in range(max(tx - 1, 0), min(tx + 2, tw)):
                    m = max(m, own[j, i])
            near[ty, tx] = m
    return near


@njit(cache=True, error_model="numpy", nogil=True)
def _bilinear(h, fx, fy):
    """``h`` at (fx, fy), which must lie inside the map."""
    H, W = h.shape
    ix, iy = int(fx), int(fy)
    ax, ay = fx - ix, fy - iy
    ix1, iy1 = min(ix + 1, W - 1), min(iy + 1, H - 1)
    return (1.0 - ay) * ((1.0 - ax) * h[iy, ix] + ax * h[iy, ix1]) + ay * (
        (1.0 - ax) * h[iy1, ix] + ax * h[iy1, ix1]
    )


@njit(cache=True, error_model="numpy", nogil=True)
def _ahead_max(h):
    """Highest point of ``h`` over (x..x+2, y..y+2): every height that a
    bilinear sample anywhere in the pixel square from (x, y) to (x+1, y+1),
    or in the square after it, can use. The first bound of _ray_bounds."""
    H, W = h.shape
    tmp = np.empty((H, W), np.float32)
    out = np.empty((H, W), np.float32)
    for y in range(H):
        for x in range(W):
            tmp[y, x] = max(
                h[y, x], max(h[y, min(x + 1, W - 1)], h[y, min(x + 2, W - 1)])
            )
    for y in range(H):
        for x in range(W):
            out[y, x] = max(
                tmp[y, x], max(tmp[min(y + 1, H - 1), x], tmp[min(y + 2, H - 1), x])
            )
    return out


@njit(cache=True, error_model="numpy", nogil=True)
def _ray_bound(v, dx, dy, n, tmp, out):
    """Double a bound along the ray (dx, dy): ``v`` bounds, for each pixel
    (x, y), every bilinear sample of the surface at p + j (dx, dy), j < n,
    for every p in the pixel square from (x, y) to (x+1, y+1); ``out`` (not
    ``v``) the same for j < 2n. The samples from n on start in one of the
    four squares around the point n on from (x, y), (x + kx, y + ky) plus a
    fraction, or off the map, where the march stops: their bound is the
    highest of ``v`` over those four (``tmp``: v's highest over each pixel
    and the next right and down)."""
    H, W = v.shape
    ox, oy = n * dx, n * dy
    kx, ky = int(math.floor(ox)), int(math.floor(oy))
    for y in range(H):
        y1 = min(y + 1, H - 1)
        for x in range(W):
            x1 = min(x + 1, W - 1)
            tmp[y, x] = max(max(v[y, x], v[y, x1]), max(v[y1, x], v[y1, x1]))
    for y in range(H):
        fy = y + oy
        cy = min(max(y + ky, 0), H - 1)
        row_in = -1.0 <= fy <= H - 1
        for x in range(W):
            m = v[y, x]
            fx = x + ox
            if row_in and -1.0 <= fx <= W - 1:
                m = max(m, tmp[cy, min(max(x + kx, 0), W - 1)])
            out[y, x] = m


@njit(cache=True, error_model="numpy", nogil=True)
def _ray_bounds(h, dx, dy, out):
    """``out[k]``: for each pixel, a bound on every sample of the surface
    over the next SKIPS[k] pixels along the ray (dx, dy), starting anywhere
    in the pixel (see _ray_bound), for the first len(out) of SKIPS."""
    H, W = h.shape
    a = _ahead_max(h)
    b = np.empty((H, W), np.float32)
    tmp = np.empty((H, W), np.float32)
    n = 1
    for k in range(out.shape[0]):
        while n < SKIPS[k]:
            _ray_bound(a, dx, dy, n, tmp, b)
            a, b = b, a
            n *= 2
        out[k] = a


@njit(cache=True, error_model="numpy", nogil=True)
def _horizon(h, bounds, near, top_all, x, y, lx, ly, tan_lo, tan_hi, reach, fade_from):
    """The steepest rise (tangent of elevation) of the surface seen from
    pixel (x, y) towards (lx, ly), sampled every pixel (bilinear) as far as
    it could matter: until it is past ``tan_hi`` (the lamp is hidden),
    nothing further could rise above ``tan_lo``, or ``reach``; from
    ``fade_from`` on, what rises there counts for less and less. A stretch
    of ray is skipped at once where its bound (``near``: the 3 x 3 tiles
    around; ``bounds``: see _ray_bounds) shows nothing on it could rise above
    what matters; the bounds hold for every sample, so no ridge, however
    thin, is stepped over."""
    H, W = h.shape
    h0 = h[y, x]
    best = -1e9
    t = 1.0
    while t <= reach:
        fx = x + t * lx
        fy = y + t * ly
        if fx < 0.0 or fy < 0.0 or fx > W - 1 or fy > H - 1:
            break
        need = max(best, tan_lo) * t
        if top_all - h0 < need:
            break
        # every sample up to TILE - 1 on uses heights in the 3 x 3 tiles
        if near[int(fy) // TILE, int(fx) // TILE] - h0 < need:
            t += TILE - 1
            continue
        qx, qy = int(fx), int(fy)
        skipped = False
        for k in range(bounds.shape[0] - 1, -1, -1):
            if bounds[k, qy, qx] - h0 < need:
                t += SKIPS[k]
                skipped = True
                break
        if skipped:
            continue
        tan = (_bilinear(h, fx, fy) - h0) / t
        if t > fade_from:
            u = min((t - fade_from) / (reach - fade_from), 1.0)
            tan *= 1.0 - u * u * (3.0 - 2.0 * u)
        if tan > best:
            best = tan
            if best >= tan_hi:
                break
        t += 1.0
    return best


@njit(cache=True, error_model="numpy", nogil=True)
def _slope(h, x, y):
    """Central-difference slope (dh/dx, dh/dy) of ``h`` at (x, y)."""
    H, W = h.shape
    xm, xp = max(x - 1, 0), min(x + 1, W - 1)
    ym, yp = max(y - 1, 0), min(y + 1, H - 1)
    return 0.5 * (h[y, xp] - h[y, xm]), 0.5 * (h[yp, x] - h[ym, x])


@njit(cache=True, error_model="numpy", nogil=True)
def shadows(h, lx3, ly3, lz3, light_size, max_reach, vis):
    """How much of the lamp in direction (lx3, ly3, lz3) (a unit vector, y
    down the image, lz3 > 0) each pixel of the height field ``h`` sees, 0..1,
    into ``vis``: the lamp as N_RAYS points across its width (``light_size``,
    its angular radius), each with a soft top and bottom edge, hidden by
    higher paint between the pixel and the lamp. Faces turned away from the
    lamp are left at 1 (they get none of it anyway). The search goes as far
    as the tallest relief could cast a shadow, but no further than
    ``max_reach``; when that cuts it short, shadows fade out over its last
    FADE rather than stop. The bounds skip most of the march, but not on a
    surface that keeps rising towards the lamp as steeply as the lamp's own
    elevation (a long, even slope): there every pixel is marched up to the
    reach, at most N_RAYS * ``max_reach`` samples per pixel (10 to 12 s for
    a 1300 x 1000 canvas tilted so, everywhere, against well under 1 s for
    painted canvases)."""
    H, W = h.shape
    vis[:] = 1.0
    horiz = math.sqrt(lx3 * lx3 + ly3 * ly3)
    if horiz < 1e-6:
        return  # a lamp straight above casts no shadow
    lx, ly = lx3 / horiz, ly3 / horiz
    az = math.atan2(ly, lx)
    rays = np.empty((N_RAYS, 2))
    for i in range(N_RAYS):
        a = az + light_size * (i - (N_RAYS - 1) / 2) / ((N_RAYS - 1) / 2)
        rays[i, 0], rays[i, 1] = math.cos(a), math.sin(a)
    elev = math.atan2(lz3, horiz)
    tan_lo = math.tan(max(elev - light_size, ELEV_MIN))
    tan_hi = math.tan(min(elev + light_size, ELEV_MAX))
    top_all, low_all = -np.inf, np.inf
    for y in range(H):
        for x in range(W):
            top_all = max(top_all, h[y, x])
            low_all = min(low_all, h[y, x])
    if not top_all - low_all > 0.0:
        return  # flat (or not finite): nothing casts a shadow
    need_reach = (top_all - low_all) / tan_lo + 1.0
    reach = min(need_reach, max_reach)
    fade_from = reach * (1.0 - FADE) if need_reach > max_reach else reach + 1.0
    near = _tile_max(h)
    # only the stretches shorter than the search are worth bounding
    levels = 0
    while levels < len(SKIPS) and SKIPS[levels] < reach:
        levels += 1
    bounds = np.empty((N_RAYS, levels, H, W), np.float32)
    for i in range(N_RAYS):
        _ray_bounds(h, rays[i, 0], rays[i, 1], bounds[i])
    for y in range(H):
        for x in range(W):
            gx, gy = _slope(h, x, y)
            if lz3 - gx * lx3 - gy * ly3 <= 0.0:
                continue  # turned away from the lamp
            v = 0.0
            for i in range(N_RAYS):
                best = _horizon(
                    h,
                    bounds[i],
                    near,
                    top_all,
                    x,
                    y,
                    rays[i, 0],
                    rays[i, 1],
                    tan_lo,
                    tan_hi,
                    reach,
                    fade_from,
                )
                if best > tan_lo:
                    u = min((best - tan_lo) / (tan_hi - tan_lo), 1.0)
                    v += 1.0 - u * u * (3.0 - 2.0 * u)
                else:
                    v += 1.0
            vis[y, x] = v / N_RAYS


# ----------------------------------------------------------------------
# light: shading and gloss
# ----------------------------------------------------------------------
#: Wet paint this thick (units of paint thickness) has a fully wet surface.
WET_SURFACE = 0.5
#: Share of the highlight lost by paint sunk fully into an absorbent ground.
MATTE_GLOSS_LOSS = 0.9


@njit(cache=True, error_model="numpy", nogil=True)
def _smith(nx, k):
    """Schlick-GGX shadowing of facets seen at cosine ``nx``."""
    return nx / (nx * (1.0 - k) + k)


@njit(cache=True, error_model="numpy", nogil=True)
def _window(q, r0, r1):
    """1 for facets up to ``r0`` alphas from the mirror slope (``q`` is 1
    plus the square of that distance), falling smoothly to 0 at ``r1``."""
    r = math.sqrt(max(q - 1.0, 0.0))
    u = min(max((r - r0) / (r1 - r0), 0.0), 1.0)
    return 1.0 - u * u * (3.0 - 2.0 * u)


@njit(cache=True, error_model="numpy", nogil=True)
def _lobe(dx, dy, mx, my, cxx, cxy, cyy, a2, g, r0, r1):
    """One gloss lobe in slope space (GGX, Trowbridge-Reitz): the share of
    facets tilted ``(dx, dy)`` away from the pixel's mean slope, for roughness
    ``a2`` (alpha squared) widened by ``(cxx, cxy, cyy)``, the spread of
    slopes inside the pixel (from its curvature: a ridge crest shows a narrow
    line of highlight along it, rather than one that flickers from pixel to
    pixel), times the shadowing term ``g``. Only facets near the mirror slope
    shine: the lobe is tapered smoothly to nothing between ``r0`` and ``r1``
    alphas from it (_window), so a highlight has no hard edge and paint far
    from the mirror slope gets no veil of reflected light. Less what flat
    paint shows of the same tapered lobe (facets tilted ``(mx, my)``): flat
    paint seen from the front keeps its colour, its mirror image of the lamp
    pointing away from the viewer."""
    axx = a2 + cxx
    ayy = a2 + cyy
    det = axx * ayy - cxy * cxy
    norm = 1.0 / (math.pi * math.sqrt(det))
    q = 1.0 + (ayy * dx * dx - 2.0 * cxy * dx * dy + axx * dy * dy) / det
    lobe = g * norm / (q * q) * _window(q, r0, r1)
    q = 1.0 + (ayy * mx * mx - 2.0 * cxy * mx * my + axx * my * my) / det
    return max(lobe - norm / (q * q) * _window(q, r0, r1), 0.0)


@njit(cache=True, error_model="numpy", nogil=True)
def _occlusion(aos, ao, ao_eps, y, x, h0):
    """Share of the room's light hidden around pixel (x, y), and the part of
    it hidden by the finest scale (crevices): at each scale, the mean rise
    of the surroundings over its radius is the tangent of how high they
    stand around, and they hide about sin^2 of that angle of the sky."""
    occ = 0.0
    cav = 0.0
    for k in range(ao.shape[0]):
        rad = ao[k, 0]
        d = max(aos[k, y, x] - h0 - ao_eps * rad, 0.0) / rad
        d = d * d / (1.0 + d * d)
        occ += ao[k, 1] * d
        if k == 0:
            cav = d
    return occ, cav


@njit(cache=True, error_model="numpy", nogil=True)
def _cover(wt, pt, film0, film1, dry0, dry1):
    """How glossy the surface is, 0..1, from the wet (``wt``) and the total
    (``pt``) paint thickness: a thin film is dull, a full film of oil
    glossy; dried paint has sunk and needs to be thicker to shine."""
    c = max(
        min(max((wt - film0) / (film1 - film0), 0.0), 1.0),
        min(max((pt - dry0) / (dry1 - dry0), 0.0), 1.0),
    )
    return c * c * (3.0 - 2.0 * c)


@njit(cache=True, error_model="numpy", nogil=True)
def _smoothstep(e0, e1, v):
    u = min(max((v - e0) / (e1 - e0), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


@njit(cache=True, error_model="numpy", nogil=True)
def _mirror(gx, gy, hxx, hyy, hxy, nz, ndl, mx, my, wt, p):
    """The oil's mirror image of the lamp at a pixel of slope (gx, gy) and
    curvature (hxx, hyy, hxy), on its own, relative to flat white paint under
    the lamp: how glossy its paint is when wet or dry (``wt`` its wet
    thickness) times the lobe (_lobe). Before the share of the lamp it sees
    and how far its film is thick enough to shine (``cover``)."""
    footprint, a2_wet, a2_dry, gloss_wet, gloss_dry, r0, r1 = p
    wetness = min(wt / WET_SURFACE, 1.0)
    # the slopes inside the pixel spread its highlight
    cxx = footprint * (hxx * hxx + hxy * hxy)
    cyy = footprint * (hyy * hyy + hxy * hxy)
    cxy = footprint * hxy * (hxx + hyy)
    # slope of the facets that mirror the lamp to the viewer, relative to
    # the pixel's own slope
    dx, dy = mx - gx, my - gy
    nl = max(ndl, 1e-4)
    a2 = a2_dry + (a2_wet - a2_dry) * wetness
    k = 0.5 * math.sqrt(a2)
    g = _smith(nl, k) * _smith(nz, k) / nz
    strength = gloss_dry + (gloss_wet - gloss_dry) * wetness
    return strength * _lobe(dx, dy, mx, my, cxx, cxy, cyy, a2, g, r0, r1)


@njit(cache=True, error_model="numpy", nogil=True)
def _near_row(src, r, out):
    """``out[x]`` = 1 if ``src`` (one row) is not 0 anywhere within ``r`` of
    x, else 0: a running window, the same as a brute-force search."""
    W = src.shape[0]
    last = -(r + 1)  # the last column with a highlight, up to x + r
    for x in range(min(r, W - 1) + 1):
        if src[x] != 0.0:
            last = x
    for x in range(W):
        ahead = x + r
        if ahead < W and src[ahead] != 0.0:
            last = ahead
        out[x] = 1 if last >= x - r else 0


#: Weights of the 5 x 5 pixels whose curvature tells a crest (_crest).
BINOMIAL = (1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16)
#: A gathering shorter than this (pixels) is no gathering: the pixel keeps
#: its own highlight exactly.
NO_REACH = 1e-3
#: Pixels this near the border are not gathered: the curvature stencil (2 px)
#: and its blur (2 px) would read beyond the canvas.
EDGE = 4


@njit(cache=True, error_model="numpy", nogil=True)
def _blur_row(src, out):
    """``src`` (3 x W: one row of H H^T) blurred along the row with BINOMIAL
    weights (edges clamped), into ``out``: the first half of _crest's 5 x 5
    blur, done once per row."""
    W = src.shape[1]
    for x in range(W):
        for f in range(3):
            acc = 0.0
            for i in range(-2, 3):
                acc += BINOMIAL[i + 2] * src[f, min(max(x + i, 0), W - 1)]
            out[f, x] = acc


@njit(cache=True, error_model="numpy", nogil=True)
def _crest(sig, R, H, W, x, y, gate):
    """Along which direction, and how far (0..1 of the span), the highlight
    of pixel (x, y) is gathered: along a narrow crest, furrow or lip only.
    From the curvature H of the 5 x 5 pixels around it, BINOMIAL-weighted:
    ``sig`` holds H H^T of each pixel (second differences two pixels apart
    in x and y, as the cross term's, so that a crest reads the same at any
    slant), already blurred along the rows, its rows at the row number
    masked with ``R``. Its larger principal curvature k1 runs across a
    crest, the smaller k2 along it. The share is a smoothstep of the
    anisotropy (|k1| - |k2|) / (|k1| + |k2|) times one of |k1| (``gate``:
    anisotropy from, to; curvature from, to), so that domes and round dabs
    (k1 ~ k2) and broad, gently curved slopes keep their own highlight. The
    direction is that of least curvature.

    Not within EDGE pixels of the canvas's border, where the stencils run
    off the canvas (clamped, they read a slope as a curvature) and the
    gathering would. And a crest only about one pixel wide at a shallow
    slant reads too little anisotropy (sigma 0.4 at 22.5 degrees: 0.44) and
    is not gathered: it still beads."""
    if x < EDGE or y < EDGE or x >= W - EDGE or y >= H - EDGE:
        return 1.0, 0.0, 0.0
    a0, a1, k0, k1 = gate
    sxx = sxy = syy = 0.0
    for j in range(-2, 3):
        row = min(max(y + j, 0), H - 1) & R
        w = BINOMIAL[j + 2]
        sxx += w * sig[row, 0, x]
        sxy += w * sig[row, 1, x]
        syy += w * sig[row, 2, x]
    tr = sxx + syy
    d = math.sqrt((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy)
    l1 = 0.5 * (tr + d)
    l2 = max(0.5 * (tr - d), 0.0)
    c1, c2 = math.sqrt(l1), math.sqrt(l2)  # rms principal curvatures
    if not c1 > 0.0:
        return 1.0, 0.0, 0.0
    share = _smoothstep(a0, a1, (c1 - c2) / (c1 + c2)) * _smoothstep(k0, k1, c1)
    if not share > 0.0:
        return 1.0, 0.0, 0.0
    # the eigenvector of the smaller eigenvalue: along the crest
    if sxx >= syy:
        tx, ty = -sxy, l1 - syy
    else:
        tx, ty = -(l1 - sxx), sxy
    n = math.sqrt(tx * tx + ty * ty)
    if not n > 0.0:
        return 1.0, 0.0, 0.0
    return tx / n, ty / n, share


@njit(cache=True, error_model="numpy", nogil=True)
def _tap(lob, R, H, W, fx, fy, tx, ty):
    """The own highlight (``lob``, row number masked with ``R``) at the point (fx, fy)
    of a line along a crest (direction (tx, ty)): interpolated along the
    row nearest the point if the line is closer to the rows, else along the
    nearest column, never between rows (columns): what it reads lies at
    most about one diagonal pixel step (0.7 px) across the line from the point,
    where a bilinear
    blend would reach a pixel away."""
    if abs(tx) >= abs(ty):
        r = min(int(fy + 0.5), H - 1) & R
        i = int(fx)
        a = fx - i
        return (1.0 - a) * lob[r, i] + a * lob[r, min(i + 1, W - 1)]
    i = min(int(fx + 0.5), W - 1)
    j = int(fy)
    a = fy - j
    return (1.0 - a) * lob[j & R, i] + a * lob[min(j + 1, H - 1) & R, i]


@njit(cache=True, error_model="numpy", nogil=True)
def _gather(lob, R, H, W, x, y, tx, ty, reach, taps):
    """The highlight of pixel (x, y) gathered along a crest: the own
    highlights (_tap) at ``taps`` points each side along (tx, ty) up to
    ``reach`` pixels away, tent-weighted, so that a glint on a crest that
    runs slantwise across the pixels is a line rather than beads at each
    step. The centre point is the pixel itself."""
    total = 0.0
    weights = 0.0
    for k in range(-taps, taps + 1):
        w = 1.0 - abs(k) / (taps + 1.0)
        o = reach * k / taps
        fx = min(max(x + o * tx, 0.0), W - 1.0)
        fy = min(max(y + o * ty, 0.0), H - 1.0)
        total += w * _tap(lob, R, H, W, fx, fy, tx, ty)
        weights += w
    return total / weights


@njit(cache=True, error_model="numpy", nogil=True)
def shade(
    h,
    vis,
    aos,
    ao,
    lin,
    paint,
    wet,
    matte,
    lx3,
    ly3,
    lz3,
    ambient,
    shadow_depth,
    cavity,
    ao_eps,
    footprint,
    oil_f0,
    rough_wet,
    rough_dry,
    gloss_wet,
    gloss_dry,
    gloss_taper,
    gloss_cut,
    film0,
    film1,
    dry0,
    dry1,
    gloss_span,
    gloss_taps,
    crest_gate,
    out,
):
    """Light the height field ``h`` (pixels) and the linear colour ``lin``
    with the lamp in direction (lx3, ly3, lz3) (a unit vector, y down the
    image, lz3 > 0), into ``out`` (linear RGB, H x W x 3). See Canvas._light
    for the model and the parameters (the constants of the same names in
    oilpaint.canvas; ``gloss_wet`` and ``gloss_dry`` already scaled by the
    ``gloss`` of render).

    ``vis`` is how much of the lamp each pixel sees (see shadows); ``aos``
    holds ``h`` box-blurred at the ambient-occlusion radii (one map per row
    of ``ao``: radius, weight); ``paint`` and ``wet`` are the total and the
    wet paint thickness; ``matte`` the sunk-in mattness, or an empty (0, 0)
    array for none.

    Each pixel's own mirror image of the lamp (_mirror) is gathered along a
    narrow crest (_crest, ``crest_gate``) over ``gloss_span`` pixels each
    way (_gather, ``gloss_taps`` points each side); then the pixel's own
    film (``cover``), the share of the lamp it sees and its mattness apply.
    Rows are worked out a few ahead of the one being lit and kept in
    buffers of a few rows (float64, and uint8 for whether an own highlight
    lies near): no map the size of the canvas. Exact: with ``gloss_span``
    0 or the gate shut, the highlight is each pixel's own; flat paint, paint
    gently modelled (beyond the cut-off), ``gloss`` 0 and pixels with no own
    highlight within ``ceil(gloss_span) + 1`` pixels are as they would be
    without gathering."""
    H, W = h.shape
    # the half vector between the lamp and the viewer (straight in front)
    hx, hy, hz = lx3, ly3, lz3 + 1.0
    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
    hx, hy, hz = hx / hn, hy / hn, hz / hn
    # the slope (dh/dx, dh/dy) of a facet facing the half vector
    mx, my = -hx / hz, -hy / hz
    # Fresnel at the (fixed) angle between the viewer and the half vector
    fresnel = oil_f0 + (1.0 - oil_f0) * (1.0 - hz) ** 5
    # facets per unit slope -> per solid angle (1 / cos^4), Fresnel, and
    # relative to flat white paint lit by the same lamp (lz3 / pi)
    spec_k = math.pi * fresnel / (4.0 * lz3 * hz**4)
    gp = (
        footprint,
        rough_wet**2,
        rough_dry**2,
        gloss_wet,
        gloss_dry,
        gloss_taper,
        gloss_cut,
    )
    has_matte = matte.shape[0] > 0
    has_gloss = gloss_wet > 0.0 or gloss_dry > 0.0
    gather = has_gloss and gloss_span > NO_REACH and gloss_taps > 0
    # rows ahead of the one being lit that the gathering reads (a tap up to
    # the span away, and the next pixel interpolated with it), and the rows
    # kept: as many behind
    lag = int(math.ceil(gloss_span)) + 1 if gather else 0
    # rows are kept in buffers of a power of two rows, R + 1, indexed by the
    # row number masked with R
    R = 1
    while 2 * lag > R:
        R = 2 * R + 1
    shading = np.empty((R + 1, W))
    recv = np.empty((R + 1, W))  # what of its highlight a pixel shows
    lob = np.zeros((R + 1, W))  # each pixel's own mirror image of the lamp
    sig = np.zeros((3, W))  # H H^T of each pixel's curvature, this row
    row_sig = np.zeros((R + 1, 3, W))  # the same blurred along the rows
    near = np.zeros((R + 1, W), np.uint8)  # an own highlight within lag in the row
    for yy in range(H + lag):
        if yy < H:
            y = yy
            r = y & R
            ym, yp = max(y - 1, 0), min(y + 1, H - 1)
            for x in range(W):
                h0 = h[y, x]
                gx, gy = _slope(h, x, y)
                nz = 1.0 / math.sqrt(1.0 + gx * gx + gy * gy)
                ndl = (lz3 - gx * lx3 - gy * ly3) * nz
                v = vis[y, x]
                # the lamp, less what paint between hides of it; the room,
                # less what hollows and grooves hide of it
                direct = max(ndl, 0.0) / lz3 * (1.0 - shadow_depth * (1.0 - v))
                occ, cav = _occlusion(aos, ao, ao_eps, y, x, h0)
                sky = (0.5 + 0.5 * nz) * max(1.0 - occ, 0.0)
                shading[r, x] = (ambient * sky + (1.0 - ambient) * direct) * (
                    1.0 - cavity * cav
                )
                # the oil mirrors the lamp
                lob[r, x] = 0.0
                recv[r, x] = 0.0
                if not has_gloss:
                    continue
                xm, xp = max(x - 1, 0), min(x + 1, W - 1)
                hxx = h[y, xp] - 2.0 * h0 + h[y, xm]
                hyy = h[yp, x] - 2.0 * h0 + h[ym, x]
                hxy = 0.25 * (h[yp, xp] - h[yp, xm] - h[ym, xp] + h[ym, xm])
                if gather:
                    # the curvature for _crest: second differences two
                    # pixels apart, the same span as hxy's, so that a crest
                    # reads the same at any slant
                    wxx = 0.25 * (
                        h[y, min(x + 2, W - 1)] - 2.0 * h0 + h[y, max(x - 2, 0)]
                    )
                    wyy = 0.25 * (
                        h[min(y + 2, H - 1), x] - 2.0 * h0 + h[max(y - 2, 0), x]
                    )
                    sig[0, x] = wxx * wxx + hxy * hxy
                    sig[1, x] = hxy * (wxx + wyy)
                    sig[2, x] = wyy * wyy + hxy * hxy
                pt = paint[y, x]
                wt = wet[y, x]
                if (wt > film0 or pt > dry0) and direct > 0.0:
                    lob[r, x] = _mirror(gx, gy, hxx, hyy, hxy, nz, ndl, mx, my, wt, gp)
                    f = spec_k * v * _cover(wt, pt, film0, film1, dry0, dry1)
                    if has_matte:
                        f *= 1.0 - MATTE_GLOSS_LOSS * matte[y, x]
                    recv[r, x] = f
            if gather:
                _near_row(lob[r], lag, near[r])
                _blur_row(sig, row_sig[r])
        y = yy - lag
        if y < 0:
            continue
        r = y & R
        for x in range(W):
            spec = 0.0
            f = recv[r, x]
            if f > 0.0:
                own = lob[r, x]
                if gather:
                    # any own highlight within reach (a lag-square)?
                    found = False
                    for j in range(max(y - lag, 0), min(y + lag, H - 1) + 1):
                        if near[j & R, x]:
                            found = True
                            break
                    if found:
                        tx, ty, share = _crest(row_sig, R, H, W, x, y, crest_gate)
                        reach = gloss_span * share
                        if reach > NO_REACH:
                            own = _gather(lob, R, H, W, x, y, tx, ty, reach, gloss_taps)
                spec = f * own

            # colour: the lights roll off to white rather than clip
            s = shading[r, x]
            for c in range(3):
                base = lin[y, x, c]
                val = base * s + spec
                if val > base and base < 1.0:
                    room = 1.0 - base
                    u = (val - base) / room
                    val = base + room * u / math.sqrt(1.0 + u * u)
                out[y, x, c] = max(val, 0.0)
