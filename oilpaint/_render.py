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

from .spectral import _check_bands

# The band count, written out for the reason given in _engine.py.
N = 24


@njit(cache=True, error_model="numpy")
def _compiled_bands():
    return N


_check_bands(__file__, N, _compiled_bands)

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
TAU_TABLE.flags.writeable = False
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


@njit(cache=True, error_model="numpy", nogil=True)
def _tally(row, count, sign):
    """``count`` += ``sign`` times ``row`` (one row of whether an own
    highlight lies near), column by column: a running count of the rows
    with one, as rows come within reach of the row being lit and leave."""
    for x in range(count.shape[0]):
        count[x] += sign * row[x]


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
def _crest_and_trace(sig, R, H, W, x, y, gate):
    """Along which direction, and how far (0..1 of the span), the highlight
    of pixel (x, y) is gathered: along a narrow crest, furrow or lip only.
    From the curvature H of the 5 x 5 pixels around it, BINOMIAL-weighted:
    ``sig`` holds H H^T of each pixel (second differences two pixels apart
    in x and y, the same span as the cross term's; a crest a few pixels
    wide reads the same at any slant, a thinner one some degrees nearer 45
    than it runs: a ridge of sigma 0.6 px, set directly, 1.8, 3.0 and 6.4
    degrees nearer at 3, 5 and 11; of sigma 1, 1 degree at 3; of sigma 2,
    0.3 or less), already blurred along the rows, its rows at the row
    number masked with ``R``. Its larger principal curvature k1 runs across
    a crest, the smaller k2 along it. The share is a smoothstep of the
    anisotropy (|k1| - |k2|) / (|k1| + |k2|) times one of |k1| (``gate``:
    anisotropy from, to; curvature from, to), so that domes and round dabs
    (k1 ~ k2) and broad, gently curved slopes keep their own highlight. The
    direction is that of least curvature. And the trace of the blurred H
    H^T, how strongly curved the paint is, which _cross and _peak find a
    crest's line by.

    Not within EDGE pixels of the canvas's border, where the stencils run
    off the canvas (clamped, they read a slope as a curvature) and the
    gathering would.

    The heights a canvas is lit by are smoothed [1, 2, 1] each way
    (Canvas.surface_height), so a crest is never much under a pixel wide
    there. At shallow slants its glint still dashes (beading: the glint
    summed across the crest, standard deviation over mean along it, as in
    tests/test_render.py), gathered over a step or not, where:

    * the paint is a line one pixel wide, whose height swells and fades
      along it as it crosses the rows, which reads as curvature along it
      too: of sigma 0.3, laid as paint, 1.44 and 1.31 at 5 and 11 degrees
      (the gate is half open at 2 and 5 degrees, shut at 11: anisotropy
      0.41). A longer view (the curvature averaged further along the
      crest, or over 9 x 9 pixels) opens it, but opens the canvas weave,
      dabs, stipple and spatter as well;
    * the crest is low: a ridge of paint of sigma 0.4 (1.08, 0.85, 0.56 at
      2, 5 and 11 degrees), under about 0.3 px high once smoothed, which
      the curvature part of the gate opens only half way (share 0.4), so
      that the gathering reaches as far less. (This is a smooth ridge of
      paint; the crests brushes and knives make, liner lines and hog-flat
      lips included, read the gate at least as open as their anisotropy:
      share 0.85 to 0.9 on a liner's lit pixels.)

    Heights not smoothed so (set directly, as tests do) of a crest under
    about 0.5 px are worse: sigma 0.4 at 22.5 degrees reads anisotropy
    0.43 and is not gathered."""
    if x < EDGE or y < EDGE or x >= W - EDGE or y >= H - EDGE:
        return 1.0, 0.0, 0.0, 0.0
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
        return 1.0, 0.0, 0.0, tr
    share = _smoothstep(a0, a1, (c1 - c2) / (c1 + c2)) * _smoothstep(k0, k1, c1)
    if not share > 0.0:
        return 1.0, 0.0, 0.0, tr
    # the eigenvector of the smaller eigenvalue: along the crest
    if sxx >= syy:
        tx, ty = -sxy, l1 - syy
    else:
        tx, ty = -(l1 - sxx), sxy
    n = math.sqrt(tx * tx + ty * ty)
    if not n > 0.0:
        return 1.0, 0.0, 0.0, tr
    return tx / n, ty / n, share, tr


@njit(cache=True, error_model="numpy", nogil=True)
def _crest(sig, R, H, W, x, y, gate):
    """_crest_and_trace without the trace: (tx, ty, share)."""
    tx, ty, share, _ = _crest_and_trace(sig, R, H, W, x, y, gate)
    return tx, ty, share


@njit(cache=True, error_model="numpy", nogil=True)
def _step(tx, ty, span_top):
    """The step of the pixel staircase along a crest in direction (tx, ty)
    (pixels): 1 / sin of its slant from the rows or the columns, up to
    ``span_top``; for slants nearer the rows or columns than that (sin
    below 1 / span_top: asin(1 / 16) = 3.58 degrees for 16), where the
    staircase drifts so slowly along the crest that it hardly beads,
    span_top^2 times that sine: continuous (with a kink there), down to 0
    for a crest along the rows or columns."""
    st = min(abs(tx), abs(ty))
    if st * span_top >= 1.0:
        return 1.0 / st
    return span_top * span_top * st


@njit(cache=True, error_model="numpy", nogil=True)
def _crest_at(sig, R, H, W, memo, val, x, y, gate):
    """_crest_and_trace at pixel (x, y), worked out once: kept in ``val``
    (tx, ty, share, trace) at the row number masked with ``R``, ``memo``
    marking which are."""
    r = y & R
    if not memo[r, x]:
        tx, ty, share, tr = _crest_and_trace(sig, R, H, W, x, y, gate)
        val[r, 0, x] = tx
        val[r, 1, x] = ty
        val[r, 2, x] = share
        val[r, 3, x] = tr
        memo[r, x] = 1
    return val[r, 0, x], val[r, 1, x], val[r, 2, x], val[r, 3, x]


@njit(cache=True, error_model="numpy", nogil=True)
def _trace(sig, R, H, W, memo, val, steep, major, minor, gate):
    """How strongly curved the paint is (_crest_and_trace's trace) at
    ``minor`` across and ``major`` along: (x, y) = (major, minor), or
    (minor, major) for a ``steep`` crest."""
    if steep:
        return _crest_at(sig, R, H, W, memo, val, minor, major, gate)[3]
    return _crest_at(sig, R, H, W, memo, val, major, minor, gate)[3]


@njit(cache=True, error_model="numpy", nogil=True)
def _peak(sig, R, H, W, memo, val, steep, major, guess, gate):
    """Where across, near ``guess``, the crest lies at ``major``: the most
    curved of the three pixels round it (_trace), and the vertex of a
    parabola through it and its neighbours (within half a pixel of it).
    Returns the position and that pixel; the pixel is -1 off the paint
    _crest reads (EDGE)."""
    size = W if steep else H
    c = int(math.floor(guess + 0.5))
    if c - 2 < EDGE or c + 2 >= size - EDGE:
        return guess, -1
    lo = _trace(sig, R, H, W, memo, val, steep, major, c - 1, gate)
    mid = _trace(sig, R, H, W, memo, val, steep, major, c, gate)
    hi = _trace(sig, R, H, W, memo, val, steep, major, c + 1, gate)
    best, top, a, b = c, mid, lo, hi
    if lo > mid and lo >= hi:
        best, top, b = c - 1, lo, mid
        a = _trace(sig, R, H, W, memo, val, steep, major, c - 2, gate)
    elif hi > mid and hi > lo:
        best, top, a = c + 1, hi, mid
        b = _trace(sig, R, H, W, memo, val, steep, major, c + 2, gate)
    den = a - 2.0 * top + b
    off = 0.0
    if den < 0.0:
        off = min(max(0.5 * (a - b) / den, -0.5), 0.5)
    return best + off, best


@njit(cache=True, error_model="numpy", nogil=True)
def _side(
    sig, R, H, W, memo, val, gate, run_on, steep, major0, p0, tr0, slope, e, t, qs, ps,
    n,
):  # fmt: skip
    """Follow the crest from where it crosses ``major0`` (at ``p0``
    across) one way (``e`` = -1 or 1), a pixel at a time along the rows (a
    shallow crest: ``major`` is x) or the columns (``steep``), each time to
    where it is most curved (_peak) near where it was heading (``slope``,
    then the slope from the start). At each, how surely it is the same
    crest: _crest's share there, times how near its direction is to ``t``
    (``run_on``: squared cosines from, to), times how near it is to where
    it was heading (from a pixel to half a pixel away); ``qs[k]`` becomes
    the least of these up to k steps, at most, for this side and whatever
    it held; ``ps[k]`` where it was found k steps on; no further than
    ``n`` steps (0 beyond). Returns (steps it went with that above 1/2,
    where it was then)."""
    size = H if steep else W
    steps = qs.shape[0] - 1
    prev, sl, least = p0, slope, 1.0
    far_k, far_p = 0, p0
    for k in range(1, steps + 1):
        major = major0 + e * k
        if k > n:
            least = 0.0
        if least > 0.0:
            p = 0.0
            if EDGE <= major < size - EDGE:
                guess = min(max(prev + e * sl, p0 - k), p0 + k)
                pos, m = _peak(sig, R, H, W, memo, val, steep, major, guess, gate)
                if m >= 0 and abs(pos - p0) <= k + 1.0:
                    if steep:
                        ux, uy, share, tr = _crest_at(
                            sig, R, H, W, memo, val, m, major, gate
                        )
                    else:
                        ux, uy, share, tr = _crest_at(
                            sig, R, H, W, memo, val, major, m, gate
                        )
                    c = t[0] * ux + t[1] * uy
                    p = share * _smoothstep(run_on[0], run_on[1], c * c)
                    # as strongly curved: not a drop or a lump, tapering off
                    ratio = min(tr, tr0) / max(tr, tr0, 1e-300)
                    p *= _smoothstep(SAME[0], SAME[1], ratio)
                    p *= _smoothstep(1.0, 0.5, abs(pos - guess))
                    prev = pos
                    ps[k] = pos
                    if k >= 2:
                        sl = (pos - p0) / (e * k)
            least = min(least, p)
            if least > 0.5:
                far_k, far_p = k, prev
        qs[k] = min(qs[k], least)
    return far_k, far_p


@njit(cache=True, error_model="numpy", nogil=True)
def _cross(sig, R, H, W, memo, val, steep, major0, minor0, gate):
    """Where the crest crosses the row (``major0`` = y; or the column,
    ``steep``) near ``minor0``: of the pixels up to FIND px away more curved
    (_trace) than both their neighbours, the one most curved for how far
    it is (its trace over 1 + (distance / 2)^2: the nearer of two alike,
    the middle of a crest rather than the weaker curve at its foot),
    refined (_peak). Returns the position across and that pixel (-1 if
    none)."""
    size = W if steep else H
    if not EDGE + FIND + 1 <= minor0 < size - EDGE - FIND - 1:
        return 0.0, -1
    best, top = -1, 0.0
    a = _trace(sig, R, H, W, memo, val, steep, major0, minor0 - FIND - 1, gate)
    v = _trace(sig, R, H, W, memo, val, steep, major0, minor0 - FIND, gate)
    for m in range(minor0 - FIND, minor0 + FIND + 1):
        b = _trace(sig, R, H, W, memo, val, steep, major0, m + 1, gate)
        if v >= a and v > b:
            d = (m - minor0) / 2.0
            score = v / (1.0 + d * d)
            if score > top:
                best, top = m, score
        a, v = v, b
    if best < 0:
        return 0.0, -1
    return _peak(sig, R, H, W, memo, val, steep, major0, best, gate)


@njit(cache=True, error_model="numpy", nogil=True)
def _straight(qs, ps, p0, slope):
    """``qs[k]`` no more than how near (STRAIGHT) the crest stays, up to k
    steps either way (where it was found: ``ps``), to the straight line
    through ``p0`` at ``slope`` that it is gathered along: not on round a
    bend or a curling rim."""
    for i in range(2):
        e = 2 * i - 1
        worst = 0.0
        for k in range(1, qs.shape[0]):
            if not qs[k] > 0.0:
                break
            worst = max(worst, abs(ps[i, k] - p0 - slope * e * k))
            qs[k] = min(qs[k], _smoothstep(STRAIGHT[0], STRAIGHT[1], worst))


@njit(cache=True, error_model="numpy", nogil=True)
def _run(sig, R, H, W, memo, val, gate, run_on, steep, major0, p0, t, tr0, qs, ps):
    """Follow the crest from where it crosses ``major0`` (at ``p0``
    across), direction ``t`` as _crest reads it there, both ways (_side):
    ``qs[k]`` becomes the least sureness up to k steps either way, and no
    more than how straight it runs (_straight). Returns its slope across
    per step along: from where it was found well (sureness above 1/2)
    furthest each way, if 4 steps apart or more (else as _crest reads it):
    _crest reads a thin crest's direction some degrees nearer 45 than it
    runs. Only as many steps as the step of the staircase _crest's reading
    gives, twice over, and 2 (the step the crest really runs at is longer,
    at most by about half)."""
    slope = t[0] / t[1] if steep else t[1] / t[0]
    qs[0] = 1.0
    qs[1:] = 1.0
    span = qs.shape[0] - 1
    n = min(span, int(2.0 * _step(t[0], t[1], span)) + 2)
    kb, pb = _side(
        sig, R, H, W, memo, val, gate, run_on, steep, major0, p0, tr0, slope, -1, t,
        qs, ps[0], n,
    )  # fmt: skip
    ka, pa = _side(
        sig, R, H, W, memo, val, gate, run_on, steep, major0, p0, tr0, slope, 1, t,
        qs, ps[1], n,
    )  # fmt: skip
    if ka + kb >= 4:
        slope = (pa - pb) / (ka + kb)
    _straight(qs, ps, p0, slope)
    return slope


@njit(cache=True, error_model="numpy", nogil=True)
def _follow(
    sig, R, H, W, memo, val, x, y, tx, ty, gate, run_on, qs, ps, followed, follow_val
):
    """Follow the crest near pixel (x, y), direction (tx, ty) as _crest
    reads it: from where it crosses the pixel's row or column (_cross), as
    _run does from there with _crest's reading at that pixel (worked out
    once for each such pixel: ``followed``, ``follow_val`` hold the slope and the
    sureness, at the row number masked with ``R``); ``qs`` the sureness.
    Returns how far across (px) the pixel is from where the crest crosses,
    the crest's direction (cx, cy), and _crest's share there (0, and no
    sureness, where no crest is found, or one running the other side of 45
    degrees)."""
    steep = abs(ty) > abs(tx)
    if steep:
        major0, minor0 = y, x
    else:
        major0, minor0 = x, y
    qs[0] = 1.0
    qs[1:] = 0.0
    p0, m0 = _cross(sig, R, H, W, memo, val, steep, major0, minor0, gate)
    if m0 < 0:
        return 99.0, tx, ty, 0.0
    xc, yc = (m0, major0) if steep else (major0, m0)
    ux, uy, share0, tr0 = _crest_at(sig, R, H, W, memo, val, xc, yc, gate)
    if not share0 > 0.0 or (abs(uy) > abs(ux)) != steep:
        return 99.0, tx, ty, 0.0
    r = yc & R
    if not followed[r, xc]:
        follow_val[r, xc, 0] = _run(
            sig, R, H, W, memo, val, gate, run_on, steep, major0, p0, (ux, uy),
            tr0, qs, ps,
        )  # fmt: skip
        follow_val[r, xc, 1:] = qs
        followed[r, xc] = 1
    slope = follow_val[r, xc, 0]
    qs[:] = follow_val[r, xc, 1:]
    n = math.sqrt(1.0 + slope * slope)
    if steep:
        cx, cy = slope / n, 1.0 / n
    else:
        cx, cy = 1.0 / n, slope / n
    if cx * tx + cy * ty < 0.0:
        cx, cy = -cx, -cy
    return minor0 - p0, cx, cy, share0


@njit(cache=True, error_model="numpy", nogil=True)
def _at(qs, k):
    """``qs`` at ``k`` steps (a fraction: between its neighbours)."""
    i = min(int(k), qs.shape[0] - 2)
    a = min(k - i, 1.0)
    return (1.0 - a) * qs[i] + a * qs[i + 1]


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
    step. The taps lie reach / taps apart, weighted 1 - k / (taps + 1) at
    the k-th: a tent falling to 0 at reach (taps + 1) / taps, its
    half-width. The centre point is the pixel itself."""
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
def _gather_even(lob, R, H, W, x, y, tx, ty, reach):
    """The highlight of pixel (x, y) gathered along (tx, ty) up to
    ``reach`` px each way: the own highlights a pixel apart along it,
    weighted 1 - |o| / (reach + 1) at o px (a tent; the weight of a tap
    that comes in as the reach grows starts at 0, so the result changes
    continuously with the reach). Read by bilinear interpolation: each tap at the same
    distance across the crest as the pixel (_tap's reading along the
    nearest row would wander up to half a pixel across it and back with
    the staircase, over so long a reach, and ripple an even line)."""
    n = int(reach) + 1
    total = 0.0
    weights = 0.0
    for k in range(-n, n + 1):
        w = 1.0 - abs(k) / (reach + 1.0)
        if w > 0.0:
            fx = min(max(x + k * tx, 0.0), W - 1.0)
            fy = min(max(y + k * ty, 0.0), H - 1.0)
            i = min(int(fx), W - 2)
            j = min(int(fy), H - 2)
            a, b = fx - i, fy - j
            top = (1.0 - a) * lob[j & R, i] + a * lob[j & R, i + 1]
            low = (1.0 - a) * lob[(j + 1) & R, i] + a * lob[(j + 1) & R, i + 1]
            total += w * ((1.0 - b) * top + b * low)
            weights += w
    return total / weights


#: How far (px) across from a pixel the crest whose highlight it gathers
#: over a step is looked for (_follow): as far as a glint lies on the
#: flank of a crest (a ridge of sigma 1 px, 1 px high: 4 px).
FIND = 4

#: Along a crest followed over a step (_side), how strongly curved it must
#: stay against where it was found (the less over the more of _crest's
#: trace there: from, to): an elongated drop or lump, curved most at its
#: middle, is not followed far.
SAME = (0.25, 0.5)

#: How far (px) a crest followed over a step may stray from the straight
#: line it is gathered along (from, to: all of the way to none; _follow).
STRAIGHT = (0.75, 0.35)

#: A crest that _crest reads as running at a slant whose staircase step
#: (_step) is no longer than this (px) is not followed: at 45 degrees and
#: near it (1.41 to 1.5) the step is shorter than GLOSS_SPAN however far
#: _crest's reading is off, which is never more than about 7 degrees.
FOLLOW_FROM = 1.5


@njit(cache=True, error_model="numpy", nogil=True)
def _on_a_step(
    lob,
    sig,
    R,
    H,
    W,
    memo,
    val,
    x,
    y,
    tx,
    ty,
    share,
    gathered,
    p,
    qs,
    ps,
    followed,
    follow_val,
):
    """The highlight of pixel (x, y), ``gathered`` over ``p``'s span (px
    each way) along a crest read as running (tx, ty) with ``share``,
    gathered on over a whole step of the pixel staircase along it where it
    is a shallow, straight crest that runs on. Only where _crest reads a
    step longer than FOLLOW_FROM. The crest is followed both ways
    (_follow): its direction is taken from where it was found, the step
    from that (_step, up to p's span_top), and the reach on towards it as
    far as it is surely the same crest, straight, all the way (the
    sureness at that many steps), times the crest's own share where it
    crosses the pixel's row (column). Gathered along the followed
    direction (_gather_even), and blended in as the step grows past the
    span by half a pixel to by one and a half (no further than 26 degrees
    from the rows or columns), as the sureness rises to 1/2, and where the pixel
    lies within FIND px of the crest's line (fading out over the next).
    ``p``: (span, span_top, gate, run_on)."""
    span, span_top, gate, run_on = p
    if not share > 0.0 or _step(tx, ty, span_top) <= FOLLOW_FROM:
        return gathered
    off, cx, cy, share0 = _follow(
        sig,
        R,
        H,
        W,
        memo,
        val,
        x,
        y,
        tx,
        ty,
        gate,
        run_on,
        qs,
        ps,
        followed,
        follow_val,
    )
    step = _step(cx, cy, span_top)
    sure = _at(qs, step * max(abs(cx), abs(cy)))
    blend = _smoothstep(span + 0.5, span + 1.5, step) * _smoothstep(0.0, 0.5, sure)
    blend *= _smoothstep(FIND + 0.5, FIND - 0.5, abs(off))
    if not blend > 0.0:
        return gathered
    reach = span * share + (step - span) * share0 * sure
    far = _gather_even(lob, R, H, W, x, y, cx, cy, reach)
    return gathered + blend * (far - gathered)


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
    gloss_span_max,
    crest_gate,
    run_on,
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
    way (_gather, ``gloss_taps`` points each side), and at a shallow slant
    on over one step of the pixel staircase, up to ``gloss_span_max``, as
    far as the crest, followed, runs on straight (_on_a_step, ``run_on``);
    then the pixel's own film (``cover``), the share of the lamp it sees
    and its mattness apply. Rows are worked out a few ahead of the one
    being lit and kept in buffers of a few rows (float64, uint8 for whether
    an own highlight lies near, with a running count per column of the rows
    within reach that have one, and _crest's work on the pixels it was asked
    for): no map the size of the canvas. Exact: with ``gloss_span`` 0 or
    the gate shut, the highlight is each pixel's own; with
    ``gloss_span_max`` no more than ``gloss_span``, or on a crest whose
    staircase step, as followed, is no longer than ``gloss_span`` + 0.5
    (a smooth straight crest along the rows or columns or within about
    0.6 degrees of them, or one more than about 26 degrees from both, for
    ``gloss_span`` 2), it is gathered over ``gloss_span`` alone; flat paint,
    paint gently modelled (beyond the cut-off), ``gloss`` 0 and pixels with
    no own highlight within ``ceil(max(gloss_span, gloss_span_max)) + 1``
    pixels are as they would be without gathering."""
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
    # the furthest a highlight is gathered
    span_top = max(gloss_span, gloss_span_max)
    # rows ahead of the one being lit that the gathering reads, and the rows
    # kept: as many behind. Following a shallow crest k = ceil(span_top)
    # steps (_follow) asks _crest at rows up to k + FIND + 3 away: the
    # crest is found within FIND + 0.5 rows of the pixel, the guess at k
    # steps lies within k rows of that and is rounded to the nearest row
    # (FIND + k + 1), and _peak asks the rows either side of the most
    # curved of the three round it (+ 2); _crest reads 2 rows further
    # (BINOMIAL): k + FIND + 5. A steep crest, and the taps of the
    # gathering (k rows away, and the next row interpolated), read less far
    lag = int(math.ceil(span_top)) + FIND + 5 if gather else 0
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
    # how many of the rows within lag of the one being lit have one (a
    # running count: what is gathered reads only within that lag-square)
    nearby = np.zeros(W, np.int64)
    # _crest_and_trace of the pixels a crest is followed through, each
    # worked out once (``memo``)
    memo = np.zeros((R + 1, W), np.uint8)
    crest_val = np.zeros((R + 1, 4, W))
    qs = np.ones(int(math.ceil(span_top)) + 1)  # how sure, steps along
    ps = np.zeros((2, qs.shape[0]))  # where the crest was found, both ways
    # the slope and sureness of the crest followed from a pixel, each worked
    # out once (_follow)
    followed = np.zeros((R + 1, W), np.uint8)
    follow_val = np.zeros((R + 1, W, qs.shape[0] + 1))
    gp_far = (gloss_span, span_top, crest_gate, run_on)
    for yy in range(H + lag):
        if gather and yy > 2 * lag:  # a row leaves the lag-square
            _tally(near[(yy - 2 * lag - 1) & R], nearby, -1)
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
                memo[r] = 0
                followed[r] = 0
                _near_row(lob[r], lag, near[r])
                _tally(near[r], nearby, 1)
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
                # any own highlight within reach (a lag-square)?
                if gather and nearby[x] > 0:
                    tx, ty, share, _ = _crest_at(
                        row_sig, R, H, W, memo, crest_val, x, y, crest_gate
                    )
                    reach = gloss_span * share
                    if reach > NO_REACH:
                        own = _gather(lob, R, H, W, x, y, tx, ty, reach, gloss_taps)
                    own = _on_a_step(
                        lob, row_sig, R, H, W, memo, crest_val, x, y, tx, ty, share,
                        own, gp_far, qs, ps, followed, follow_val,
                    )  # fmt: skip
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
