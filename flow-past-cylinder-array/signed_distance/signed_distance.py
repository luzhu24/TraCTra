import numpy as np

TAU = 2.0 * np.pi

def _as2(p):
    p = np.asarray(p, dtype=float)
    if p.shape != (2,):
        raise ValueError("Point must be shape (2,).")
    return p

def _norm_angle(a): return np.mod(a, TAU)
def _sweep_ccw(a0, a1): return np.mod(a1 - a0, TAU)

def _angle_in_sweep(theta, a0, a1, ccw=True):
    theta = _norm_angle(theta)
    a0 = _norm_angle(a0)
    a1 = _norm_angle(a1)
    if ccw:
        sweep = _sweep_ccw(a0, a1)
        rel = np.mod(theta - a0, TAU)
        return rel <= sweep
    else:
        sweep = _sweep_ccw(a1, a0)
        rel = np.mod(a0 - theta, TAU)
        return rel <= sweep


class LineSeg:
    def __init__(self, p0, p1):
        self.p0 = _as2(p0)
        self.p1 = _as2(p1)

    def shifted(self, dx, dy):
        s = np.array([dx, dy], float)
        return LineSeg(self.p0 + s, self.p1 + s)

    def sample(self, tol, include_end=False):
        L = np.linalg.norm(self.p1 - self.p0)
        n = max(2, int(np.ceil(L / max(tol, 1e-12))) + 1)
        t = np.linspace(0.0, 1.0, n, endpoint=include_end)
        return self.p0[None, :] * (1 - t)[:, None] + self.p1[None, :] * t[:, None]

    def distance(self, X, Y):
        Ax, Ay = self.p0
        Bx, By = self.p1
        ABx, ABy = (Bx - Ax), (By - Ay)
        denom = ABx * ABx + ABy * ABy + 1e-30

        APx = X - Ax
        APy = Y - Ay
        t = (APx * ABx + APy * ABy) / denom
        t = np.clip(t, 0.0, 1.0)

        Cx = Ax + t * ABx
        Cy = Ay + t * ABy
        return np.hypot(X - Cx, Y - Cy)


class ArcSeg:
    def __init__(self, *, center, ccw=True, p0=None, p1=None, radius=None, a0=None, a1=None):
        self.center = _as2(center)
        self.ccw = bool(ccw)

        if p0 is not None and p1 is not None:
            self.p0 = _as2(p0)
            self.p1 = _as2(p1)
            v0 = self.p0 - self.center
            v1 = self.p1 - self.center
            r0 = np.linalg.norm(v0)
            r1 = np.linalg.norm(v1)
            self.r  = float(0.5 * (r0 + r1)) if radius is None else float(radius)
            self.a0 = float(np.arctan2(v0[1], v0[0])) if a0 is None else float(a0)
            self.a1 = float(np.arctan2(v1[1], v1[0])) if a1 is None else float(a1)
        elif radius is not None and a0 is not None and a1 is not None:
            self.r  = float(radius)
            self.a0 = float(a0)
            self.a1 = float(a1)
            self.p0 = self.center + self.r * np.array([np.cos(self.a0), np.sin(self.a0)])
            self.p1 = self.center + self.r * np.array([np.cos(self.a1), np.sin(self.a1)])
        else:
            raise ValueError("ArcSeg needs either (p0,p1) or (radius,a0,a1).")

    def shifted(self, dx, dy):
        s = np.array([dx, dy], float)
        return ArcSeg(center=self.center + s, ccw=self.ccw, radius=self.r, a0=self.a0, a1=self.a1)

    def sample(self, tol, include_end=False):
        r = max(self.r, 1e-12)
        dtheta = 2.0 * np.arcsin(min(1.0, max(tol, 1e-12) / (2.0 * r)))
        dtheta = max(dtheta, 1e-3)

        a0 = _norm_angle(self.a0)
        a1 = _norm_angle(self.a1)
        if self.ccw:
            sweep = _sweep_ccw(a0, a1)
            n = max(2, int(np.ceil(sweep / dtheta)) + 1)
            s = np.linspace(0.0, sweep, n, endpoint=include_end)
            ang = a0 + s
        else:
            sweep = _sweep_ccw(a1, a0)
            n = max(2, int(np.ceil(sweep / dtheta)) + 1)
            s = np.linspace(0.0, sweep, n, endpoint=include_end)
            ang = a0 - s

        return self.center[None, :] + self.r * np.c_[np.cos(ang), np.sin(ang)]

    def distance(self, X, Y):
        cx, cy = self.center
        dx = X - cx
        dy = Y - cy
        rho = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)

        on_arc = _angle_in_sweep(theta, self.a0, self.a1, ccw=self.ccw)
        d_radial = np.abs(rho - self.r)

        d_end0 = np.hypot(X - self.p0[0], Y - self.p0[1])
        d_end1 = np.hypot(X - self.p1[0], Y - self.p1[1])
        d_end = np.minimum(d_end0, d_end1)

        return np.where(on_arc, d_radial, d_end)


def _ray_cast_inside(X, Y, P):
    P = np.asarray(P, float)
    inside = np.zeros_like(X, dtype=bool)
    x = X
    y = Y
    M = P.shape[0]
    for i in range(M):
        x0, y0 = P[i]
        x1, y1 = P[(i + 1) % M]
        cond = ((y0 > y) != (y1 > y))
        x_int = x0 + (x1 - x0) * (y - y0) / (y1 - y0 + 1e-30)
        inside ^= (cond & (x < x_int))
    return inside


class SignedDistanceFromSegments2D:
    def __init__(self, *, x=None, y=None, Nx=256, Ny=256, Lx=2.0, Ly=2.0, center=(0.0, 0.0)):
        """Non-uniform tensor-product grid."""
        if x is None:
            self.Nx = int(Nx)
            self.Lx = float(Lx)
            cx = float(center[0])
            dx = self.Lx / self.Nx
            self.x = (np.arange(self.Nx) + 0.5) * dx - self.Lx / 2 + cx
        else:
            self.x = np.asarray(x, float)
            self.Nx = self.x.size
            self.Lx = float(Lx)

        if y is None:
            self.Ny = int(Ny)
            self.Ly = float(Ly)
            cy = float(center[1])
            dy = self.Ly / self.Ny
            self.y = (np.arange(self.Ny) + 0.5) * dy - self.Ly / 2 + cy
        else:
            self.y = np.asarray(y, float)
            self.Ny = self.y.size
            self.Ly = float(Ly)

        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        hx = np.min(np.abs(np.diff(self.x))) if self.Nx > 1 else 1.0
        hy = np.min(np.abs(np.diff(self.y))) if self.Ny > 1 else 1.0
        self.h_min = min(hx, hy)

        self.objects = []
        self.polylines = None
        self.S = None
        self.phi = None
        self.inside = None
        self.dist = None

    @staticmethod
    def _parse_segments(segments):
        segs = []
        for s in segments:
            if isinstance(s, (LineSeg, ArcSeg)):
                segs.append(s)
            else:
                st = s.get("type", None)
                if st == "line":
                    segs.append(LineSeg(s["p0"], s["p1"]))
                elif st == "arc":
                    segs.append(
                        ArcSeg(
                            center=s["center"],
                            ccw=s.get("ccw", True),
                            p0=s.get("p0", None),
                            p1=s.get("p1", None),
                            radius=s.get("radius", None),
                            a0=s.get("a0", None),
                            a1=s.get("a1", None),
                        )
                    )
                else:
                    raise ValueError(f"Unknown segment type: {st!r}")
        return segs

    def set_objects(self, objects):
        if not isinstance(objects, (list, tuple)) or len(objects) == 0:
            raise ValueError("objects must be a non-empty list")
        first = objects[0]
        if isinstance(first, (dict, LineSeg, ArcSeg)):
            self.objects = [self._parse_segments(objects)]
        else:
            self.objects = [self._parse_segments(obj) for obj in objects]
        return self

    def set_segments(self, segments):
        return self.set_objects(segments)

    @staticmethod
    def _build_polyline_from_segments(segments, tol):
        pts = []
        for seg in segments:
            samp = seg.sample(tol, include_end=False)
            if len(pts) == 0:
                pts.append(samp[0])
                pts.extend(samp[1:])
            else:
                if np.allclose(pts[-1], samp[0]):
                    pts.extend(samp[1:])
                else:
                    pts.extend(samp)
        if len(segments) > 0:
            pts.append(segments[-1].p1)

        P = np.asarray(pts, float)
        if P.shape[0] >= 2 and np.allclose(P[0], P[-1]):
            P = P[:-1]
        return P

    def _unsigned_distance(self, segments):
        dmin = np.inf * np.ones_like(self.X, dtype=float)
        for seg in segments:
            dmin = np.minimum(dmin, seg.distance(self.X, self.Y))
        return dmin

    @staticmethod
    def _spacings_1d(coord, L, periodic):
        coord = np.asarray(coord, float)
        if periodic:
            dp = np.roll(coord, -1) - coord
            dm = coord - np.roll(coord, 1)
            dp[dp <= 0] += L
            dm[dm <= 0] += L
        else:
            d = np.diff(coord)
            if d.size == 0:
                dp = np.ones_like(coord)
                dm = np.ones_like(coord)
            else:
                dp = np.empty_like(coord)
                dm = np.empty_like(coord)
                dp[:-1] = d; dp[-1] = d[-1]
                dm[1:]  = d; dm[0]  = d[0]
        return dp, dm

    @staticmethod
    def _shift_plus(d, axis, periodic):
        if periodic:
            return np.roll(d, -1, axis=axis)
        out = np.empty_like(d)
        if axis == 0:
            out[:-1, :] = d[1:, :]; out[-1, :] = d[-1, :]
        else:
            out[:, :-1] = d[:, 1:]; out[:, -1] = d[:, -1]
        return out

    @staticmethod
    def _shift_minus(d, axis, periodic):
        if periodic:
            return np.roll(d, 1, axis=axis)
        out = np.empty_like(d)
        if axis == 0:
            out[1:, :] = d[:-1, :]; out[0, :] = d[0, :]
        else:
            out[:, 1:] = d[:, :-1]; out[:, 0] = d[:, 0]
        return out

    def _godunov_grad_mag(self, d, signS, *, periodic_x, periodic_y):
        dxp, dxm = self._spacings_1d(self.x, self.Lx, periodic_x)
        dyp, dym = self._spacings_1d(self.y, self.Ly, periodic_y)

        d_xp = self._shift_plus(d, axis=0, periodic=periodic_x)
        d_xm = self._shift_minus(d, axis=0, periodic=periodic_x)
        d_yp = self._shift_plus(d, axis=1, periodic=periodic_y)
        d_ym = self._shift_minus(d, axis=1, periodic=periodic_y)

        Dx_plus  = (d_xp - d) / (dxp[:, None] + 1e-30)
        Dx_minus = (d - d_xm) / (dxm[:, None] + 1e-30)
        Dy_plus  = (d_yp - d) / (dyp[None, :] + 1e-30)
        Dy_minus = (d - d_ym) / (dym[None, :] + 1e-30)

        grad2 = np.empty_like(d)

        mask_pos = signS > 0
        if np.any(mask_pos):
            a = np.maximum(Dx_minus[mask_pos], 0.0)**2 + np.minimum(Dx_plus[mask_pos], 0.0)**2
            b = np.maximum(Dy_minus[mask_pos], 0.0)**2 + np.minimum(Dy_plus[mask_pos], 0.0)**2
            grad2[mask_pos] = a + b

        mask_neg = signS < 0
        if np.any(mask_neg):
            a = np.minimum(Dx_minus[mask_neg], 0.0)**2 + np.maximum(Dx_plus[mask_neg], 0.0)**2
            b = np.minimum(Dy_minus[mask_neg], 0.0)**2 + np.maximum(Dy_plus[mask_neg], 0.0)**2
            grad2[mask_neg] = a + b

        mask_zero = ~(mask_pos | mask_neg)
        if np.any(mask_zero):
            d_dx_c = np.zeros_like(d)
            d_dy_c = np.zeros_like(d)

            if periodic_x:
                dx_c = 0.5 * (dxp + dxm)
                d_dx_c = (np.roll(d, -1, axis=0) - np.roll(d, 1, axis=0)) / (dx_c[:, None] + 1e-30)
            else:
                if self.Nx >= 3:
                    denom = (self.x[2:] - self.x[:-2])[:, None]
                    d_dx_c[1:-1, :] = (d[2:, :] - d[:-2, :]) / (denom + 1e-30)
                if self.Nx >= 2:
                    d_dx_c[0, :]  = (d[1, :] - d[0, :]) / (dxp[0] + 1e-30)
                    d_dx_c[-1, :] = (d[-1, :] - d[-2, :]) / (dxm[-1] + 1e-30)

            if periodic_y:
                dy_c = 0.5 * (dyp + dym)
                d_dy_c = (np.roll(d, -1, axis=1) - np.roll(d, 1, axis=1)) / (dy_c[None, :] + 1e-30)
            else:
                if self.Ny >= 3:
                    denom = (self.y[2:] - self.y[:-2])[None, :]
                    d_dy_c[:, 1:-1] = (d[:, 2:] - d[:, :-2]) / (denom + 1e-30)
                if self.Ny >= 2:
                    d_dy_c[:, 0]  = (d[:, 1] - d[:, 0]) / (dyp[0] + 1e-30)
                    d_dy_c[:, -1] = (d[:, -1] - d[:, -2]) / (dym[-1] + 1e-30)

            grad2[mask_zero] = d_dx_c[mask_zero]**2 + d_dy_c[mask_zero]**2

        return np.sqrt(grad2)

    def reinitialize_tapered(self, S, *, n_steps=50, cfl=0.3, W_inner_factor=20.0, W_outer_factor=20.0,
                             periodic_x=True, periodic_y=True):
        d = S.copy()
        eps = 1e-6
        signS = S / np.sqrt(S**2 + eps)

        W1 = W_inner_factor * self.h_min
        W2 = W_outer_factor * self.h_min
        absS = np.abs(S)

        alpha = np.zeros_like(S)
        alpha[absS <= W1] = 1.0
        mid = (absS > W1) & (absS < W2)
        s = (absS[mid] - W1) / (W2 - W1 + 1e-30)
        alpha[mid] = 0.5 * (1.0 + np.cos(np.pi * s))

        dtau = cfl * self.h_min
        for _ in range(int(n_steps)):
            grad_mag = self._godunov_grad_mag(d, signS, periodic_x=periodic_x, periodic_y=periodic_y)
            d = d - dtau * alpha * signS * (grad_mag - 1.0)
        return d

    def build(self, *, tol=None, inside_positive=True,
              periodic=False, shifts=(-1, 0, 1),
              periodic_x=None, periodic_y=None,
              shifts_x=None, shifts_y=None, #periodic copy of images
              union_mode="outer",
              do_reinit=None, reinit_kwargs=None):
        """Build phi on the grid."""
        if tol is None:
            tol = 0.5 * self.h_min
        if len(self.objects) == 0:
            raise RuntimeError("No geometry set. Call set_objects(...).")

        # Backward compatibility
        if periodic_x is None:
            periodic_x = bool(periodic)
        if periodic_y is None:
            periodic_y = bool(periodic)

        if shifts_x is None:
            shifts_x = shifts if periodic_x else (0,)
        if shifts_y is None:
            shifts_y = shifts if periodic_y else (0,)

        self.polylines = [self._build_polyline_from_segments(obj, tol) for obj in self.objects]
        nobj = len(self.objects)

        if do_reinit is None:
            do_reinit = (union_mode == "outer" and nobj > 1)

        if union_mode not in ("outer", "all"):
            raise ValueError("union_mode must be 'outer' or 'all'")

        def _shift_obj(obj, dx, dy):
            if dx == 0.0 and dy == 0.0:
                return obj
            return [seg.shifted(dx, dy) for seg in obj]

        if union_mode == "all":
            dist = np.inf * np.ones_like(self.X, float)
            inside = np.zeros_like(self.X, bool)

            for i in shifts_x:
                for j in shifts_y:
                    dx = float(i) * self.Lx if periodic_x else 0.0
                    dy = float(j) * self.Ly if periodic_y else 0.0
                    shift = np.array([dx, dy], float)

                    for obj, P in zip(self.objects, self.polylines):
                        obj_ij = _shift_obj(obj, dx, dy)
                        dist = np.minimum(dist, self._unsigned_distance(obj_ij))
                        inside |= _ray_cast_inside(self.X, self.Y, P + shift[None, :])

            phi = np.where(inside, dist, -dist)

        else:
            S = -np.inf * np.ones_like(self.X, float)
            for i in shifts_x:
                for j in shifts_y:
                    dx = float(i) * self.Lx if periodic_x else 0.0
                    dy = float(j) * self.Ly if periodic_y else 0.0
                    shift = np.array([dx, dy], float)

                    for obj, P in zip(self.objects, self.polylines):
                        obj_ij = _shift_obj(obj, dx, dy)
                        dist_i = self._unsigned_distance(obj_ij)
                        inside_i = _ray_cast_inside(self.X, self.Y, P + shift[None, :])
                        phi_i = np.where(inside_i, dist_i, -dist_i)
                        S = np.maximum(S, phi_i)

            self.S = S
            phi = S
            if do_reinit:
                kw = {} if reinit_kwargs is None else dict(reinit_kwargs)
                phi = self.reinitialize_tapered(phi, periodic_x=periodic_x, periodic_y=periodic_y, **kw)

        if not inside_positive:
            phi = -phi

        self.phi = phi
        self.dist = np.abs(phi)
        self.inside = phi > 0
        return self


# ---- helpers for examples ----
def rectangle_segments(cx, cy, w, h):

    x0, x1 = cx - w/2, cx + w/2
    y0, y1 = cy - h/2, cy + h/2
    return [
        {"type":"line", "p0":(x0, y1), "p1":(x1, y1)},
        {"type":"line", "p0":(x1, y1), "p1":(x1, y0)},
        {"type":"line", "p0":(x1, y0), "p1":(x0, y0)},
        {"type":"line", "p0":(x0, y0), "p1":(x0, y1)},
    ]


def rounded_rectangle_segments(cx, cy, w, h, r):
    if r < 0 or r > 0.5 * min(w, h):
        raise ValueError("Need 0 <= r <= min(w,h)/2")
    x0, x1 = cx - w/2, cx + w/2
    y0, y1 = cy - h/2, cy + h/2
    Ctr = (x1 - r, y1 - r)
    Cbr = (x1 - r, y0 + r)
    Cbl = (x0 + r, y0 + r)
    Ctl = (x0 + r, y1 - r)
    return [
        {"type":"line", "p0":(x0 + r, y1),     "p1":(x1 - r, y1)},
        {"type":"arc",  "center":Ctr, "radius":r, "a0":np.pi/2, "a1":0.0,        "ccw":False},
        {"type":"line", "p0":(x1,     y1 - r), "p1":(x1,     y0 + r)},
        {"type":"arc",  "center":Cbr, "radius":r, "a0":0.0,     "a1":-np.pi/2,   "ccw":False},
        {"type":"line", "p0":(x1 - r, y0),     "p1":(x0 + r, y0)},
        {"type":"arc",  "center":Cbl, "radius":r, "a0":-np.pi/2,"a1":-np.pi,     "ccw":False},
        {"type":"line", "p0":(x0,     y0 + r), "p1":(x0,     y1 - r)},
        {"type":"arc",  "center":Ctl, "radius":r, "a0":np.pi,   "a1":np.pi/2,    "ccw":False},
    ]




