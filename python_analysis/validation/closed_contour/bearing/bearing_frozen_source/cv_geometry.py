"""Circle fitting and edge-based refinement for robust diameter estimation."""

from __future__ import annotations

import math

import cv2
import numpy as np


def fit_circle_kasa(points_xy: np.ndarray) -> tuple[float, float, float]:
    """
    Algebraic (Kåsa) circle fit: minimize sum (x^2+y^2 - a x - b y - c)^2 in least squares.

    Returns (cx, cy, r) in pixel coordinates.
    """
    if points_xy.shape[0] < 3:
        raise ValueError("At least 3 points are required for circle fit")
    x = points_xy[:, 0].astype(np.float64)
    y = points_xy[:, 1].astype(np.float64)
    z = x * x + y * y
    a_mat = np.column_stack([x, y, np.ones(len(x))])
    coef, _, rank, _ = np.linalg.lstsq(a_mat, z, rcond=None)
    if rank < 3:
        raise ValueError("Degenerate point set for circle fit")
    cx = float(coef[0] / 2.0)
    cy = float(coef[1] / 2.0)
    r_sq = coef[2] + cx * cx + cy * cy
    if r_sq <= 0:
        raise ValueError("Non-positive radius in algebraic circle fit")
    return cx, cy, float(math.sqrt(r_sq))


def _canny_edges(gray: np.ndarray, low_ratio: float = 0.66, high_ratio: float = 1.33) -> np.ndarray:
    med = float(np.median(gray))
    low = max(1.0, med * low_ratio)
    high = max(low + 1.0, med * high_ratio)
    return cv2.Canny(gray, int(low), int(high))


def refine_circle_from_edges(
    gray: np.ndarray,
    circle: tuple[int, int, int],
    band_px: int,
    min_edge_points: int = 24,
    round_result: bool = True,
    canny_thresholds: tuple[int, int] | None = None,
) -> tuple[float, float, float] | tuple[int, int, int]:
    """
    Refine an approximate integer circle by fitting to Canny edge pixels in an annulus.

    Falls back to the input circle if too few edge points are found.
    """
    cx_i, cy_i, r_i = circle
    h, w = gray.shape[:2]
    margin = int(math.ceil(r_i + band_px + 3))
    left, top = max(0,int(cx_i)-margin), max(0,int(cy_i)-margin)
    right, bottom = min(w,int(cx_i)+margin+1), min(h,int(cy_i)+margin+1)
    crop = gray[top:bottom,left:right]
    edges = cv2.Canny(crop,*canny_thresholds) if canny_thresholds else _canny_edges(crop)
    yy, xx = np.ogrid[top:bottom, left:right]
    dist = np.sqrt((xx.astype(np.float64) - float(cx_i)) ** 2 + (yy.astype(np.float64) - float(cy_i)) ** 2)
    inner = max(1.0, float(r_i) - float(band_px))
    outer = float(r_i) + float(band_px)
    mask = (dist >= inner) & (dist <= outer)
    ring = edges.astype(bool) & mask
    pts_y, pts_x = np.where(ring)
    if pts_y.size < min_edge_points:
        return circle
    pts_xy = np.column_stack([pts_x.astype(np.float64)+left, pts_y.astype(np.float64)+top])
    try:
        cx, cy, r = fit_circle_kasa(pts_xy)
    except ValueError:
        return circle
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(r)):
        return circle
    if r < 2.0 or r > 0.95 * min(h, w) / 2.0:
        return circle
    shift = math.hypot(cx - cx_i, cy - cy_i)
    if shift > 0.35 * r_i:
        return circle
    return (int(round(cx)), int(round(cy)), int(round(r))) if round_result else (cx, cy, r)


def annulus_edge_support_score(
    edges: np.ndarray,
    circle: tuple[int, int, int],
    band_px: int = 1,
) -> float:
    """Mean edge strength on a thin ring (0..1)."""
    height, width = edges.shape[:2]
    x, y, r = circle
    samples: list[float] = []
    for offset in range(-band_px, band_px + 1):
        rr = max(1, r + offset)
        angles = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
        xs = np.clip(np.round(x + rr * np.cos(angles)).astype(int), 0, width - 1)
        ys = np.clip(np.round(y + rr * np.sin(angles)).astype(int), 0, height - 1)
        samples.append(float(edges[ys, xs].mean() / 255.0))
    return float(np.mean(samples))


from dataclasses import dataclass
from scipy.optimize import least_squares


@dataclass(frozen=True)
class RadialCircleFit:
    circle: tuple[float, float, float] | None
    status: str
    coverage: float
    residual_rms_px: float | None = None
    axis_ratio: float | None = None
    max_unsupported_arc_deg: float | None = None
    min_quadrant_coverage: float | None = None


def refine_circle_radial(gray, circle, band_px=6, polarity="either",
                         min_coverage=0.65, max_axis_ratio=1.15, min_gradient=0.2):
    """Subpixel radial gradients followed by robust geometric least squares.

    Rejections are explicit; a Hough seed is never silently returned as a fit.
    Coverage and residuals describe the image fit, not physical uncertainty.
    """
    if polarity not in ("either", "rising", "falling"):
        raise ValueError("Unknown radial gradient polarity")
    if not math.isfinite(min_gradient) or min_gradient<=0:
        raise ValueError("Invalid gradient floor")
    if not math.isfinite(band_px) or band_px < 2 or not 0 < min_coverage <= 1 or max_axis_ratio < 1:
        raise ValueError("Invalid radial fitting limits")
    if polarity == "either":
        # A silhouette has one edge direction. Mixing rising and falling peaks
        # around a circle can manufacture a contour from unrelated reflections.
        fits = [refine_circle_radial(gray, circle, band_px, direction,
                    min_coverage, max_axis_ratio, min_gradient)
                for direction in ("rising", "falling")]
        accepted = [fit for fit in fits if fit.circle is not None]
        if accepted:
            return max(accepted, key=lambda fit: fit.coverage
                - 0.05 * abs(fit.circle[2] - circle[2]) / max(circle[2], 1)
                - 0.02 * fit.residual_rms_px)
        return max(fits, key=lambda fit: fit.coverage)
    x,y,r = map(float,circle)
    h,w=gray.shape[:2]
    if not all(map(math.isfinite,(x,y,r))) or r < 3:
        return RadialCircleFit(None,"invalid_seed",0)
    if min(x-r,y-r,w-1-x-r,h-1-y-r) < 2:
        return RadialCircleFit(None,"clipped_contour",0)
    margin=int(math.ceil(r+band_px+4))
    left,top=max(0,int(x)-margin),max(0,int(y)-margin)
    right,bottom=min(w,int(x)+margin+1),min(h,int(y)+margin+1)
    image=cv2.GaussianBlur(gray[top:bottom,left:right].astype(np.float32),(5,5),1.0)
    gx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3)/8
    gy=cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3)/8
    n=int(np.clip(math.ceil(2*math.pi*r),96,720))
    angles=np.arange(n)*2*math.pi/n
    ux,uy=np.cos(angles),np.sin(angles)
    offsets=np.arange(max(-band_px, 2-r),band_px+0.25,0.5)
    radii=r+offsets
    mx=(x-left+ux[:,None]*radii).astype(np.float32)
    my=(y-top+uy[:,None]*radii).astype(np.float32)
    dx=cv2.remap(gx,mx,my,cv2.INTER_LINEAR)
    dy=cv2.remap(gy,mx,my,cv2.INTER_LINEAR)
    radial=dx*ux[:,None]+dy*uy[:,None]
    strength=np.abs(radial) if polarity=="either" else radial if polarity=="rising" else -radial
    aligned=np.abs(radial)/(np.hypot(dx,dy)+1e-6)
    # A weak distance prior resolves neighbouring edges without averaging both.
    scored=np.where(aligned>=0.75,strength,-1)*(1-0.15*np.abs(offsets)/band_px)
    idx=np.argmax(scored,axis=1)
    peak=strength[np.arange(n),idx]
    threshold=max(min_gradient,float(np.median(peak))*0.15)
    valid=(idx>0)&(idx<len(offsets)-1)&(peak>=threshold)&(aligned[np.arange(n),idx]>=0.75)
    coverage=float(valid.mean())
    if coverage<min_coverage:
        return RadialCircleFit(None,"insufficient_radial_coverage",coverage)
    rows=np.flatnonzero(valid); k=idx[valid]
    a,b,c=strength[rows,k-1],strength[rows,k],strength[rows,k+1]
    denom=a-2*b+c
    delta=np.divide(0.5*(a-c),denom,out=np.zeros_like(b),where=np.abs(denom)>1e-6)
    rr=r+offsets[k]+np.clip(delta,-0.5,0.5)*0.5
    points=np.column_stack([x+rr*ux[valid],y+rr*uy[valid]])
    def residual(v):
        return np.hypot(points[:,0]-v[0],points[:,1]-v[1])-v[2]
    limit=float(band_px)
    result=least_squares(residual,[x,y,r],bounds=([x-limit,y-limit,max(2,r-limit)],[x+limit,y+limit,r+limit]),
                         loss="soft_l1",f_scale=1.0,max_nfev=60)
    errors=residual(result.x)
    med=float(np.median(errors)); mad=max(.25,1.4826*float(np.median(np.abs(errors-med))))
    inliers=np.abs(errors-med)<=max(1.5,3*mad)
    coverage=float(inliers.sum()/n)
    rms=float(np.sqrt(np.mean(errors[inliers]**2))) if inliers.any() else None
    if not result.success or coverage<min_coverage:
        return RadialCircleFit(None,"unstable_circle_fit",coverage,rms)
    if rms is None or rms>max(1.5,0.025*r):
        return RadialCircleFit(None,"noncircular_residual",coverage,rms)
    # Overall coverage alone can accept a long arc of a vessel or a glint.
    # Require support on every side, and reject large unsupported angular gaps.
    support_angles=np.mod(np.arctan2(points[inliers,1]-result.x[1],
                                    points[inliers,0]-result.x[0]),2*math.pi)
    ordered=np.sort(support_angles)
    gap=float(np.max(np.diff(np.r_[ordered,ordered[0]+2*math.pi]))*180/math.pi)
    quadrant=float(min(np.sum((support_angles>=q*math.pi/2)&
                             (support_angles<(q+1)*math.pi/2))/(n/4)
                       for q in range(4)))
    # Illumination can make one quadrant weak while leaving short distributed
    # gaps. Reject an empty side rather than requiring uniform contrast.
    if gap>90 or quadrant<0.05:
        return RadialCircleFit(None,"open_or_localized_contour",coverage,rms,
                               max_unsupported_arc_deg=gap,min_quadrant_coverage=quadrant)
    # Outliers must not continue to influence a diameter after being rejected.
    clean=points[inliers]
    final=least_squares(lambda v:np.hypot(clean[:,0]-v[0],clean[:,1]-v[1])-v[2],
                        result.x,loss="soft_l1",f_scale=1.0,max_nfev=40)
    if not final.success or np.linalg.norm(final.x[:2]-[x,y])>band_px or abs(final.x[2]-r)>band_px:
        return RadialCircleFit(None,"unstable_inlier_fit",coverage,rms,
                               max_unsupported_arc_deg=gap,min_quadrant_coverage=quadrant)
    result=final
    rms=float(np.sqrt(np.mean((np.hypot(clean[:,0]-result.x[0],clean[:,1]-result.x[1])-result.x[2])**2)))
    ellipse=cv2.fitEllipseAMS(points[inliers].astype(np.float32))
    axes=ellipse[1]
    ratio=float(max(axes)/min(axes)) if min(axes)>0 else None
    if ratio is None or ratio>max_axis_ratio:
        return RadialCircleFit(None,"elliptical_contour",coverage,rms,ratio)
    fitted=tuple(float(v) for v in result.x)
    if min(fitted[0]-fitted[2],fitted[1]-fitted[2],w-1-fitted[0]-fitted[2],h-1-fitted[1]-fitted[2])<0:
        return RadialCircleFit(None,"clipped_fit",coverage,rms,ratio)
    return RadialCircleFit(fitted,"accepted",coverage,rms,ratio,gap,quadrant)
