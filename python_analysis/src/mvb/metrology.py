"""Conditional bearing checks from completed ClickHouse series.

The reference and validation halves share one physical object. These statistics
test scale transfer and repeatability, never independent absolute accuracy.
"""
from pathlib import Path
from dataclasses import asdict
import csv
import json
import math
import hashlib
import numpy as np


def ratio_standard_uncertainty(reference_mm,diameter_px,reference_px,covariance):
    """GUM linear propagation for d=D*N/R; covariance order is (D,N,R).

    This is the uncertainty of the supplied model, not a physical validation.
    Missing covariance is intentionally rejected instead of assuming zeros.
    """
    values=np.asarray([reference_mm,diameter_px,reference_px],dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values<=0):
        raise ValueError('Positive finite diameters required')
    cov=np.asarray(covariance,dtype=float)
    if cov.shape!=(3,3) or not np.all(np.isfinite(cov)) or not np.allclose(cov,cov.T,rtol=1e-12,atol=1e-15):
        raise ValueError('A finite symmetric 3x3 covariance matrix is required')
    if np.min(np.linalg.eigvalsh(cov)) < -1e-12*max(np.linalg.norm(cov),1e-12):
        raise ValueError('Covariance matrix must be positive semidefinite')
    d,n,r=values;gradient=np.array([n/r,d/r,-d*n/r**2])
    return math.sqrt(max(0,float(gradient@cov@gradient)))


def bearing_check(measurements, decoded_frames, reference_mm, block_frames=15,
                  bootstrap_samples=2000, seed=1729):
    if not math.isfinite(reference_mm) or reference_mm <= 0:
        raise ValueError("Reference diameter must be finite and positive")
    if decoded_frames < 24 or block_frames < 1 or bootstrap_samples < 100:
        raise ValueError("Insufficient frames or invalid bootstrap configuration")
    ids={int(r['ball_id']) for r in measurements}
    if len(ids)!=1:
        raise ValueError("Bearing check requires exactly one verified object track")
    # Preserve missing frames and the temporal spacing; do not compress gaps.
    pixels=np.full(decoded_frames,np.nan)
    for row in measurements:
        i=int(row['frame_index']);diam=float(row['ball_diameter_px'])
        if not 0<=i<decoded_frames or not math.isfinite(diam) or diam<=0:
            raise ValueError("Invalid observation")
        if math.isfinite(pixels[i]):
            raise ValueError("Multiple observations in one frame")
        pixels[i]=diam
    split=decoded_frames//2
    calibration,check=pixels[:split],pixels[split:]
    a,b=calibration[np.isfinite(calibration)],check[np.isfinite(check)]
    if min(len(a),len(b))<12:
        raise ValueError("At least twelve observations required in each half")
    n_ref=float(np.median(a));scale=reference_mm/n_ref
    d=b*scale;errors=d-reference_mm
    rng=np.random.default_rng(seed)
    def sample_blocks(series,length):
        length=min(length,len(series));count=math.ceil(len(series)/length)
        starts=rng.integers(0,len(series)-length+1,size=count)
        resampled=np.concatenate([series[s:s+length] for s in starts])[:len(series)]
        return resampled[np.isfinite(resampled)]
    sensitivity={}
    for length in sorted({min(block_frames,min(len(calibration),len(check))),
                          min(5,min(len(calibration),len(check))),
                          min(30,min(len(calibration),len(check)))}):
        draws=[]
        for _ in range(bootstrap_samples):
            aa,bb=sample_blocks(calibration,length),sample_blocks(check,length)
            if len(aa) and len(bb):
                draws.append(reference_mm*(float(np.mean(bb))/float(np.median(aa))-1))
        if len(draws)<bootstrap_samples*.9:
            raise ValueError("Too few nonempty bootstrap replicates")
        sensitivity[str(length)]={'conditional_u_mean_mm':float(np.std(draws,ddof=1)),
                                  'conditional_bias_ci95_mm':[float(v) for v in np.percentile(draws,[2.5,97.5])]}
    selected=sensitivity[str(min(block_frames,min(len(calibration),len(check))))]
    u_n=1/math.sqrt(6)  # two independent uniform +/-0.5-pixel boundaries
    summary={'frames':decoded_frames,'calibration_observations':len(a),
             'check_observations':len(b),'accepted_observations':len(a)+len(b),
             'reference_diameter_mm':reference_mm,'calibration_median_px':n_ref,
             'scale_mm_per_video_px':scale,'mean_check_mm':float(np.mean(d)),
             'median_check_mm':float(np.median(d)),
             'conditional_bias_mm':float(np.mean(errors)),
             'conditional_rmse_mm':float(np.sqrt(np.mean(errors**2))),
             'repeatability_std_mm':float(np.std(d,ddof=1)),
             'quantization_single_chord_u_mm':scale*u_n,
             'conditional_two_chord_u_mm':float(np.mean(d))*u_n*math.hypot(1/float(np.mean(b)),1/n_ref),
             'bootstrap_block_frames':block_frames,'bootstrap_samples':bootstrap_samples,
             'bootstrap_seed':seed,'bootstrap_sensitivity':sensitivity,**selected,
             'reference_standard_uncertainty_mm':None,'full_physical_uncertainty_mm':None,
             'scope':'Conditional same-object scale-transfer check, not independent accuracy. Bootstrap assumes locally stationary blocks; drift and acceptance selection can bias it. Quantization is a conventional single-chord sensitivity model, not measured subpixel error or an error bound.'}
    points=[{'frame_index':i,'diameter_px':float(v),'conditional_diameter_mm':float(v*scale),
             'role':'calibration' if i<split else 'check'}
            for i,v in enumerate(pixels) if math.isfinite(v)]
    return summary,points


def export_bearing_check(store,run_id,output_dir,reference_mm,block_frames=15):
    measurements,frames,tracks,summary,protocol=store.load(run_id)
    measurements=[asdict(row) for row in measurements]
    if summary.get('decode_status')!='complete':
        raise ValueError("A complete decoded series is required")
    result,points=bearing_check(measurements,int(protocol['decoded_frames']),reference_mm,block_frames)
    result.update(run_id=run_id,video_sha256=protocol['video_sha256'],source_sha256=protocol['source_sha256'])
    result['metrology_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'bearing_uncertainty.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    with (out/'bearing_check.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
    return result
