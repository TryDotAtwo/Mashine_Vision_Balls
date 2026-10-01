"""Replay user-supplied video configurations through ClickHouse, without private paths."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from mvb.pipeline import VideoAnalysisConfig,DetectionConfig
from mvb.series_analysis import analyze_series
from mvb.clickhouse_store import ClickHouseStore

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    args=parser.parse_args()
    entries=json.loads(args.manifest.read_text(encoding="utf8"))
    for item in entries:
        identifier=item["id"]
        if not identifier or Path(identifier).name!=identifier or identifier in (".",".."):
            raise ValueError("Series id must be a plain directory name")
        target=args.output_root/identifier
        if target.exists():
            raise ValueError(f"Use a fresh output directory: {target}")
        detection=DetectionConfig(**item.get("detection",{}))
        result=analyze_series(VideoAnalysisConfig(item["video"],str(target),frame_step=item.get("frame_step",1)),detection,item.get("measurement_mode","pixel"),calibration_path=item.get("calibration"),cell_mask=item.get("cell_mask",False),tracking_distance_px=item.get("tracking_distance_px",100),tracking_max_missed=item.get("tracking_max_missed",10),storage=ClickHouseStore(),annotated_video=item.get("annotated_video",True))
        print(identifier,json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
