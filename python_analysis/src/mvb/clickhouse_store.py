"""ClickHouse-backed scientific series. Completed manifest is the publication gate.

No automatic insert retries: failed imports remain unpublished; a retry gets a new UUID.
Raw videos stay in immutable file storage and are linked by path and SHA-256.
"""
from dataclasses import asdict, fields
import json
import os
import re
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError, URLError
from uuid import uuid4, UUID

from .pipeline import FrameMeasurement


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class ClickHouseStore:
    def __init__(self, url=None, database=None):
        self.url = (url or os.environ.get("MVB_CLICKHOUSE_URL", "http://127.0.0.1:8124")).rstrip("/")
        self.database = database or os.environ.get("MVB_CLICKHOUSE_DATABASE", "mvb_analysis")
        parsed = urlsplit(self.url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("ClickHouse URL must be HTTP(S), without credentials or query")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.database):
            raise ValueError("Invalid ClickHouse database identifier")
        self.user = os.environ.get("MVB_CLICKHOUSE_USER", "mvb")
        self.password = os.environ.get("MVB_CLICKHOUSE_PASSWORD", "")
        if self.password and parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Remote ClickHouse with a password requires HTTPS")
        self.opener = build_opener(_NoRedirect())

    def request(self, sql, rows=None, params=None):
        query = {"query": sql, "wait_end_of_query": "1"}
        query.update({"param_" + k: str(v) for k, v in (params or {}).items()})
        body = ("\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in rows) + "\n").encode("utf-8") if rows is not None else b""
        req = Request(self.url + "/?" + urlencode(query), data=body, method="POST",
                      headers={"X-ClickHouse-User": self.user, "X-ClickHouse-Key": self.password})
        try:
            with self.opener.open(req, timeout=60) as response:
                return response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            # Do not leak request URLs, credentials, or private server response bodies.
            raise ValueError("ClickHouse request failed; series was not confirmed. Check server/access and retry with a new run ID.") from None

    def select(self, sql, params=None):
        data = self.request(sql + " FORMAT JSONEachRow", params=params)
        return [json.loads(line) for line in data.splitlines() if line.strip()]

    def initialize(self):
        db = self.database
        self.request(f"CREATE DATABASE IF NOT EXISTS {db}")
        cols = []
        for field in fields(FrameMeasurement):
            if field.name == "method":
                kind = "LowCardinality(String)"
            elif field.name in ("frame_index", "ball_valid_candidates", "ball_id"):
                kind = "UInt64"
            elif field.name in ("reference_radius_px", "reference_diameter_px", "scale_mm_per_px", "diameter_mm", "sigma_model_mm"):
                kind = "Nullable(Float64)"
            else:
                kind = "Float64"
            cols.append(field.name + " " + kind)
        self.request(f"CREATE TABLE IF NOT EXISTS {db}.python_measurements_v1 (run_id UUID, " + ", ".join(cols) + ") ENGINE=MergeTree ORDER BY (run_id, ball_id, frame_index)")
        # Backward-compatible migration; archival run IDs and rows are preserved.
        for name in ("scale_mm_per_px", "diameter_mm", "sigma_model_mm"):
            self.request(f"ALTER TABLE {db}.python_measurements_v1 MODIFY COLUMN {name} Nullable(Float64)")
        self.request(f"CREATE TABLE IF NOT EXISTS {db}.python_frames_v1 (run_id UUID, frame_index UInt64, time_s Float64, objects UInt64, scale_available Bool) ENGINE=MergeTree ORDER BY (run_id, frame_index)")
        self.request(f"CREATE TABLE IF NOT EXISTS {db}.python_runs_v1 (run_id UUID, created_at DateTime64(3,'UTC') DEFAULT now64(3), video_sha256 String, provenance_json String, summary_json String) ENGINE=MergeTree ORDER BY (run_id)")

    def save(self, measurements, frames, provenance, summary):
        self.initialize()
        run_id = str(uuid4())
        for table, rows in (("python_measurements_v1", [asdict(r) for r in measurements]),
                            ("python_frames_v1", frames)):
            for offset in range(0, len(rows), 10000):
                batch = [dict(r, run_id=run_id) for r in rows[offset:offset+10000]]
                self.request(f"INSERT INTO {self.database}.{table} FORMAT JSONEachRow", batch)
        counts = self.select(f"SELECT (SELECT count() FROM {self.database}.python_measurements_v1 WHERE run_id={{id:UUID}}) AS measurements, (SELECT count() FROM {self.database}.python_frames_v1 WHERE run_id={{id:UUID}}) AS frames", {"id": run_id})[0]
        if int(counts["measurements"]) != len(measurements) or int(counts["frames"]) != len(frames):
            raise ValueError("ClickHouse row-count verification failed; run remains unpublished")
        # Insert manifest last: incomplete batches are never selected as completed runs.
        self.request(f"INSERT INTO {self.database}.python_runs_v1 FORMAT JSONEachRow", [{"run_id": run_id, "video_sha256": provenance["video_sha256"],
                      "provenance_json": json.dumps(provenance, ensure_ascii=False, allow_nan=False),
                      "summary_json": json.dumps(summary, ensure_ascii=False, allow_nan=False)}])
        self.load(run_id)  # Verify manifest readback, not just the HTTP acknowledgement.
        return run_id

    def load(self, run_id):
        run_id = str(UUID(run_id))
        params = {"id": run_id}
        result = self.select(f"SELECT provenance_json, summary_json FROM {self.database}.python_runs_v1 WHERE run_id={{id:UUID}}", params)
        if len(result) != 1:
            raise ValueError("Completed ClickHouse run not found or duplicate manifest")
        provenance = json.loads(result[0]["provenance_json"])
        summary = json.loads(result[0]["summary_json"])
        columns = ", ".join(f.name for f in fields(FrameMeasurement))
        measurements = [FrameMeasurement(**r) for r in self.select(f"SELECT {columns} FROM {self.database}.python_measurements_v1 WHERE run_id={{id:UUID}} ORDER BY frame_index, ball_id", params)]
        frames = self.select(f"SELECT frame_index,time_s,objects,scale_available FROM {self.database}.python_frames_v1 WHERE run_id={{id:UUID}} ORDER BY frame_index", params)
        tracks = self.select(f"SELECT ball_id, count() AS observations, min(time_s) AS time_start_s, max(time_s) AS time_end_s, (quantileExactLow(0.5)(diameter_mm)+quantileExactHigh(0.5)(diameter_mm))/2 AS diameter_mm_median, avg(diameter_mm) AS diameter_mm_mean, if(count()>1,stddevSamp(diameter_mm),NULL) AS diameter_mm_std, avg(sigma_model_mm) AS sigma_model_mm_mean, (quantileExactLow(0.5)(ball_diameter_px)+quantileExactHigh(0.5)(ball_diameter_px))/2 AS diameter_px_median, avg(ball_diameter_px) AS diameter_px_mean, if(count()>1,stddevSamp(ball_diameter_px),NULL) AS diameter_px_std FROM {self.database}.python_measurements_v1 WHERE run_id={{id:UUID}} GROUP BY ball_id ORDER BY ball_id", params)
        eligible = [r for r in tracks if int(r["observations"]) >= summary["min_track_observations"]]
        aggregate = self.select(f"SELECT count() AS n, if(count()>0,(quantileExactLow(0.5)(d)+quantileExactHigh(0.5)(d))/2,NULL) AS median, if(count()>1,stddevSamp(d),NULL) AS std FROM (SELECT (quantileExactLow(0.5)(diameter_mm)+quantileExactHigh(0.5)(diameter_mm))/2 AS d FROM {self.database}.python_measurements_v1 WHERE run_id={{id:UUID}} GROUP BY ball_id HAVING count()>={{minimum:UInt64}})", {"id": run_id, "minimum": summary["min_track_observations"]})[0]
        pixel_aggregate = self.select(f"SELECT count() AS n, if(count()>0,(quantileExactLow(0.5)(d)+quantileExactHigh(0.5)(d))/2,NULL) AS median, if(count()>1,stddevSamp(d),NULL) AS std FROM (SELECT (quantileExactLow(0.5)(ball_diameter_px)+quantileExactHigh(0.5)(ball_diameter_px))/2 AS d FROM {self.database}.python_measurements_v1 WHERE run_id={{id:UUID}} GROUP BY ball_id HAVING count()>={{minimum:UInt64}})", {"id": run_id, "minimum": summary["min_track_observations"]})[0]
        summary.update(track_diameter_median_px=pixel_aggregate["median"], track_diameter_std_px=pixel_aggregate["std"])
        frame_stats = self.select(f"SELECT count() AS sampled, countIf(objects>0) AS detected FROM {self.database}.python_frames_v1 WHERE run_id={{id:UUID}}", params)[0]
        sampled, detected = int(frame_stats["sampled"]), int(frame_stats["detected"])
        summary.update(sampled_frames=sampled, frames_with_detections=detected, detection_ratio=detected / sampled if sampled else 0)
        summary.update(storage="clickhouse", run_id=run_id, measurements=len(measurements), tracks=len(tracks), eligible_tracks=len(eligible), track_diameter_median_mm=aggregate["median"], track_diameter_std_mm=aggregate["std"])
        if summary.get("expected_ball_diameter_mm", 0) > 0:
            summary["bias_to_expected_mm"] = sum(r["diameter_mm_median"] for r in eligible) / len(eligible) - summary["expected_ball_diameter_mm"] if eligible else None
        return measurements, frames, tracks, summary, provenance

    def export(self, run_id, output_dir):
        from pathlib import Path
        from .series_analysis import export_series_tables, write_series_plots
        measurements, frames, tracks, summary, provenance = self.load(run_id)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        export_series_tables(out, measurements, frames, tracks, summary, provenance)
        write_series_plots(out, measurements, [r for r in tracks if int(r["observations"]) >= summary["min_track_observations"]])
        return summary
