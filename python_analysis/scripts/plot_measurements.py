"""
Строит графики по measurements.csv: время–диаметр, гистограмма, скользящая sigma,
при наличии колонки reference_diameter_px — прокси стабильности масштаба/уровня.

Запуск (из корня репозитория):
  py -3 scripts/plot_measurements.py --csv output/analysis/<stem>/measurements.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from mvb.ball_episodes import episode_spans_from_ball_ids, segment_episodes_by_time_gaps


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    return fields, rows


def _f(rows: list[dict[str, str]], key: str) -> np.ndarray:
    out = []
    for r in rows:
        v = r.get(key, "").strip()
        if v == "":
            out.append(np.nan)
        else:
            out.append(float(v))
    return np.array(out, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Графики стабильности и распределения по measurements.csv")
    parser.add_argument("--csv", type=Path, required=True, help="Путь к measurements.csv")
    parser.add_argument("--out-dir", type=Path, default=None, help="Папка для PNG (по умолчанию рядом с CSV)")
    parser.add_argument("--gap-s", type=float, default=0.8, help="Порог разрыва времени для нового шарика, с")
    parser.add_argument("--rolling", type=int, default=7, help="Окно скользящего std (кадров), нечётное предпочтительно")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Нужен пакет matplotlib: pip install matplotlib") from e

    csv_path = args.csv.resolve()
    out_dir = args.out_dir or csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    _, rows = _read_csv(csv_path)
    if not rows:
        raise SystemExit("CSV пуст")

    t = _f(rows, "time_s")
    d_mm = _f(rows, "diameter_mm")
    valid = np.isfinite(t) & np.isfinite(d_mm)
    t = t[valid]
    d_mm = d_mm[valid]

    has_ball_id = "ball_id" in rows[0] and any(str(r.get("ball_id", "")).strip() for r in rows)
    if has_ball_id:
        bid = _f(rows, "ball_id").astype(int)
        bid = bid[valid]
        spans = episode_spans_from_ball_ids(list(t), list(bid))
        episode_mode = "ball_id из CSV"
    else:
        spans = segment_episodes_by_time_gaps(list(t), gap_threshold_s=args.gap_s)
        episode_mode = f"gap={args.gap_s} с"

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax0 = axes[0, 0]
    n_ep = len(spans)
    for sp in spans:
        sl = slice(sp.start_row, sp.end_row + 1)
        ax0.plot(
            t[sl],
            d_mm[sl],
            color=f"C{sp.ball_id % 10}",
            lw=1.2,
            label=f"шарик {sp.ball_id}" if n_ep <= 10 else None,
        )
    ax0.set_xlabel("t, с")
    ax0.set_ylabel("D, мм")
    ax0.set_title(f"Диаметр во времени ({episode_mode})")
    if n_ep <= 10 and n_ep > 1:
        ax0.legend(fontsize=7, loc="best")

    ax1 = axes[0, 1]
    ax1.hist(d_mm, bins=min(40, max(10, len(d_mm) // 5)), color="steelblue", edgecolor="white", alpha=0.85)
    ax1.set_xlabel("D, мм")
    ax1.set_ylabel("число кадров")
    ax1.set_title("Гистограмма диаметра (вся серия)")

    win = max(3, args.rolling | 1)
    rolling_std = np.full_like(d_mm, np.nan)
    for i in range(len(d_mm)):
        a = max(0, i - win // 2)
        b = min(len(d_mm), i + win // 2 + 1)
        seg = d_mm[a:b]
        rolling_std[i] = float(np.std(seg, ddof=1)) if len(seg) > 1 else 0.0

    ax2 = axes[1, 0]
    ax2.plot(t, rolling_std, color="darkgreen", lw=1.0)
    ax2.set_xlabel("t, с")
    ax2.set_ylabel(f"rolling std(D), окно={win}")
    ax2.set_title("Стабильность размера по времени (ниже — стабильнее)")

    ax3 = axes[1, 1]
    ref_key = "reference_diameter_px"
    if rows and ref_key in rows[0] and any(r.get(ref_key, "").strip() for r in rows):
        ref_all = _f(rows, ref_key)
        ref_px = ref_all[valid]
        ax3.plot(t, ref_px, color="orange", lw=1.0)
        ax3.set_xlabel("t, с")
        ax3.set_ylabel("reference_diameter_px")
        ax3.set_title("Прокси масштаба: диаметр референса в px (дрейф → уровень/расстояние)")
        rstd = float(np.std(ref_px[np.isfinite(ref_px)], ddof=1)) if np.sum(np.isfinite(ref_px)) > 1 else 0.0
        ax3.text(
            0.02,
            0.98,
            f"std(ref_px)={rstd:.3f} px",
            transform=ax3.transAxes,
            va="top",
            fontsize=9,
        )
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "Нет колонки reference_diameter_px\n(reference mode)", ha="center", va="center")

    plt.tight_layout()
    out_png = out_dir / "plots_stability.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    summary_path = out_dir / "plots_stability_summary.txt"
    lines = [
        f"csv={csv_path}",
        f"episode_mode={episode_mode}",
        f"gap_threshold_s={args.gap_s}",
        f"episodes={len(spans)}",
        f"overall: mean_D={float(np.mean(d_mm)):.6f} mm std={float(np.std(d_mm, ddof=1)):.6f} mm n={len(d_mm)}",
    ]
    for sp in spans:
        sl = slice(sp.start_row, sp.end_row + 1)
        seg = d_mm[sl]
        lines.append(
            f"  ball_id={sp.ball_id}: mean={float(np.mean(seg)):.6f} std={float(np.std(seg, ddof=1)):.6f} "
            f"n={len(seg)} t=[{sp.time_start_s:.3f},{sp.time_end_s:.3f}]"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_png}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
