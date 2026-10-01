"""Окно matplotlib с живыми графиками для run_live_preview."""

from __future__ import annotations

from typing import Literal

import numpy as np

MeasurementMode = Literal["fov", "reference"]


class LiveMetricsPlot:
    def __init__(self, measurement_mode: MeasurementMode, max_points: int = 3000) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self.max_points = max(100, max_points)
        self._t: list[float] = []
        self._d: list[float] = []
        self._bid: list[int] = []
        self._sec: list[float] = []

        self.fig, (self.ax_d, self.ax_s) = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
        self.fig.canvas.manager.set_window_title("MVB live metrics")
        self._scat = self.ax_d.scatter(
            [],
            [],
            c=[],
            cmap="tab10",
            vmin=0,
            vmax=9,
            s=16,
            alpha=0.9,
        )
        self.ax_d.set_xlabel("t, s")
        self.ax_d.set_ylabel("D, mm")
        self.ax_d.set_title("Diameter (цвет = ball_id % 10)")

        self._line_s, = self.ax_s.plot([], [], color="darkorange", lw=1.1)
        self.ax_s.set_xlabel("t, s")
        if measurement_mode == "reference":
            self.ax_s.set_ylabel("reference_diameter_px")
            self.ax_s.set_title("Прокси масштаба (референс в px)")
        else:
            self.ax_s.set_ylabel("scale_mm_per_px")
            self.ax_s.set_title("Масштаб мм/px (дрейф FOV/дистанции)")

        plt.ion()
        self.fig.show()
        self._closed = False
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _on_close(self, _event: object) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def push(self, time_s: float, diameter_mm: float, ball_id: int, secondary: float) -> None:
        self._t.append(time_s)
        self._d.append(diameter_mm)
        self._bid.append(ball_id)
        self._sec.append(secondary)
        over = len(self._t) - self.max_points
        if over > 0:
            del self._t[:over]
            del self._d[:over]
            del self._bid[:over]
            del self._sec[:over]

    def refresh(self) -> None:
        if self._closed or not self._t:
            return
        t_arr = np.asarray(self._t, dtype=float)
        d_arr = np.asarray(self._d, dtype=float)
        bid_arr = np.asarray(self._bid, dtype=int)
        self._scat.set_offsets(np.column_stack([t_arr, d_arr]))
        self._scat.set_array(bid_arr % 10)
        self.ax_d.relim()
        self.ax_d.autoscale_view()
        self._line_s.set_data(t_arr, np.asarray(self._sec, dtype=float))
        self.ax_s.relim()
        self.ax_s.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._plt.pause(0.001)

    def close(self) -> None:
        if self._closed:
            return
        self._plt.close(self.fig)
