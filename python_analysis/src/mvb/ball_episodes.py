"""Сегментация серии кадров на эпизоды «отдельный шарик» по разрывам во времени детекций."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeSpan:
    ball_id: int
    start_row: int
    end_row: int
    time_start_s: float
    time_end_s: float


def episode_spans_from_ball_ids(times_s: list[float], ball_ids: list[int]) -> list[EpisodeSpan]:
    """Непрерывные отрезки с одним ball_id (уже присвоенным пайплайном)."""
    if not times_s or len(times_s) != len(ball_ids):
        return []
    spans: list[EpisodeSpan] = []
    i = 0
    n = len(times_s)
    while i < n:
        j = i + 1
        while j < n and ball_ids[j] == ball_ids[i]:
            j += 1
        spans.append(
            EpisodeSpan(
                ball_id=int(ball_ids[i]),
                start_row=i,
                end_row=j - 1,
                time_start_s=float(times_s[i]),
                time_end_s=float(times_s[j - 1]),
            )
        )
        i = j
    return spans


def segment_episodes_by_time_gaps(
    times_s: list[float],
    gap_threshold_s: float,
) -> list[EpisodeSpan]:
    """
    times_s: монотонно неубывающая последовательность (как в measurements.csv).
    Новый эпизод начинается, если разрыв между соседними отметками > gap_threshold_s.
    """
    if not times_s:
        return []
    spans: list[EpisodeSpan] = []
    start = 0
    ball_id = 0
    for i in range(1, len(times_s)):
        if times_s[i] - times_s[i - 1] > gap_threshold_s:
            spans.append(
                EpisodeSpan(
                    ball_id=ball_id,
                    start_row=start,
                    end_row=i - 1,
                    time_start_s=times_s[start],
                    time_end_s=times_s[i - 1],
                )
            )
            ball_id += 1
            start = i
    spans.append(
        EpisodeSpan(
            ball_id=ball_id,
            start_row=start,
            end_row=len(times_s) - 1,
            time_start_s=times_s[start],
            time_end_s=times_s[-1],
        )
    )
    return spans


def advance_ball_id(track_was_empty: bool, current_ball_id: int) -> int:
    """
    Новый номер шарика: current_ball_id увеличивается, если в начале кадра трек пуст
    (старт записи или сброс после clear_track_after_lost_frames).
    """
    if track_was_empty:
        return current_ball_id + 1
    return current_ball_id


def row_to_ball_id(row_index: int, spans: list[EpisodeSpan]) -> int | None:
    for span in spans:
        if span.start_row <= row_index <= span.end_row:
            return span.ball_id
    return None
