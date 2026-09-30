from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvb.ball_episodes import (
    advance_ball_id,
    episode_spans_from_ball_ids,
    segment_episodes_by_time_gaps,
)


def test_single_episode():
    t = [0.0, 0.1, 0.2]
    spans = segment_episodes_by_time_gaps(t, gap_threshold_s=1.0)
    assert len(spans) == 1
    assert spans[0].ball_id == 0
    assert spans[0].start_row == 0
    assert spans[0].end_row == 2


def test_two_episodes_by_gap():
    t = [0.0, 0.1, 2.0, 2.1]
    spans = segment_episodes_by_time_gaps(t, gap_threshold_s=0.8)
    assert len(spans) == 2
    assert spans[0].end_row == 1
    assert spans[1].start_row == 2


def test_advance_ball_id():
    assert advance_ball_id(True, -1) == 0
    assert advance_ball_id(False, 0) == 0
    assert advance_ball_id(True, 0) == 1


def test_episode_spans_from_ball_ids():
    t = [0.0, 0.1, 0.2, 1.0, 1.1]
    bid = [0, 0, 0, 1, 1]
    spans = episode_spans_from_ball_ids(t, bid)
    assert len(spans) == 2
    assert spans[0].ball_id == 0
    assert spans[1].ball_id == 1
