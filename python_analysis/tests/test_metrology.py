import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mvb.metrology import bearing_check,ratio_standard_uncertainty


class BearingMetrologyTests(unittest.TestCase):
    def test_common_image_uncertainty_cancels_in_same_object_ratio(self):
        shared=[[.0001,0,0],[0,1,1],[0,1,1]]
        independent=[[.0001,0,0],[0,1,0],[0,0,1]]
        self.assertAlmostEqual(ratio_standard_uncertainty(4,50,50,shared),.01)
        self.assertGreater(ratio_standard_uncertainty(4,50,50,independent),.1)
        with self.assertRaises(ValueError):ratio_standard_uncertainty(4,50,50,None)
        with self.assertRaises(ValueError):ratio_standard_uncertainty(4,50,50,[[1,2,0],[2,1,0],[0,0,1]])

    def rows(self,values):
        return [{'frame_index':i,'ball_id':0,'ball_diameter_px':float(v)}
                for i,v in enumerate(values) if np.isfinite(v)]

    def test_scale_uses_only_first_half_and_reports_unknown_full_uncertainty(self):
        result,points=bearing_check(self.rows([50]*30+[52]*30),60,4,bootstrap_samples=100)
        self.assertAlmostEqual(result['scale_mm_per_video_px'],.08)
        self.assertAlmostEqual(result['conditional_bias_mm'],.16)
        self.assertAlmostEqual(result['conditional_rmse_mm'],.16)
        self.assertIsNone(result['full_physical_uncertainty_mm'])
        self.assertIsNone(result['reference_standard_uncertainty_mm'])
        self.assertGreater(result['quantization_single_chord_u_mm'],0)
        self.assertEqual(points[30]['role'],'check')

    def test_missing_frames_and_seed_preserved(self):
        v=np.r_[np.linspace(48,52,30),np.linspace(49,53,30)]
        v[[2,8,34,47]]=np.nan
        a,p=bearing_check(self.rows(v),60,4,bootstrap_samples=100)
        b,q=bearing_check(self.rows(v),60,4,bootstrap_samples=100)
        self.assertEqual(a,b)
        self.assertEqual(a['calibration_observations'],28)
        self.assertEqual(a['check_observations'],28)
        self.assertEqual(p[2]['frame_index'],3)

    def test_mixed_tracks_and_duplicate_frames_rejected(self):
        rows=self.rows([50]*60);rows[30]['ball_id']=1
        with self.assertRaises(ValueError):bearing_check(rows,60,4)
        rows=self.rows([50]*60);rows.append(rows[0])
        with self.assertRaises(ValueError):bearing_check(rows,60,4)
