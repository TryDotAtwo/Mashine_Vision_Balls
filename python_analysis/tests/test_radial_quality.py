import sys,unittest
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from mvb.cv_geometry import refine_circle_radial
from mvb.series_analysis import CircleTracker

class RadialTests(unittest.TestCase):
    def test_polarity_noise_and_distractors(self):
        for bright in (False,True):
            image=np.full((240,320),30 if bright else 180,np.uint8)
            cv2.circle(image,(150,120),30,230 if bright else 30,-1)
            cv2.line(image,(130,70),(130,170),100,2)
            image=np.clip(image.astype(float)+np.random.default_rng(17).normal(0,6,image.shape),0,255).astype('uint8')
            fit=refine_circle_radial(image,(151,119,30),8,'falling' if bright else 'rising')
            self.assertEqual(fit.status,'accepted')
            self.assertLess(np.linalg.norm(np.array(fit.circle[:2])-[150,120]),.5)
            self.assertAlmostEqual(fit.circle[2],30,delta=.5)
    def test_rejects_blank_clipped_and_ellipse(self):
        image=np.full((240,320),180,np.uint8)
        self.assertIsNone(refine_circle_radial(image,(150,120,30),8).circle)
        self.assertEqual(refine_circle_radial(image,(5,120,30),8).status,'clipped_contour')
        cv2.ellipse(image,(150,120),(32,24),0,0,360,30,-1)
        self.assertIsNone(refine_circle_radial(image,(150,120,30),8,'rising').circle)
    def test_irregular_timestamps_and_gap(self):
        tracker=CircleTracker(6,3)
        self.assertEqual(tracker.update([(50,50,10)],0),[0])
        self.assertEqual(tracker.update([(54,50,10)],.2),[0])
        self.assertEqual(tracker.update([], .6),[])
        self.assertEqual(tracker.update([(70,50,10)],1),[0])
        with self.assertRaises(ValueError):tracker.update([],1)


import os,csv,json,tempfile
from mvb.clickhouse_store import ClickHouseStore
from mvb.pipeline import DetectionConfig,VideoAnalysisConfig
from mvb.series_analysis import analyze_series

@unittest.skipUnless(os.environ.get('MVB_CLICKHOUSE_TEST')=='1','local ClickHouse required')
class RadialDatabaseTests(unittest.TestCase):
    def test_radial_video_diagnostics_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'radial.avi'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),20,(320,240))
            for i in range(8):
                f=np.full((240,320,3),180,np.uint8)
                cv2.circle(f,(100+i,120),24,(30,30,30),-1);writer.write(f)
            writer.release();store=ClickHouseStore()
            cfg=DetectionConfig(detector_profile='dark-ball',min_ball_diameter_px=36,max_ball_diameter_px=60,min_ball_edge_support=.35,edge_fit_method='radial')
            result=analyze_series(VideoAnalysisConfig(str(video),str(root/'out'),frame_step=1),cfg,'pixel',cell_mask=False,storage=store)
            self.assertEqual(result['frames_with_detections'],8)
            diagnostics=store.load_quality(result['run_id'])
            self.assertEqual(len(diagnostics),8)
            self.assertTrue(all(r['status']=='accepted' for r in diagnostics))
            store.export(result['run_id'],root/'export')
            self.assertEqual((root/'out/quality.json').read_bytes(),(root/'export/quality.json').read_bytes())
            self.assertEqual((root/'out/measurements.csv').read_bytes(),(root/'export/measurements.csv').read_bytes())
