import sys,unittest
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from mvb.cv_geometry import refine_circle_radial
from mvb.series_analysis import CircleTracker

class RadialTests(unittest.TestCase):
    def test_rejects_long_open_arc_despite_high_total_coverage(self):
        image=np.full((260,260),30,np.uint8)
        cv2.ellipse(image,(130,130),(45,45),0,0,250,220,3)
        fit=refine_circle_radial(image,(130,130,47),6,'falling')
        self.assertIsNone(fit.circle)

    def test_accepts_closed_boundary_with_spatially_varying_contrast(self):
        image=np.full((260,260),120,np.uint8)
        yy,xx=np.indices(image.shape)
        inside=(xx-130)**2+(yy-130)**2<=45**2
        image[inside&(xx>=130)]=230
        image[inside&(xx<130)]=20
        fit=refine_circle_radial(image,(130,130,45),6,'either')
        self.assertEqual(fit.status,'accepted')
        self.assertAlmostEqual(fit.circle[2],45,delta=.5)

    def test_closed_ring_outer_boundary_not_mixed_with_inner_edge(self):
        image=np.full((260,260),30,np.uint8)
        cv2.circle(image,(130,130),45,220,5)
        fit=refine_circle_radial(image,(130,130,48),8,'falling')
        self.assertEqual(fit.status,'accepted')
        self.assertAlmostEqual(fit.circle[2],48,delta=1)

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
            for i in range(40):
                f=np.full((240,320,3),180,np.uint8)
                cv2.circle(f,(100+i,120),24,(30,30,30),-1);writer.write(f)
            writer.release();store=ClickHouseStore()
            cfg=DetectionConfig(detector_profile='dark-ball',min_ball_diameter_px=36,max_ball_diameter_px=60,min_ball_edge_support=.35,edge_fit_method='radial')
            result=analyze_series(VideoAnalysisConfig(str(video),str(root/'out'),frame_step=1),cfg,'pixel',cell_mask=False,storage=store)
            self.assertEqual(result['frames_with_detections'],40)
            diagnostics=store.load_quality(result['run_id'])
            self.assertEqual(len(diagnostics),40)
            self.assertTrue(all(r['status']=='accepted' for r in diagnostics))
            store.export(result['run_id'],root/'export')
            self.assertEqual((root/'out/quality.json').read_bytes(),(root/'export/quality.json').read_bytes())
            self.assertEqual((root/'out/measurements.csv').read_bytes(),(root/'export/measurements.csv').read_bytes())
            from mvb.metrology import export_bearing_check
            check=export_bearing_check(store,result['run_id'],root/'bearing',3.98)
            self.assertEqual(check['accepted_observations'],40)
            self.assertIsNone(check['full_physical_uncertainty_mm'])
            self.assertTrue((root/'bearing/bearing_check.csv').exists())
