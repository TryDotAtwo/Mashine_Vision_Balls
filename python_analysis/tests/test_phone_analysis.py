import csv,json,os,sys,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from mvb.pipeline import DetectionConfig,VideoAnalysisConfig
from mvb.series_analysis import detect_dark_circles,analyze_series,calibration_from_csv,load_calibration
from mvb.clickhouse_store import ClickHouseStore

class PhoneTests(unittest.TestCase):
    def test_dark_silhouette_rejects_bubble_and_clipped_circle(self):
        cfg=DetectionConfig(detector_profile="dark-ball",min_ball_edge_support=.35)
        image=np.full((240,320),180,np.uint8)
        cv2.circle(image,(90,120),24,35,-1)
        cv2.circle(image,(85,114),6,230,-1)
        cv2.circle(image,(220,120),24,35,2)
        found=detect_dark_circles(cv2.GaussianBlur(image,(5,5),1.5),18,30,None,cfg)
        self.assertEqual(len(found),1)
        self.assertLess(np.linalg.norm(np.array(found[0][0][:2])-[90,120]),4)
        cropped=np.full((240,320),180,np.uint8)
        cv2.circle(cropped,(5,120),24,35,-1)
        self.assertEqual(detect_dark_circles(cropped,18,30,None,cfg),[])

    def test_pixel_exports_and_provisional_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'sample.avi'
            w=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),20,(320,240))
            for i in range(12):
                f=np.full((240,320,3),180,np.uint8);cv2.circle(f,(100+i,120),24,(35,35,35),-1);w.write(f)
            w.release()
            cfg=DetectionConfig(detector_profile='dark-ball',min_ball_diameter_px=36,max_ball_diameter_px=60,min_ball_edge_support=.35)
            summary=analyze_series(VideoAnalysisConfig(str(video),str(root/'out'),frame_step=1),cfg,'pixel',cell_mask=False,annotated_video=True)
            self.assertEqual(summary['sampled_frames'],12)
            self.assertEqual(summary['decode_status'],'complete')
            self.assertEqual(summary['frames_with_detections'],12)
            with (root/'out/measurements.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertTrue(all(r['diameter_mm']=='' and r['scale_mm_per_px']=='' for r in rows))
            self.assertEqual(len({r['ball_id'] for r in rows}),1)
            cal=calibration_from_csv(root/'out/measurements.csv',3.98,None,root/'cal.json',provisional=True)
            self.assertEqual(load_calibration(root/'cal.json')['status'],'provisional')
            self.assertIsNone(cal['sigma_scale_mm_per_px'])
            summary=analyze_series(VideoAnalysisConfig(str(video),str(root/'metric'),frame_step=1),cfg,'pixel',cell_mask=False,calibration_path=root/'cal.json')
            self.assertEqual(summary['calibration_status'],'provisional')
            with (root/'metric/measurements.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertTrue(all(r['sigma_model_mm']=='' and r['method']=='provisional_calibration' for r in rows))
            with self.assertRaises(ValueError):
                calibration_from_csv(root/'out/measurements.csv',3.98,None,root/'invalid.json')

    def test_decoder_short_read_is_not_complete(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'short.avi'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),20,(320,240))
            for _ in range(3):writer.write(np.full((240,320,3),180,np.uint8))
            writer.release()
            capture=cv2.VideoCapture(str(video))
            class IncompleteCapture:
                def isOpened(self):return capture.isOpened()
                def get(self,key):return 6 if key==cv2.CAP_PROP_FRAME_COUNT else capture.get(key)
                def read(self):return capture.read()
                def release(self):capture.release()
            with patch('mvb.series_analysis.cv2.VideoCapture',return_value=IncompleteCapture()):
                summary=analyze_series(VideoAnalysisConfig(str(video),str(root/'incomplete'),frame_step=1),DetectionConfig(),'pixel',cell_mask=False)
            self.assertEqual(summary['decode_status'],'incomplete')
            self.assertEqual(summary['decoded_frames'],3)
            self.assertEqual(summary['reported_frames'],6)
            summary=analyze_series(VideoAnalysisConfig(str(video),str(root/'limited'),frame_step=1,max_frames=2),DetectionConfig(),'pixel',cell_mask=False)
            self.assertEqual(summary['decode_status'],'limited_by_request')

@unittest.skipUnless(os.environ.get('MVB_CLICKHOUSE_TEST')=='1','local ClickHouse required')
class PixelDatabaseTests(unittest.TestCase):
    def test_sql_preserves_unknown_scale_and_reexport(self):
        from mvb.pipeline import FrameMeasurement
        store=ClickHouseStore()
        rows=[FrameMeasurement(i,i/10,100,120,24,48+i,None,None,None,None,None,'pixel',.8,1,0) for i in range(4)]
        frames=[dict(frame_index=i,time_s=i/10,objects=1,scale_available=False) for i in range(4)]
        run=store.save(rows,frames,{'video_sha256':'pixel_test'},{'min_track_observations':2,'units':'px'})
        read,_,tracks,summary,_=store.load(run)
        self.assertTrue(all(r.diameter_mm is None for r in read))
        self.assertEqual(tracks[0]['diameter_px_median'],49.5)
        self.assertIsNone(summary['track_diameter_median_mm'])
        with tempfile.TemporaryDirectory() as tmp:
            store.export(run,tmp)
            with (Path(tmp)/'measurements.csv').open() as stream:
                exported=list(csv.DictReader(stream))
            self.assertEqual(len(exported),4)
            self.assertTrue(all(r['diameter_mm']=='' for r in exported))
