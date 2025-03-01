import torch.backends.mps
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import cv2
import numpy as np

class DeepShipDetector:
    def __init__(self, path, device):
        self.device = device
        self.path = path
        self.model = self.load_model()
        self.names = self.model.names
        self.tracker = DeepSort(max_iou_distance=0.5, max_age=40)


    def load_model(self):
        model = YOLO("weights/ship_detector.pt")
        model.to(self.device)
        model.fuse()
        return model

    def results(self, frame):
        results = self.model(frame)[0]
        return results

    def get_frame(self, frame, results):
        res_arr = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > 0.5:
                res_arr.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], float(score), int(class_id)))

            tracks = self.tracker.update_tracks(raw_detections=res_arr, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue

                bbox = track.to_tlbr()
                x1, y1, x2, y2 = bbox
                idx = track.track_id
                class_id = track.get_det_class()

                text = f"{idx}:{self.names[int(class_id)]}"

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, text, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.results(frame)
            frame = self.get_frame(frame, results)

            cv2.imshow('Deep Sort Tracker', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

path = "civil_ship.mp4"
device = "mps" if torch.backends.mps.is_available() else "cpu"
tracker = DeepShipDetector(path, device)
tracker()