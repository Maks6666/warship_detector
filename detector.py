import cv2
from torch.xpu import device
from ultralytics import YOLO
import torch
from sort import Sort
import numpy as np

class ShipDetecor:
    def __init__(self, device, path):
        self.device = device
        self.model = self.load_model()
        self.names = self.model.names
        self.path = path
        self.sort = Sort(max_age=100, min_hits=8, iou_threshold=0.4)


    def load_model(self):
        model = YOLO("weights/ship_detector.pt")
        model.fuse()
        model.to(self.device)
        return model

    def results(self, model, frame):
        results = model.predict(frame, conf=0.4)
        return results

    def get_results(self, results):
        arr = []
        for result in results[0]:
            bbox = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()

            t_arr = [bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], conf[0], cls[0]]
            arr.append(t_arr)

        return np.array(arr)

    def draw_boxes(self, frame, bboxes, indicies, classes):
        for bbox, index, clas in zip(bboxes, indicies, classes):
            name = self.names[int(clas)]

            color = (0, 0, 0)

            if clas == 1:
                color = (0, 0, 255)
            elif clas == 0:
                color = (0, 255, 0)

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame, name, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame



    def __call__(self):


        cap = cv2.VideoCapture(self.path)

        assert cap.isOpened()
        while True:
            ret, frame = cap.read()
            if not ret:
                break


            results = self.results(self.model, frame)
            arr = self.get_results(results)

            if len(arr) == 0:
                arr = np.empty((0, 5))

            res = self.sort.update(arr)

            bboxes = res[:, :-1]
            indicies = res[:, -1].astype(int)
            classes = arr[:, -1].astype(int)

            frame = self.draw_boxes(frame, bboxes, indicies, classes)

            cv2.imshow('Framw', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


device = "mps" if torch.backends.mps.is_available() else "cpu"
path = "civil_ship.mp4"
ship_detector = ShipDetecor(device, path)
ship_detector()
