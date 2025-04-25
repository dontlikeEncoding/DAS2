from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v5/yolov5-all.yaml')
    model.train(data='ultralytics/cfg/datasets/data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                patience=200,
                batch=24,
                project='runs/train',
                name='yolov5-all',
                )