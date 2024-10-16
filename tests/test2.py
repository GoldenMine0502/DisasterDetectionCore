from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")


# metrics = model.val()

# Perform object detection on an image
results = model("images/zidane.jpg")
# results[0].show()
results[0].save('results/zidane.jpg')


# [ultralytics/ultralytics/cfg/datasets/coco8.yaml at main · ultralytics/ultralytics](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml)
# [객체 감지 데이터 세트 개요 - Ultralytics YOLO 문서](https://docs.ultralytics.com/ko/datasets/detect/)