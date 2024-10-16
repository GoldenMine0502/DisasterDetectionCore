import yolov5

# load model
# model = yolov5.load('keremberke/yolov5m-license-plate')
model = yolov5.load('fcakyon/yolov5s-v7.0')

# set model parameters
model.conf = 0.2  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# set image
img = 'images/main_1.jpg'

# perform inference
# results = model(img, size=640)

# inference with test time augmentation
results = model(img, augment=True)

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4]  # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

print(len(predictions))
print(predictions)

# show detection bounding boxes on image
# results.show()

# save results into "results/" folder
results.save(save_dir='results/', labels=False)