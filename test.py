from ultralytics import YOLO

project = "nom-detection"

# model = YOLO("nom-detection/yolo10m-12806/weights/best.pt") 
# name = "yolo10m-1280-tkh-test"

model = YOLO("runs/detect/train/weights/best.pt")
name = "yolo10m-1280-tkh-mth-test"

# model = YOLO("nom-detection/yolo11n-12806/weights/best.pt")
# name = "yolo11n-1280-tkh-test"

def generate_gt(model, path = "nom-ocr/nomnaocr/val"):
    model.predict(path,
                  max_det = 500,
                  iou=0.0,
                  conf=0.1,
                  save_txt=True,
                  save_conf=True,
                  batch=32,
                  project=project,
                  name=name,
                  imgsz=1280,
                  single_cls=True,
                  verbose=True,
                  agnostic_nms = True) # augment=True need to train with augment

def generate_pred(model, path = "nom-ocr/nomnaocr/val"):
    model.predict(path,
                  max_det = 500,
                  iou=0.01,
                  conf=0.1,
                  save= True,
                  show_labels=False,
                  show_conf=False,
                  show_boxes=True,
                  batch=32,
                  project=project,
                  name=name,
                  imgsz=1280,
                #   single_cls=True,
                  agnostic_nms = True,
                  verbose=True,
                  line_width=1,
                  )

generate_pred(model)
    