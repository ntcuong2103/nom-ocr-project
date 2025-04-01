from ultralytics import YOLO
import os
import glob
project = "nom-detection"

# model = YOLO("models/yolo10m-1280-tkh.pt")
# name = "yolo10m-1280-tkh-test"

model = YOLO("models/yolo10m-1280-tkh-mth.pt")
name = "yolo10m-1280-tkh-mth-test"

# model = YOLO("models/yolo11n-1280-tkh.pt")
# name = "yolo11n-1280-tkh-test"


def generate_gt(model_checkpoint, path, output_folder):

    model = YOLO(model_checkpoint)
    model.predict(
        path,
        max_det=500,
        iou=0.0,
        conf=0.1,
        save_txt=True,
        save_conf=True,
        batch=4,
        project=output_folder,
        name=os.path.splitext(os.path.basename(model_checkpoint))[0],
        imgsz=1280,
        single_cls=True,
        verbose=True,
        agnostic_nms=True,
    )  # augment=True need to train with augment


def generate_pred(model, path="datasets/nomnaocr/val"):
    model.predict(
        path,
        max_det=500,
        iou=0.01,
        conf=0.1,
        save=True,
        show_labels=False,
        show_conf=False,
        show_boxes=True,
        batch=32,
        project=project,
        name=name,
        imgsz=1280,
        #   single_cls=True,
        # agnostic_nms = True, # comment out
        verbose=True,
        line_width=1,
    )


if __name__ == "__main__":
    # parse args
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/yolo10m-1280-tkh-mth.pt")
    parser.add_argument("--path", type=str, default="datasets/nomnaocr/Pages/**/*.jpg")
    parser.add_argument("--output", type=str, default="nom-detection-gt")
    args = parser.parse_args()

    generate_gt(args.model, args.path, args.output)
