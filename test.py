import glob
from ultralytics import YOLO
import os
from ultralytics.models.yolo.detect import DetectionPredictor
from tqdm import tqdm

def generate_gt(model_checkpoint, path, output_folder):
    model = YOLO(model_checkpoint)

    for file in tqdm(glob.glob(path + "/**/*.jpg", recursive=True)):
        results = model(file, max_det=500, iou=0.1, conf=0.1, imgsz=1280, single_cls=True, agnostic_nms=True, save=False, show_labels=False, show_conf=False, show_boxes=True)
        id = os.path.relpath(file, path)[:-4]
        for result in results:
            result.save_txt(f"{output_folder}/{id}.txt", save_conf=False)


    # model.predict(
    #     path,
    #     max_det=500,
    #     iou=0.1,
    #     conf=0.1,
    #     save_txt=True,
    #     save_conf=True,
    #     batch=32,
    #     project=output_folder,
    #     name=os.path.splitext(os.path.basename(model_checkpoint))[0],
    #     imgsz=1280,
    #     single_cls=True,
    #     verbose=True,
    #     agnostic_nms=True,
    # )  # augment=True need to train with augment



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
    parser.add_argument("--path", type=str, default="datasets/nomnaocr/val/images")
    parser.add_argument("--model", type=str, default="models/yolo11n-1280-nom-data/weights/best.pt")
    parser.add_argument("--output", type=str, default="nom-detection-gt/nomnaocr-val")
    args = parser.parse_args()

    generate_gt(args.model, args.path, args.output)
