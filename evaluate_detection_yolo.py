from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.utils.ops import xywh2xyxy

# def box_iou(box1, box2):
#     """Calculate IoU between two sets of boxes."""
#     return bbox_iou(box1, box2, xywh=False)  # xywh=False means boxes are in xyxy format

def load_labels(path):
    """Load YOLO format labels [cls, x, y, w, h]."""
    if not path.exists():
        return torch.empty((0, 5))
    data = np.loadtxt(path, ndmin=2)
    return torch.tensor(data[:, :5], dtype=torch.float32)  # Ensure we only take the first 5 columns


def evaluate(gt_dir, pred_dir, iou_thres=0.5):
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    metrics = DetMetrics(names={0:"character"})
    metrics.box.nc = 1  # Number of classes, set to 1 for character detection

    gt_filemap = {f.stem[:f.stem.find("_jpg")]: f for f in gt_dir.glob("*.txt")}

    iouv = torch.linspace(0.5, 0.95, 10)
    niou = len(iouv)


    for file_dir in tqdm(pred_dir.glob("**/*.txt")):
        name = file_dir.stem
        gt_path = gt_filemap.get(name, None)
        if gt_path is None:
            print(f"Warning: No ground truth found for {name}, skipping.")
            continue
        pred_path = file_dir

        # Load ground truth and prediction
        gt = load_labels(gt_path)  # [cls, x, y, w, h]
        pred = load_labels(pred_path)  # same format: [cls, x, y, w, h]

        if len(gt):
            gt_boxes = xywh2xyxy(gt[:, 1:])
            gt_cls = gt[:, 0].int()
        else:
            gt_boxes = torch.empty((0, 4))
            gt_cls = torch.empty((0,), dtype=torch.int)

        if len(pred):
            pred_boxes = xywh2xyxy(pred[:, 1:])
            pred_cls = pred[:, 0].int()
            conf = torch.ones(len(pred))  # assume confidence = 1.0
        else:
            pred_boxes = torch.empty((0, 4))
            pred_cls = torch.empty((0,), dtype=torch.int)
            conf = torch.empty((0,))

        npr = len(pred_boxes)
        ntg = len(gt_boxes)
        tp = torch.zeros((npr, niou))

        if npr and ntg:
            ious = box_iou(pred_boxes, gt_boxes)  # [npr, ntg]
            iou_max, iou_argmax = ious.max(1)
            matches = []

            for i in range(npr):
                gt_idx = iou_argmax[i].item()
                if pred_cls[i] == gt_cls[gt_idx] and gt_idx not in matches:
                    iou_vals = ious[i, gt_idx]
                    tp[i] = iou_vals >= iouv
                    matches.append(gt_idx)

        # Feed into DetMetrics
        metrics.process(tp, conf, pred_cls, gt_cls)

    # Output results
    results = metrics.results_dict
    return results


# Example usage
if __name__ == "__main__":
    # parse args
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate YOLO detection results.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth labels.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing predicted labels.")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IoU threshold for matching.")
    args = parser.parse_args()
    # Run evaluation
    results = evaluate(args.gt_dir, args.pred_dir, args.iou_thres)
    print(results)
    # Example command to run the script
    # python evaluate_detection_yolo.py --pred_dir nom-detection-gt/nomnaocr-val --gt_dir datasets/nomnaocr/test/labels
