# generate detection results
python test.py --path nomnaocr-val --model best.pt --output nomnaocr-val-detection
# evaluate detection results
python evaluate_detection_yolo.py --pred_dir nomnaocr-val-detection --gt_dir nomnaocr-val-roboflow/labels
