# Step 1: Copy the dataset to folder "nomnaocr-val" using nomnaocr-val-list.txt

# Step 2: generate detection results
python test.py --path nomnaocr-val --model best.pt --output nomnaocr-val-detection
# Step 3: evaluate detection results
python evaluate_detection_yolo.py --pred_dir nomnaocr-val-detection --gt_dir nomnaocr-val-roboflow/labels
