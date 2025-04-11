import numpy as np
import imagesize
import os
import glob

def convert_chinese_to_yolo_format(label_path, img_path, output_path, single_class=False):
    labels, x1, y1, x2, y2 = list(zip(*[line.split() for line in open(label_path, 'r').readlines()]))
    x1 = np.array(x1, dtype=float)
    y1 = np.array(y1, dtype=float)
    x2 = np.array(x2, dtype=float)
    y2 = np.array(y2, dtype=float)

    img_w, img_h = imagesize.get(img_path)
    x = (x1 + x2) / 2 / img_w
    y = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    with open(output_path, 'w') as f:
        for i in range(len(labels)):
            if single_class:
                f.write(f'0 {x[i]:6f} {y[i]:6f} {w[i]:6f} {h[i]:6f}\n')
            else:
                f.write(f'{labels[i]} {x[i]:6f} {y[i]:6f} {w[i]:6f} {h[i]:6f}\n')

def convert_yolo_single_class(label_path, output_path):
    labels, x, y, w, h = list(zip(*[line.split() for line in open(label_path, 'r').readlines()]))

    with open(output_path, 'w') as f:
        for i in range(len(labels)):
            f.write(f'0 {x[i]} {y[i]} {w[i]} {h[i]}\n')


# find all image files *.png or *.jpg
data_dir = 'datasets/tkh-mth2k2/MTH1000/images'

for img_path in glob.glob(f'{data_dir}/*.png') + glob.glob(f'{data_dir}/*.jpg'):
    label_path = img_path.replace('images', 'label_char').replace('.png', '.txt').replace('.jpg', '.txt')
    output_path = img_path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    convert_chinese_to_yolo_format(label_path, img_path, output_path)

# label_dir = "datasets/nom-nakagawa-lab/labels"
# output_dir = "datasets/nom-nakagawa-lab/labels_single"
# os.makedirs(output_dir, exist_ok=True)

# for label_path in glob.glob(f'{label_dir}/*.txt'):
#     output_path = label_path.replace(label_dir, output_dir)
#     convert_yolo_single_class(label_path, output_path)    
