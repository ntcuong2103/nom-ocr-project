from ultralytics import YOLO

import wandb
wandb.init(project="nom-detection", name="yolo11n-640")
from wandb.integration.ultralytics import add_wandb_callback



# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

add_wandb_callback(model, enable_model_checkpointing=True)

# Train the model
model.train(data="datasets/tkh-mth2k2/tkh-mkh2k2.yaml", epochs=10, project="nom-detection", name="yolo11n-640", imgsz=640, single_cls=True, mosaic=0, save=True)  
wandb.finish()