from ultralytics import YOLO
project = "nom-detection"
name = "yolo10m-1280p"
import wandb
wandb.init(project=project, name=name)
from wandb.integration.ultralytics import add_wandb_callback



# Load a model
# model = YOLO("yolov10m.pt")  # load a pretrained model (recommended for training)
model = YOLO("nom-detection/yolo10m-12803/weights/best.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

add_wandb_callback(model, enable_model_checkpointing=True)

# https://docs.ultralytics.com/modes/train/#train-settings
# Train the model
model.train(data="datasets/tkh-mth2k2/tkh-mth2k2.yaml",
            epochs=100,
            imgsz=1280,
            project=project,
            name=name,
            single_cls=True,
            mosaic=0,
            save=True,
            patience=5,
            resume=False,
            device=[0,1,2],
            batch=60,
            cache=True,
            plots=False,
            crop_fraction=0.5,
            pretrained=True)
wandb.finish()