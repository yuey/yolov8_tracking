from ultralytics import YOLO
import os
import torch

torch.cuda.empty_cache()
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16"

# Load a model
model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/home/ubuntu/cs231n/data/yolo_base/data.yaml", epochs=10, patience=3, imgsz=640, name='yolov8l_trial_1_ST_', optimizer='Adam', lr0=1e-3, pretrained=True, batch=24)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
success = model.export(format='onnx', device=0)  # export the model to ONNX format