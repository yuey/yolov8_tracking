from ultralytics import YOLO

# Load a model
model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/home/ubuntu/cs231n/data/yolo_base/data.yaml", epochs=10, patience=3, imgsz=(1920,1080), name='yolov8l_trial_1_ST', optimizer='Adam', lr0=1e-3, )  # train the model
metrics = model.val()  # evaluate model performance on the validation set
success = model.export(format='onnx', device=0)  # export the model to ONNX format