from ultralytics import YOLO

model = YOLO("models/best.pt")
print(model.names)

helmet_model = YOLO("models/helmet.pt")
print(helmet_model.names)
