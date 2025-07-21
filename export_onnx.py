from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='onnx', dynamic=True, imgsz=320)
print('המודל יוצא ל-best.onnx בגודל מוקטן (imgsz=320) בהצלחה!') 