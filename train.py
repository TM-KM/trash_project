from ultralytics import YOLO

# 새로 학습한 다중 클래스 쓰레기 모델 로드
model = YOLO("trash_best.pt")

# 추론 실행
results = model.predict(
    source=r"C:\Users\LG\Desktop\test2.png",
    conf=0.20,
    imgsz=1280,
    save=True,
    show=True
)