from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Read image
def detect_objects(img):
    """
    img: OpenCV image (numpy array)
    return: list of detected object names
    """
    results = model(img)

    detected_objects = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        detected_objects.append(label)

    return list(set(detected_objects))  # unique objects


# img = cv2.imread("image.png")
# objects = detect_objects(img)

# print("Detected Objects:", objects)