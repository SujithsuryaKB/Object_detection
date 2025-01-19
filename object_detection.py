import cv2
import numpy as np

# Load the pre-trained MobileNet-SSD model and configuration file
model_path = "MobileNetSSD_deploy.caffemodel"
config_path = "MobileNetSSD_deploy.prototxt"

# Class labels the MobileNet-SSD model is trained to detect
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# Load the model
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

def detect_objects(image_path):
    """
    Perform object detection on an image using MobileNet-SSD.

    Args:
        image_path (str): Path to the input image.

    Returns:
        None
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    (h, w) = image.shape[:2]

    # Prepare the image for object detection
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)

    # Set the input to the pre-trained model
    net.setInput(blob)

    # Perform forward pass to get predictions
    detections = net.forward()

    # Loop through detected objects
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label on the image
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = f"{label}: {confidence * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with detected objects
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    image_path = "test_image.jpg"  # Replace with the path to your image
    detect_objects(image_path)
