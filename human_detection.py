import cv2
import numpy as np
import random
import pyautogui
import mss
import mss.tools

# Function to load the pre-trained MobileNet SSD model
def load_model():
    proto = "deploy.prototxt"
    model = "mobilenet_iter_73000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(proto, model)
    if net.empty():
        print("Error loading model files.")
        exit(-1)
    return net

# Function to generate a random offset
def random_offset(max_offset):
    return random.randint(-max_offset, max_offset)

# Function to detect humans in the given frame
def detect_humans(net, frame):
    (h, w) = frame.shape[:2]

    # Create a blob from the input image to feed into the model
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])

            if idx == 15:  # Index 15 corresponds to 'person' in MobileNet SSD
                startX = max(0, int(detections[0, 0, i, 3] * w))
                startY = max(0, int(detections[0, 0, i, 4] * h))
                endX = min(w - 1, int(detections[0, 0, i, 5] * w))
                endY = min(h - 1, int(detections[0, 0, i, 6] * h))

                # Draw the bounding box around the detected person
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Move the mouse pointer to the head of the detected human figure with a random offset
                headX = ((startX + endX) // 2) + random_offset(3)
                headY = startY + random_offset(3)
                pyautogui.moveTo(headX, headY)

# Main function
def main():
    model = load_model()
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Use the first monitor

        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            detect_humans(model, frame)

            # Display the frame with detected humans
            cv2.imshow("Human Detection", frame)

            # Press 'q' to quit the window
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
