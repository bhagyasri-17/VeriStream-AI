import cv2
import sys
import os

# Allow Python to find our backend modules
sys.path.append(
    os.path.dirname(os.path.dirname(__file__))
)

from backend.models.face_analyzer import FaceAnalyzer


# Create face analyzer
analyzer = FaceAnalyzer()

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    analyzer.close()
    sys.exit()


print("Webcam started.")
print("Press Q to quit.")


while True:

    # Capture frame
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break


    # Analyze frame
    result = analyzer.analyze_frame(frame)


    # If face detected
    if result["face_detected"]:

        face_box = result["face_box"]

        frame_height, frame_width = frame.shape[:2]

        x = int(
            face_box["x"] * frame_width
        )

        y = int(
            face_box["y"] * frame_height
        )

        width = int(
            face_box["width"] * frame_width
        )

        height = int(
            face_box["height"] * frame_height
        )

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        # Display detection text
        cv2.putText(
            frame,
            "Face detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # Show webcam
    cv2.imshow(
        "VeriStream AI - Face Detection",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup
camera.release()
cv2.destroyAllWindows()

analyzer.close()

print("Face detection test completed.")
