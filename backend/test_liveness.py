import cv2
import sys
import os

# Allow Python to find backend modules
sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from backend.models.liveness_detector import LivenessDetector


# Create liveness detector
detector = LivenessDetector()


# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open webcam.")

    detector.close()

    sys.exit()


print("====================================")
print("VeriStream AI - Liveness Test")
print("====================================")
print("Webcam started.")
print("Look at the camera and blink naturally.")
print("Press Q inside the webcam window to quit.")
print("====================================")


while True:

    # Capture frame
    success, frame = camera.read()

    if not success:

        print("ERROR: Could not read webcam frame.")

        break


    # Analyze frame
    result = detector.analyze_frame(frame)


    # --------------------------------
    # Display results
    # --------------------------------

    if result["face_detected"]:

        # Face detected
        cv2.putText(
            frame,
            "FACE DETECTED",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # EAR value
        cv2.putText(
            frame,
            f"EAR: {result['eye_aspect_ratio']}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # Blink count
        cv2.putText(
            frame,
            f"BLINKS: {result['blink_count']}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # Liveness score
        cv2.putText(
            frame,
            f"LIVENESS: {result['liveness_score']}/100",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # Blink notification
        if result["blink_detected"]:

            cv2.putText(
                frame,
                "BLINK DETECTED!",
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )


    else:

        # No face
        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # Show webcam
    cv2.imshow(
        "VeriStream AI - Liveness Test",
        frame
    )


    # Press Q to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# --------------------------------
# Cleanup
# --------------------------------

camera.release()

cv2.destroyAllWindows()

detector.close()

print("Liveness test completed.")