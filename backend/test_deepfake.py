import cv2
import sys
import os
import time


sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)


from backend.models.deepfake_detector import (
    DeepfakeDetector
)


print("======================================")
print("VeriStream AI - Deepfake Test")
print("======================================")

print()
print("Loading AI model...")
print("The first run may take some time.")
print()


detector = DeepfakeDetector()


camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("ERROR: Could not open webcam.")

    sys.exit()


print()
print("Webcam started.")
print()
print("Keep your face inside the camera.")
print()
print("Press Q to quit.")
print("======================================")


last_analysis = 0

result = {
    "deepfake_risk": None,
    "prediction": "collecting",
    "confidence": 0
}


while True:

    success, frame = camera.read()


    if not success:

        print(
            "ERROR: Could not read webcam."
        )

        break


    current_time = time.time()


    # Run the heavy AI model
    # approximately once every 2 seconds.

    if current_time - last_analysis >= 2:

        print()
        print("------------------------------")
        print("Running deepfake analysis...")
        print("------------------------------")


        result = detector.analyze_frame(
            frame
        )


        print(
            "Prediction:",
            result["prediction"]
        )


        print(
            "Deepfake Risk:",
            result["deepfake_risk"],
            "%"
        )


        print(
            "Confidence:",
            result["confidence"],
            "%"
        )


        last_analysis = current_time


    # ==================================
    # DISPLAY RESULT
    # ==================================

    cv2.putText(
        frame,
        "DEEPFAKE AI TEST",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    if result["deepfake_risk"] is not None:

        cv2.putText(
            frame,
            f"Deepfake Risk: "
            f"{result['deepfake_risk']}%",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Prediction: "
            f"{result['prediction']}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Confidence: "
            f"{result['confidence']}%",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            "Collecting...",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    cv2.imshow(
        "VeriStream AI - Deepfake Test",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


camera.release()

cv2.destroyAllWindows()


print()
print("======================================")
print("Deepfake test completed.")
print("======================================")