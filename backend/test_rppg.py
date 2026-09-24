import cv2
import sys
import os
import time


sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)


from backend.models.face_analyzer import FaceAnalyzer
from backend.models.rppg_detector import RPPGDetector


face_analyzer = FaceAnalyzer()
rppg = RPPGDetector()

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("ERROR: Could not open webcam.")

    face_analyzer.close()

    sys.exit()


print("======================================")
print("VeriStream AI - rPPG Test")
print("======================================")
print("Webcam started.")
print()
print("Keep your face inside the camera.")
print("Remain reasonably still.")
print()
print("Collecting physiological signal...")
print("Press Q to quit.")
print("======================================")


fps = 30
last_analysis = time.time()


while True:

    success, frame = camera.read()

    if not success:

        print("ERROR: Could not read webcam.")
        break

    face_result = face_analyzer.analyze_frame(frame)

    if face_result["face_detected"]:

        face_box = face_result["face_box"]

        region = rppg.extract_face_region(
            frame,
            face_box
        )

        rppg.add_sample(region)

        height, width = frame.shape[:2]

        x = int(face_box["x"] * width)
        y = int(face_box["y"] * height)
        w = int(face_box["width"] * width)
        h = int(face_box["height"] * height)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "FACE DETECTED",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    current_time = time.time()

    if current_time - last_analysis >= 1:

        result = rppg.analyze(fps=fps)

        print("rPPG:", result)

        last_analysis = current_time

    result = rppg.analyze(fps=fps)

    cv2.putText(
        frame,
        f"Samples: {result['samples']}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Signal: {result['signal_strength']}%",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    if result["heart_rate"] is not None:

        cv2.putText(
            frame,
            f"Pulse estimate: {result['heart_rate']} BPM",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "Pulse estimate: collecting...",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        "VeriStream AI - rPPG Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
face_analyzer.close()

print()
print("rPPG test completed.")