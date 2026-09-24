import cv2
import mediapipe as mp
import os


class FaceAnalyzer:

    def __init__(self):

        # Path to MediaPipe face detector model
        model_path = os.path.join(
            os.path.dirname(__file__),
            "blaze_face_short_range.tflite"
        )

        # Create MediaPipe Face Detector
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )

        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=0.5
        )

        self.detector = (
            mp.tasks.vision.FaceDetector
            .create_from_options(options)
        )


    def analyze_frame(self, frame):

        """
        Analyze a single webcam frame.

        Returns:
            face_detected
            face_count
            face_box
        """

        # OpenCV uses BGR.
        # MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Run face detection
        result = self.detector.detect(mp_image)

        face_count = len(
            result.detections
        )

        face_box = None

        if face_count > 0:

            bounding_box = (
                result.detections[0].bounding_box
            )

            frame_height, frame_width = frame.shape[:2]

            face_box = {
                "x": round(
                    bounding_box.origin_x / frame_width,
                    4
                ),
                "y": round(
                    bounding_box.origin_y / frame_height,
                    4
                ),
                "width": round(
                    bounding_box.width / frame_width,
                    4
                ),
                "height": round(
                    bounding_box.height / frame_height,
                    4
                )
            }

        return {
            "face_detected": face_count > 0,
            "face_count": face_count,
            "face_box": face_box
        }


    def close(self):

        self.detector.close()