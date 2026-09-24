import cv2
import mediapipe as mp
import os
import math


class LivenessDetector:

    def __init__(self):

        model_path = os.path.join(
            os.path.dirname(__file__),
            "face_landmarker.task"
        )

        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = (
            mp.tasks.vision.FaceLandmarker
            .create_from_options(options)
        )

        # Blink tracking
        self.closed_frames = 0
        self.blink_count = 0

        # Movement tracking
        self.previous_nose = None
        self.movement_frames = 0

        # Analysis counter
        self.total_frames = 0

    # =========================================================
    # DISTANCE BETWEEN TWO LANDMARKS
    # =========================================================
    def calculate_distance(self, point1, point2):

        return math.sqrt(
            (point1.x - point2.x) ** 2 +
            (point1.y - point2.y) ** 2
        )

    # =========================================================
    # EYE ASPECT RATIO
    # =========================================================
    def calculate_ear(self, landmarks, indices):

        p1 = landmarks[indices[0]]
        p2 = landmarks[indices[1]]
        p3 = landmarks[indices[2]]
        p4 = landmarks[indices[3]]
        p5 = landmarks[indices[4]]
        p6 = landmarks[indices[5]]

        vertical_1 = self.calculate_distance(
            p2,
            p6
        )

        vertical_2 = self.calculate_distance(
            p3,
            p5
        )

        horizontal = self.calculate_distance(
            p1,
            p4
        )

        if horizontal == 0:
            return 0.0

        return (
            vertical_1 + vertical_2
        ) / (2.0 * horizontal)

    # =========================================================
    # CALCULATE HEAD MOVEMENT
    # =========================================================
    def calculate_movement(self, nose):

        if self.previous_nose is None:

            self.previous_nose = nose

            return 0.0

        movement = self.calculate_distance(
            nose,
            self.previous_nose
        )

        self.previous_nose = nose

        return movement

    # =========================================================
    # ANALYZE FRAME
    # =========================================================
    def analyze_frame(self, frame):

        self.total_frames += 1

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect face landmarks
        result = self.landmarker.detect(
            mp_image
        )

        # =====================================================
        # NO FACE DETECTED
        # =====================================================
        if not result.face_landmarks:

            self.closed_frames = 0
            self.previous_nose = None

            return {
                "face_detected": False,
                "blink_detected": False,
                "blink_count": self.blink_count,
                "eye_aspect_ratio": 0.0,
                "head_movement": False,
                "movement_amount": 0.0,
                "movement_frames": self.movement_frames,
                "liveness_score": 0
            }

        # Get first detected face
        landmarks = result.face_landmarks[0]

        # =====================================================
        # EYE LANDMARKS
        # =====================================================

        left_eye = [
            33,
            160,
            158,
            133,
            153,
            144
        ]

        right_eye = [
            362,
            385,
            387,
            263,
            373,
            380
        ]

        # Calculate left and right EAR
        left_ear = self.calculate_ear(
            landmarks,
            left_eye
        )

        right_ear = self.calculate_ear(
            landmarks,
            right_eye
        )

        # Average EAR
        ear = (
            left_ear + right_ear
        ) / 2.0

        # =====================================================
        # BLINK DETECTION
        # =====================================================

        blink_threshold = 0.20
        blink_detected = False

        if ear < blink_threshold:

            self.closed_frames += 1

        else:

            if self.closed_frames >= 2:

                self.blink_count += 1
                blink_detected = True

            self.closed_frames = 0

        # =====================================================
        # HEAD MOVEMENT
        # =====================================================

        # Nose landmark
        nose = landmarks[1]

        movement_amount = self.calculate_movement(
            nose
        )

        # Ignore very small movements
        movement_threshold = 0.008

        head_movement = (
            movement_amount >
            movement_threshold
        )

        if head_movement:

            self.movement_frames += 1

        # =====================================================
        # CALCULATE LIVENESS SCORE
        # =====================================================

        score = 30

        # Face detected
        score += 20

        # Blink evidence
        if self.blink_count >= 1:

            score += 30

        # Movement evidence
        if self.movement_frames >= 3:

            score += 20

        elif self.movement_frames >= 1:

            score += 10

        # Maximum score = 100
        liveness_score = min(
            score,
            100
        )

        # =====================================================
        # RETURN RESULT
        # =====================================================

        return {
            "face_detected": True,

            "blink_detected": blink_detected,

            "blink_count": self.blink_count,

            "eye_aspect_ratio": round(
                ear,
                3
            ),

            "head_movement": head_movement,

            "movement_amount": round(
                movement_amount,
                4
            ),

            "movement_frames": self.movement_frames,

            "liveness_score": liveness_score
        }

    # =========================================================
    # RESET LIVENESS STATE
    # =========================================================
    def reset(self):

        self.closed_frames = 0
        self.blink_count = 0
        self.previous_nose = None
        self.movement_frames = 0
        self.total_frames = 0

    # =========================================================
    # CLOSE MODEL
    # =========================================================
    def close(self):

        self.landmarker.close()