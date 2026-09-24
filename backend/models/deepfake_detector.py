from transformers import pipeline
from PIL import Image
import cv2
import numpy as np


class DeepfakeDetector:

    def __init__(self):

        print("Loading Deepfake Detection Model...")

        self.classifier = pipeline(
            "image-classification",
            model="dima806/deepfake_vs_real_image_detection"
        )

        print("Deepfake Detection Model Loaded.")


    # =========================================================
    # ANALYZE SINGLE FRAME
    # =========================================================

    def analyze_frame(self, frame):

        if frame is None:

            return {
                "status": "error",
                "deepfake_risk": None,
                "prediction": "unknown",
                "confidence": 0
            }

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb_frame
        )

        results = self.classifier(image)

        if not results:

            return {
                "status": "error",
                "deepfake_risk": None,
                "prediction": "unknown",
                "confidence": 0
            }

        print(
            "Deepfake model result:",
            results
        )

        fake_score = 0.0
        real_score = 0.0

        for result in results:

            label = result["label"].lower()

            score = float(
                result["score"]
            )

            if "fake" in label:
                fake_score = score

            elif "real" in label:
                real_score = score


        # Keep decimal precision internally
        deepfake_risk = round(
            fake_score * 100,
            2
        )


        if fake_score >= real_score:

            prediction = "potential_fake"

        else:

            prediction = "likely_real"


        confidence = round(
            max(
                fake_score,
                real_score
            ) * 100,
            2
        )


        return {
            "status": "analyzed",

            "deepfake_risk":
                deepfake_risk,

            "prediction":
                prediction,

            "confidence":
                confidence
        }


    # =========================================================
    # ANALYZE VIDEO
    # =========================================================

    def analyze_video(
        self,
        video_path,
        sample_count=16
    ):

        print(
            "\nStarting video deepfake analysis..."
        )

        cap = cv2.VideoCapture(
            video_path
        )


        if not cap.isOpened():

            print(
                "ERROR: Could not open video."
            )

            return {
                "status": "error",
                "deepfake_risk": None,
                "prediction": "unknown",
                "confidence": 0,
                "frames_analyzed": 0
            }


        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        fps = cap.get(
            cv2.CAP_PROP_FPS
        )


        if fps <= 0:

            fps = 30


        if total_frames <= 0:

            cap.release()

            return {
                "status": "error",
                "deepfake_risk": None,
                "prediction": "unknown",
                "confidence": 0,
                "frames_analyzed": 0
            }


        # =====================================================
        # SAMPLE MORE FRAMES
        # =====================================================

        number_of_samples = min(
            sample_count,
            total_frames
        )


        frame_indices = np.linspace(
            0,
            total_frames - 1,
            number_of_samples,
            dtype=int
        )


        fake_scores = []
        real_scores = []

        frame_results = []


        # =====================================================
        # ANALYZE FRAMES
        # =====================================================

        for index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(index)
            )


            success, frame = cap.read()


            if not success:

                print(
                    f"Could not read frame {index}"
                )

                continue


            result = self.analyze_frame(
                frame
            )


            if result["status"] != "analyzed":

                continue


            fake_risk = (
                float(
                    result["deepfake_risk"]
                ) / 100.0
            )


            real_probability = (
                1.0 - fake_risk
            )


            fake_scores.append(
                fake_risk
            )


            real_scores.append(
                real_probability
            )


            frame_results.append({

                "frame":
                    int(index),

                "deepfake_risk":
                    result["deepfake_risk"],

                "prediction":
                    result["prediction"],

                "confidence":
                    result["confidence"]

            })


            print(
                f"Frame {index}: "
                f"{result['prediction']} "
                f"("
                f"{result['deepfake_risk']:.2f}% "
                f"fake risk)"
            )


        cap.release()


        # =====================================================
        # CHECK RESULTS
        # =====================================================

        if not fake_scores:

            return {
                "status": "error",
                "deepfake_risk": None,
                "prediction": "unknown",
                "confidence": 0,
                "frames_analyzed": 0
            }


        # =====================================================
        # AGGREGATE VIDEO RESULT
        # =====================================================

        average_fake_score = np.mean(
            fake_scores
        )


        average_real_score = np.mean(
            real_scores
        )


        deepfake_risk = round(
            average_fake_score * 100,
            2
        )


        confidence = round(
            max(
                average_fake_score,
                average_real_score
            ) * 100,
            2
        )


        if (
            average_fake_score
            >=
            average_real_score
        ):

            prediction = "potential_fake"

        else:

            prediction = "likely_real"


        # =====================================================
        # ADD TEMPORAL VARIATION
        # =====================================================

        score_variation = round(
            float(
                np.std(fake_scores) * 100
            ),
            2
        )


        result = {

            "status":
                "analyzed",

            "deepfake_risk":
                deepfake_risk,

            "prediction":
                prediction,

            "confidence":
                confidence,

            "frames_analyzed":
                len(frame_results),

            "total_frames":
                total_frames,

            "video_fps":
                round(
                    fps,
                    2
                ),

            "score_variation":
                score_variation,

            "frame_results":
                frame_results
        }


        # =====================================================
        # FINAL LOG
        # =====================================================

        print(
            "\nVideo deepfake analysis complete."
        )

        print(
            "Frames analyzed:",
            result["frames_analyzed"]
        )

        print(
            "Average fake risk:",
            result["deepfake_risk"],
            "%"
        )

        print(
            "Score variation:",
            result["score_variation"],
            "%"
        )

        print(
            "Final prediction:",
            result["prediction"]
        )


        return result