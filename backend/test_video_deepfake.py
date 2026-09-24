import os

from models.deepfake_detector import DeepfakeDetector


print("\n")
print("=" * 60)
print("VERISTREAM AI - VIDEO DEEPFAKE TEST")
print("=" * 60)


# =========================================================
# FIND PROJECT ROOT
# =========================================================

project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# =========================================================
# VIDEO PATH
# =========================================================

video_path = os.path.join(
    project_root,
    "test_videos",
    "fake.mp4"
)


print("\nVideo path:")
print(video_path)


# =========================================================
# CHECK FILE
# =========================================================

print("\nChecking video file...")

if not os.path.exists(video_path):

    print("\nERROR: Video file was NOT found.")

    print("\nExpected location:")
    print(video_path)

    print("\nPlease make sure your file is named exactly:")
    print("fake.mp4")

    print("\nAnd placed inside:")
    print(
        os.path.join(
            project_root,
            "test_videos"
        )
    )

    print("\nTest stopped.")
    print("=" * 60)

    exit()


print("Video file found.")

file_size = os.path.getsize(video_path)

print(
    "File size:",
    round(
        file_size / (1024 * 1024),
        2
    ),
    "MB"
)


# =========================================================
# INITIALIZE DETECTOR
# =========================================================

print("\nInitializing Deepfake Detector...\n")

detector = DeepfakeDetector()


# =========================================================
# ANALYZE VIDEO
# =========================================================

result = detector.analyze_video(
    video_path,
    sample_count=8
)


# =========================================================
# DISPLAY RESULT
# =========================================================

print("\n")
print("=" * 60)
print("VIDEO DEEPFAKE RESULT")
print("=" * 60)

print(
    "Status:",
    result.get("status")
)

print(
    "Prediction:",
    result.get("prediction")
)

print(
    "Deepfake Risk:",
    result.get("deepfake_risk"),
    "%"
)

print(
    "Confidence:",
    result.get("confidence"),
    "%"
)

print(
    "Frames Analyzed:",
    result.get("frames_analyzed")
)

print(
    "Total Frames:",
    result.get("total_frames")
)

print(
    "Video FPS:",
    result.get("video_fps")
)

print("=" * 60)


# =========================================================
# FRAME RESULTS
# =========================================================

if result.get("frame_results"):

    print("\nIndividual Frame Results:")
    print("-" * 60)

    for frame_result in result["frame_results"]:

        print(
            f"Frame {frame_result['frame']}: "
            f"{frame_result['prediction']} | "
            f"Fake Risk: "
            f"{frame_result['deepfake_risk']}% | "
            f"Confidence: "
            f"{frame_result['confidence']}%"
        )


print("\nTest completed.")