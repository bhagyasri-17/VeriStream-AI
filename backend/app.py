from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

import os
import cv2
import numpy as np
import time
import threading
import uuid

from models.face_analyzer import FaceAnalyzer
from models.liveness_detector import LivenessDetector
from models.rppg_detector import RPPGDetector
from models.deepfake_detector import DeepfakeDetector


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)

print("=" * 60)
print("Initializing VeriStream AI...")
print("=" * 60)


# ============================================================
# AI COMPONENTS
# ============================================================

print("Loading Face Analyzer...")
face_analyzer = FaceAnalyzer()
print("Face Analyzer Loaded.")

print("Loading Liveness Detector...")
liveness_detector = LivenessDetector()
print("Liveness Detector Loaded.")

print("Loading rPPG Detector...")
rppg_detector = RPPGDetector()
print("rPPG Detector Loaded.")

print("Loading Deepfake Detector...")
deepfake_detector = DeepfakeDetector()
print("Deepfake Detector Loaded.")

print("=" * 60)
print("All AI components initialized successfully.")
print("=" * 60)


# ============================================================
# SESSION MANAGEMENT
# ============================================================

current_session_id = None
session_active = False
session_lock = threading.Lock()

# Store latest result of each session in memory.
# This is NOT permanent database storage.
session_summaries = {}


# ============================================================
# DEEPFAKE STATE
# ============================================================

last_deepfake_analysis = 0

last_deepfake_result = {
    "status": "collecting",
    "deepfake_risk": None,
    "prediction": "collecting",
    "confidence": 0
}

deepfake_lock = threading.Lock()
deepfake_running = False

DEEPFAKE_INTERVAL = 5


# ============================================================
# TRUST ENGINE WEIGHTS
# ============================================================

TRUST_WEIGHTS = {
    "face": 20,
    "liveness": 25,
    "rppg": 20,
    "av_sync": 15,
    "deepfake": 20
}


# ============================================================
# SAFE SCORE
# ============================================================

def safe_score(value):

    if value is None:
        return None

    try:
        value = float(value)

        if not np.isfinite(value):
            return None

        return max(
            0,
            min(100, value)
        )

    except (TypeError, ValueError):
        return None


# ============================================================
# TRUST ENGINE
# ============================================================

def calculate_trust_score(
    face_score,
    liveness_score,
    rppg_score,
    av_sync_score,
    deepfake_risk
):

    signals = {
        "face": safe_score(face_score),

        "liveness": safe_score(liveness_score),

        "rppg": safe_score(rppg_score),

        "av_sync": safe_score(av_sync_score),

        "deepfake": (
            None
            if deepfake_risk is None
            else safe_score(
                100 - deepfake_risk
            )
        )
    }

    weighted_score = 0
    available_weight = 0

    breakdown = {}

    available_signals = []
    missing_signals = []

    for signal_name, weight in TRUST_WEIGHTS.items():

        score = signals[signal_name]

        if score is not None:

            contribution = (
                score * weight / 100
            )

            weighted_score += contribution
            available_weight += weight

            available_signals.append(
                signal_name
            )

            breakdown[signal_name] = {
                "score": round(
                    score,
                    2
                ),

                "weight": weight,

                "contribution": round(
                    contribution,
                    2
                )
            }

        else:

            missing_signals.append(
                signal_name
            )

            breakdown[signal_name] = {
                "score": None,
                "weight": weight,
                "contribution": 0
            }

    # ========================================================
    # NO SIGNALS AVAILABLE
    # ========================================================

    if available_weight == 0:

        return {
            "score": None,

            "risk_level": "unknown",

            "status": "collecting",

            "coverage": 0,

            "available_signals": [],

            "missing_signals": list(
                TRUST_WEIGHTS.keys()
            ),

            "breakdown": breakdown
        }

    # ========================================================
    # CONSERVATIVE TRUST SCORE
    # ========================================================

    # Missing signals are NOT ignored.
    # Their contribution remains zero.

    final_score = weighted_score

    coverage = (
        available_weight /
        sum(TRUST_WEIGHTS.values())
    ) * 100

    final_score = round(
        final_score
    )

    # ========================================================
    # RISK LEVEL
    # ========================================================

    if final_score >= 80:

        risk_level = "low"

    elif final_score >= 50:

        risk_level = "medium"

    else:

        risk_level = "high"

    # ========================================================
    # VERIFICATION STATUS
    # ========================================================

    if (
        len(available_signals)
        == len(TRUST_WEIGHTS)
    ):

        status = "complete"

    else:

        status = "preliminary"

    return {
        "score": final_score,

        "risk_level": risk_level,

        "status": status,

        "coverage": round(
            coverage
        ),

        "available_signals":
            available_signals,

        "missing_signals":
            missing_signals,

        "breakdown":
            breakdown
    }


# ============================================================
# RESET AI STATE
# ============================================================

def reset_ai_state():

    global last_deepfake_analysis
    global last_deepfake_result
    global deepfake_running

    print(
        "Resetting AI state for new session..."
    )

    # --------------------------------------------------------
    # Reset rPPG
    # --------------------------------------------------------

    try:

        rppg_detector.reset()

    except Exception as error:

        print(
            "rPPG reset error:",
            error
        )

    # --------------------------------------------------------
    # Reset Liveness
    # --------------------------------------------------------

    try:

        liveness_detector.reset()

    except Exception as error:

        print(
            "Liveness reset error:",
            error
        )

    # --------------------------------------------------------
    # Reset Deepfake
    # --------------------------------------------------------

    last_deepfake_analysis = 0

    last_deepfake_result = {
        "status": "collecting",

        "deepfake_risk": None,

        "prediction": "collecting",

        "confidence": 0
    }

    deepfake_running = False

    print(
        "AI state reset successfully."
    )


# ============================================================
# START SESSION
# ============================================================

@app.route(
    "/api/session/start",
    methods=["POST"]
)
def start_session():

    global current_session_id
    global session_active

    with session_lock:

        # ----------------------------------------------------
        # End previous session if one exists
        # ----------------------------------------------------

        if current_session_id:

            print(
                "Previous session detected. "
                "Replacing it with a new session."
            )

        # ----------------------------------------------------
        # Reset AI components
        # ----------------------------------------------------

        reset_ai_state()

        # ----------------------------------------------------
        # Generate unique session ID
        # ----------------------------------------------------

        current_session_id = str(
            uuid.uuid4()
        )

        session_active = True

        # ----------------------------------------------------
        # Create session summary
        # ----------------------------------------------------

        session_summaries[
            current_session_id
        ] = {

            "session_id":
                current_session_id,

            "status":
                "active",

            "final_result":
                None,

            "created_at":
                time.time(),

            "ended_at":
                None
        }

        print()
        print("=" * 60)
        print("NEW VERIFICATION SESSION")
        print(
            "Session ID:",
            current_session_id
        )
        print("=" * 60)

        return jsonify({

            "status":
                "success",

            "session_id":
                current_session_id,

            "message":
                "Verification session started."
        })


# ============================================================
# END SESSION
# ============================================================

@app.route(
    "/api/session/end",
    methods=["POST"]
)
def end_session():

    global current_session_id
    global session_active

    with session_lock:

        if not current_session_id:

            return jsonify({

                "status":
                    "success",

                "message":
                    "No active session."
            })

        session_id = current_session_id

        # ----------------------------------------------------
        # Mark session as ended
        # ----------------------------------------------------

        if session_id in session_summaries:

            session_summaries[
                session_id
            ]["status"] = "ended"

            session_summaries[
                session_id
            ]["ended_at"] = time.time()

        # ----------------------------------------------------
        # Save final result
        # ----------------------------------------------------

        final_result = None

        if session_id in session_summaries:

            final_result = (
                session_summaries[
                    session_id
                ]["final_result"]
            )

        print()
        print("=" * 60)
        print("VERIFICATION SESSION ENDED")
        print(
            "Session ID:",
            session_id
        )
        print(
            "Final result:",
            final_result
        )
        print("=" * 60)

        # ----------------------------------------------------
        # Clear active session
        # ----------------------------------------------------

        current_session_id = None
        session_active = False

        return jsonify({

            "status":
                "success",

            "session_id":
                session_id,

            "message":
                "Verification session ended.",

            "final_result":
                final_result
        })


# ============================================================
# DEEPFAKE ANALYSIS
# ============================================================

def run_deepfake_analysis(
    frame,
    session_id
):

    global last_deepfake_analysis
    global last_deepfake_result
    global deepfake_running

    if deepfake_running:
        return

    current_time = time.time()

    if (
        current_time -
        last_deepfake_analysis
        < DEEPFAKE_INTERVAL
    ):
        return

    if not deepfake_lock.acquire(
        blocking=False
    ):
        return

    try:

        deepfake_running = True

        last_deepfake_analysis = (
            current_time
        )

        print()
        print(
            "Running deepfake analysis..."
        )

        print(
            "Session:",
            session_id
        )

        result = (
            deepfake_detector
            .analyze_frame(
                frame
            )
        )

        last_deepfake_result = result

        print(
            "Deepfake result:",
            result
        )

    except Exception as error:

        print(
            "Deepfake analysis error:",
            error
        )

        last_deepfake_result = {

            "status":
                "error",

            "deepfake_risk":
                None,

            "prediction":
                "unknown",

            "confidence":
                0
        }

    finally:

        deepfake_running = False

        deepfake_lock.release()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    frontend_path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                __file__
            )
        ),
        "frontend"
    )

    return send_from_directory(
        frontend_path,
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/api/health"
)
def health():

    return jsonify({

        "status":
            "healthy",

        "service":
            "VeriStream AI",

        "version":
            "1.0"
    })


# ============================================================
# ANALYZE FRAME
# ============================================================

@app.route(
    "/api/analyze",
    methods=["POST"]
)
def analyze():

    global last_deepfake_result

    # ========================================================
    # CHECK SESSION
    # ========================================================

    session_id = request.form.get(
        "session_id"
    )

    if not session_id:

        return jsonify({

            "status":
                "error",

            "message":
                "No session ID provided."
        }), 400

    if (
        not session_active
        or
        session_id != current_session_id
    ):

        return jsonify({

            "status":
                "error",

            "message":
                "Invalid or inactive verification session."
        }), 409

    # ========================================================
    # CHECK FRAME
    # ========================================================

    if "frame" not in request.files:

        return jsonify({

            "status":
                "error",

            "message":
                "No frame received."
        }), 400

    file = request.files["frame"]

    image_bytes = file.read()

    np_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        return jsonify({

            "status":
                "error",

            "message":
                "Unable to decode image."
        }), 400

    # ========================================================
    # FACE ANALYSIS
    # ========================================================

    face_result = (
        face_analyzer
        .analyze_frame(
            frame
        )
    )

    # ========================================================
    # LIVENESS ANALYSIS
    # ========================================================

    liveness_result = (
        liveness_detector
        .analyze_frame(
            frame
        )
    )

    # ========================================================
    # RPPG
    # ========================================================

    if face_result[
        "face_detected"
    ]:

        face_box = (
            face_result[
                "face_box"
            ]
        )

        region = (
            rppg_detector
            .extract_face_region(
                frame,
                face_box
            )
        )

        rppg_detector.add_sample(
            region
        )

    rppg_result = (
        rppg_detector
        .analyze(
            fps=10
        )
    )

    # ========================================================
    # AUDIO / VIDEO ACTIVITY
    # ========================================================

    av_sync_score = request.form.get(
        "av_sync_score"
    )

    if av_sync_score is not None:

        try:

            av_sync_score = float(
                av_sync_score
            )

        except ValueError:

            av_sync_score = None

    # ========================================================
    # DEEPFAKE
    # ========================================================

    if face_result[
        "face_detected"
    ]:

        deepfake_frame = frame.copy()

        thread = threading.Thread(
            target=run_deepfake_analysis,

            args=(
                deepfake_frame,
                session_id
            ),

            daemon=True
        )

        thread.start()

    deepfake_result = (
        last_deepfake_result
    )

    # ========================================================
    # TRUST SIGNALS
    # ========================================================

    face_score = (

        100
        if face_result[
            "face_detected"
        ]
        else 0
    )

    liveness_score = (
        liveness_result.get(
            "liveness_score"
        )
    )

    if rppg_result.get(
        "rppg_available"
    ):

        rppg_score = (
            rppg_result.get(
                "confidence"
            )
        )

    else:

        rppg_score = None

    deepfake_risk = (
        deepfake_result.get(
            "deepfake_risk"
        )
    )

    # ========================================================
    # TRUST ENGINE
    # ========================================================

    trust_result = (
        calculate_trust_score(

            face_score=face_score,

            liveness_score=liveness_score,

            rppg_score=rppg_score,

            av_sync_score=av_sync_score,

            deepfake_risk=deepfake_risk
        )
    )

    # ========================================================
    # COMPLETE ANALYSIS OBJECT
    # ========================================================

    analysis_result = {

        "face":
            face_result,

        "liveness":
            liveness_result,

        "rppg":
            rppg_result,

        "deepfake":
            deepfake_result,

        "trust":
            trust_result
    }

    # ========================================================
    # SAVE LATEST SESSION RESULT
    # ========================================================

    if session_id in session_summaries:

        session_summaries[
            session_id
        ][
            "final_result"
        ] = analysis_result

    # ========================================================
    # RESPONSE
    # ========================================================

    return jsonify({

        "status":
            "success",

        "session_id":
            session_id,

        "analysis":
            analysis_result
    })


# ============================================================
# SESSION RESULT
# ============================================================

@app.route(
    "/api/session/result",
    methods=["GET"]
)
def session_result():

    session_id = request.args.get(
        "session_id"
    )

    if not session_id:

        return jsonify({

            "status":
                "error",

            "message":
                "Session ID required."
        }), 400

    if session_id not in session_summaries:

        return jsonify({

            "status":
                "error",

            "message":
                "Session not found."
        }), 404

    session = session_summaries[
        session_id
    ]

    return jsonify({

        "status":
            "success",

        "session":
            session
    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("VeriStream AI Server")
    print("=" * 60)
    print(
        "Open: http://127.0.0.1:5000"
    )
    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )