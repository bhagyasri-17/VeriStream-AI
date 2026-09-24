import cv2
import numpy as np
from scipy.signal import butter, filtfilt


class RPPGDetector:

    def __init__(self):

        self.signal_buffer = []

        # Maximum signal history
        self.max_buffer_size = 300

        # Minimum samples required
        self.minimum_samples = 150

        # Previous heart rate for smoothing
        self.previous_heart_rate = None

        # Previous confidence for smoothing
        self.previous_confidence = None

        # Previous signal strength for smoothing
        self.previous_signal_strength = None

    # =====================================================
    # EXTRACT FACE REGION
    # =====================================================

    def extract_face_region(self, frame, face_box):

        if face_box is None:
            return None

        height, width = frame.shape[:2]

        x = int(face_box["x"] * width)
        y = int(face_box["y"] * height)

        w = int(face_box["width"] * width)
        h = int(face_box["height"] * height)

        # Forehead / upper-face region
        x1 = x + int(w * 0.20)
        x2 = x + int(w * 0.80)

        y1 = y + int(h * 0.12)
        y2 = y + int(h * 0.38)

        x1 = max(0, x1)
        x2 = min(width, x2)

        y1 = max(0, y1)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        region = frame[y1:y2, x1:x2]

        if region.size == 0:
            return None

        return region

    # =====================================================
    # ADD rPPG SAMPLE
    # =====================================================

    def add_sample(self, region):

        if region is None:
            return

        rgb = cv2.cvtColor(
            region,
            cv2.COLOR_BGR2RGB
        )

        pixels = rgb.reshape(-1, 3)

        # Median is more resistant to noise/outliers
        average_rgb = np.percentile(
            pixels,
            50,
            axis=0
        )

        green_value = average_rgb[1]

        self.signal_buffer.append(
            float(green_value)
        )

        if len(self.signal_buffer) > self.max_buffer_size:

            self.signal_buffer.pop(0)

    # =====================================================
    # BANDPASS FILTER
    # =====================================================

    def filter_signal(self, signal, fps):

        low_frequency = 0.75
        high_frequency = 3.0

        nyquist = fps / 2.0

        if nyquist <= low_frequency:
            return signal

        high_frequency = min(
            high_frequency,
            nyquist * 0.9
        )

        low = low_frequency / nyquist
        high = high_frequency / nyquist

        if low >= high:
            return signal

        b, a = butter(
            3,
            [low, high],
            btype="band"
        )

        try:

            filtered = filtfilt(
                b,
                a,
                signal
            )

            return filtered

        except ValueError:

            return signal

    # =====================================================
    # EMA SMOOTHING
    # =====================================================

    def smooth_value(
        self,
        new_value,
        previous_value,
        alpha=0.15
    ):

        if previous_value is None:

            return new_value

        return (
            alpha * new_value
            +
            (1 - alpha) * previous_value
        )

    # =====================================================
    # ANALYZE rPPG
    # =====================================================

    def analyze(self, fps=10):

        sample_count = len(
            self.signal_buffer
        )

        # ---------------------------------------------
        # Still collecting samples
        # ---------------------------------------------

        if sample_count < self.minimum_samples:

            return {
                "status": "collecting",
                "samples": sample_count,
                "signal_strength": 0,
                "heart_rate": None,
                "rppg_available": False,
                "confidence": 0
            }

        # ---------------------------------------------
        # Convert signal to numpy
        # ---------------------------------------------

        signal = np.array(
            self.signal_buffer,
            dtype=np.float64
        )

        # Remove DC component
        signal = signal - np.mean(signal)

        # ---------------------------------------------
        # Bandpass filtering
        # ---------------------------------------------

        filtered = self.filter_signal(
            signal,
            fps
        )

        # ---------------------------------------------
        # Signal strength
        # ---------------------------------------------

        signal_std = np.std(filtered)

        if signal_std < 0.05:

            return {
                "status": "low_signal",
                "samples": sample_count,
                "signal_strength": 0,
                "heart_rate": None,
                "rppg_available": False,
                "confidence": 0
            }

        # ---------------------------------------------
        # FFT
        # ---------------------------------------------

        frequencies = np.fft.rfftfreq(
            len(filtered),
            d=1.0 / fps
        )

        spectrum = np.abs(
            np.fft.rfft(filtered)
        )

        # Human heart-rate range
        valid = (
            (frequencies >= 0.75)
            &
            (frequencies <= 2.5)
        )

        if not np.any(valid):

            return {
                "status": "no_valid_frequency",
                "samples": sample_count,
                "signal_strength": 0,
                "heart_rate": None,
                "rppg_available": False,
                "confidence": 0
            }

        valid_frequencies = frequencies[valid]
        valid_spectrum = spectrum[valid]

        # ---------------------------------------------
        # Dominant frequency
        # ---------------------------------------------

        dominant_index = np.argmax(
            valid_spectrum
        )

        dominant_frequency = (
            valid_frequencies[
                dominant_index
            ]
        )

        heart_rate = (
            dominant_frequency * 60
        )

        # ---------------------------------------------
        # Calculate spectral confidence
        # ---------------------------------------------

        peak_value = np.max(
            valid_spectrum
        )

        median_value = np.median(
            valid_spectrum
        )

        if median_value <= 0:

            peak_ratio = 0

        else:

            peak_ratio = (
                peak_value /
                median_value
            )

        raw_confidence = np.clip(
            (peak_ratio - 1) * 25,
            0,
            100
        )

        # ---------------------------------------------
        # Signal strength
        # ---------------------------------------------

        raw_signal_strength = np.clip(
            signal_std * 100,
            0,
            100
        )

        # ---------------------------------------------
        # SMOOTH CONFIDENCE
        # ---------------------------------------------

        self.previous_confidence = (
            self.smooth_value(
                raw_confidence,
                self.previous_confidence,
                alpha=0.12
            )
        )

        confidence = int(
            round(
                self.previous_confidence
            )
        )

        # ---------------------------------------------
        # SMOOTH SIGNAL STRENGTH
        # ---------------------------------------------

        self.previous_signal_strength = (
            self.smooth_value(
                raw_signal_strength,
                self.previous_signal_strength,
                alpha=0.12
            )
        )

        signal_strength = int(
            round(
                self.previous_signal_strength
            )
        )

        # ---------------------------------------------
        # SMOOTH HEART RATE
        # ---------------------------------------------

        if confidence >= 40:

            if self.previous_heart_rate is None:

                self.previous_heart_rate = (
                    heart_rate
                )

            else:

                self.previous_heart_rate = (
                    0.85 *
                    self.previous_heart_rate
                    +
                    0.15 *
                    heart_rate
                )

            heart_rate = round(
                self.previous_heart_rate
            )

        else:

            heart_rate = None

        # ---------------------------------------------
        # Availability
        # ---------------------------------------------

        rppg_available = (
            heart_rate is not None
            and confidence >= 40
        )

        # ---------------------------------------------
        # Final result
        # ---------------------------------------------

        return {

            "status": "analyzed",

            "samples": sample_count,

            "signal_strength":
                signal_strength,

            "heart_rate":
                heart_rate,

            "rppg_available":
                rppg_available,

            "confidence":
                confidence
        }

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        self.signal_buffer = []

        self.previous_heart_rate = None

        self.previous_confidence = None

        self.previous_signal_strength = None