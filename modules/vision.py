from pathlib import Path
from typing import Any
from urllib.request import urlretrieve
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def _ensure_model_file(filename: str, url: str) -> Path | None:
    """Download MediaPipe .task bundle locally if not already present."""
    models_dir = Path(__file__).resolve().parent.parent / "assets" / "mp_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / filename
    if target.exists() and target.stat().st_size > 100_000:
        return target
    # Download to a temp name so an interrupted download never leaves a corrupt model behind.
    partial = target.with_suffix(target.suffix + ".part")
    try:
        urlretrieve(url, str(partial))
        if partial.exists() and partial.stat().st_size > 100_000:
            partial.replace(target)
            return target
    except Exception:
        pass
    finally:
        partial.unlink(missing_ok=True)
    return None


def _compute_lighting_score(frame_bgr: np.ndarray) -> float:
    """Estimate lighting quality (0-100) from frame luminance."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    ideal_center = 135.0
    distance = abs(mean_brightness - ideal_center)
    score = max(0.0, min(100.0, 100.0 - (distance / 135.0) * 100.0))
    return round(score, 1)


def _compute_eye_contact_from_landmarks(landmarks: list[Any]) -> float:
    """Approximate horizontal iris/eye centering (0-100) from FaceLandmarker landmarks."""
    try:
        # Left eye corners (33, 133) and left iris center (468) when 478 landmarks are present
        if len(landmarks) > 468:
            iris_x = landmarks[468].x
            left_x = min(landmarks[33].x, landmarks[133].x)
            right_x = max(landmarks[33].x, landmarks[133].x)
            eye_width = max(right_x - left_x, 1e-5)
            rel_pos = (iris_x - left_x) / eye_width
            deviation = abs(rel_pos - 0.5)
            return round(max(0.0, min(100.0, 100.0 - (deviation / 0.35) * 100.0)), 1)

        # Fallback using nose tip (1) horizontal centering
        nose_dev = abs(landmarks[1].x - 0.5)
        return round(max(0.0, min(100.0, 100.0 - (nose_dev / 0.25) * 100.0)), 1)
    except Exception:
        return 50.0


def _compute_posture_from_landmarks(pose_landmarks: list[Any]) -> float:
    """Estimate shoulder levelness and symmetry (0-100) from PoseLandmarker landmarks."""
    try:
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]

        # Only default when visibility is missing; 0.0 means "not visible", not "unknown".
        l_vis = getattr(left_shoulder, "visibility", None)
        r_vis = getattr(right_shoulder, "visibility", None)
        l_vis = 0.8 if l_vis is None else l_vis
        r_vis = 0.8 if r_vis is None else r_vis
        if min(l_vis, r_vis) < 0.4:
            return 60.0

        y_diff = abs(left_shoulder.y - right_shoulder.y)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)

        tilt_penalty = min(50.0, (y_diff / 0.12) * 50.0)
        width_score = max(0.0, min(50.0, (shoulder_width / 0.35) * 50.0))
        return round(max(0.0, min(100.0, width_score + (50.0 - tilt_penalty))), 1)
    except Exception:
        return 60.0


def analyze_video_visuals(video_path: str, sample_fps: float = 2.0) -> dict[str, Any]:
    """Analyze a video file for eye contact, posture alignment, and lighting."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 0:
        native_fps = 30.0

    frame_interval = max(1, int(round(native_fps / max(sample_fps, 0.5))))

    face_model_path = _ensure_model_file("face_landmarker.task", FACE_LANDMARKER_URL)
    pose_model_path = _ensure_model_file("pose_landmarker_lite.task", POSE_LANDMARKER_URL)

    use_tasks_api = (
        hasattr(mp, "tasks")
        and hasattr(mp.tasks, "vision")
        and face_model_path is not None
    )

    face_landmarker = None
    pose_landmarker = None
    haar_cascade = None

    if use_tasks_api:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=str(face_model_path))
        face_options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)

        if pose_model_path is not None:
            pose_base = mp_python.BaseOptions(model_asset_path=str(pose_model_path))
            pose_options = mp_vision.PoseLandmarkerOptions(
                base_options=pose_base,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)
    else:
        cascade_file = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        haar_cascade = cv2.CascadeClassifier(str(cascade_file))

    records: list[dict[str, Any]] = []
    frame_idx = 0
    sampled_count = 0
    faces_detected = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_sec = round(frame_idx / native_fps, 1)
                lighting_score = _compute_lighting_score(frame_bgr)

                if use_tasks_api and face_landmarker is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    face_res = face_landmarker.detect(mp_image)

                    if face_res.face_landmarks:
                        faces_detected += 1
                        eye_score = _compute_eye_contact_from_landmarks(face_res.face_landmarks[0])
                    else:
                        eye_score = 35.0

                    if pose_landmarker is not None:
                        pose_res = pose_landmarker.detect(mp_image)
                        if pose_res.pose_landmarks:
                            posture_score = _compute_posture_from_landmarks(pose_res.pose_landmarks[0])
                        else:
                            posture_score = 55.0
                    else:
                        posture_score = 65.0 if face_res.face_landmarks else 50.0
                else:
                    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    h, w = gray.shape
                    detected_faces = (
                        haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                        if haar_cascade is not None
                        else ()
                    )
                    if len(detected_faces) > 0:
                        faces_detected += 1
                        x, y, fw, fh = detected_faces[0]
                        center_x = (x + fw / 2.0) / max(w, 1)
                        dev = abs(center_x - 0.5)
                        eye_score = round(max(0.0, min(100.0, 100.0 - (dev / 0.3) * 100.0)), 1)
                        posture_score = 72.0
                    else:
                        eye_score = 35.0
                        posture_score = 55.0

                records.append(
                    {
                        "time_sec": timestamp_sec,
                        "eye_contact": eye_score,
                        "posture": posture_score,
                        "lighting": lighting_score,
                    }
                )
                sampled_count += 1

            frame_idx += 1
    finally:
        cap.release()
        if face_landmarker is not None:
            face_landmarker.close()
        if pose_landmarker is not None:
            pose_landmarker.close()

    if not records:
        empty_df = pd.DataFrame(columns=["time_sec", "eye_contact", "posture", "lighting"])
        return {
            "eye_contact_score": 0.0,
            "posture_score": 0.0,
            "lighting_score": 0.0,
            "face_detection_rate": 0.0,
            "samples_analyzed": 0,
            "timeline_df": empty_df,
        }

    df = pd.DataFrame(records)
    return {
        "eye_contact_score": round(float(df["eye_contact"].mean()), 1),
        "posture_score": round(float(df["posture"].mean()), 1),
        "lighting_score": round(float(df["lighting"].mean()), 1),
        "face_detection_rate": round((faces_detected / max(sampled_count, 1)) * 100.0, 1),
        "samples_analyzed": sampled_count,
        "timeline_df": df,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python modules/vision.py <path_to_video.mp4>")
        sys.exit(1)

    result = analyze_video_visuals(sys.argv[1])
    summary = {k: v for k, v in result.items() if k != "timeline_df"}
    print("Vision Analysis Summary:", summary)
    print("\nFirst 5 timeline rows:")
    print(result["timeline_df"].head())