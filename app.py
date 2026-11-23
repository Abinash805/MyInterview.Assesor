import json
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from database.db import clear_sessions, get_sessions, init_db, save_session
from modules.audio import analyze_audio_file, analyze_video_audio
from modules.nlp import evaluate_transcript_nlp
from modules.scoring import compute_predictive_score, get_target_score
from modules.vision import (
    _compute_eye_contact_from_landmarks,
    _compute_lighting_score,
    _ensure_model_file,
    FACE_LANDMARKER_URL,
    analyze_video_visuals,
)

load_dotenv()

st.set_page_config(
    page_title="MyInterview.Assessor",
    layout="wide",
    page_icon="🎯",
)

st.markdown(
    """
    <style>
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 14px 16px;
        border-radius: 8px;
        border-left: 4px solid #38bdf8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

init_db()

SAMPLE_QUESTIONS = {
    "Software Engineer": [
        "Tell me about a time you debugged a slow API or production bottleneck. What was your approach and impact?",
        "How do you design a scalable system with caching and rate limiting? Walk me through trade-offs.",
        "Describe a time you disagreed with a teammate on an engineering design and how you resolved it.",
    ],
    "Product Manager": [
        "Tell me about a time you had to prioritize competing features under a tight deadline.",
        "Describe a product decision you made using data and what the outcome was.",
    ],
    "Data Analyst": [
        "Tell me about a time you cleaned messy data or uncovered an insight that changed a decision.",
        "How do you validate whether a metric shift is real or just noise?",
    ],
    "Marketing Executive": [
        "Tell me about a campaign you ran—how did you measure success and optimize performance?",
        "Describe a time positioning or messaging didn't resonate at first and how you adapted.",
    ],
}

if "role" not in st.session_state:
    st.session_state.role = "Software Engineer"
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "Mid-Level"
if "question" not in st.session_state:
    st.session_state.question = SAMPLE_QUESTIONS["Software Engineer"][0]
if "latest_evaluation" not in st.session_state:
    st.session_state.latest_evaluation = None


def _analyze_live_snapshot(camera_buffer: Any) -> dict[str, Any]:
    """Analyze a single live webcam frame for eye contact and lighting."""
    file_bytes = np.asarray(bytearray(camera_buffer.getvalue()), dtype=np.uint8)
    frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return {
            "eye_contact_score": 65.0,
            "posture_score": 70.0,
            "lighting_score": 65.0,
            "face_detection_rate": 0.0,
            "samples_analyzed": 0,
            "timeline_df": pd.DataFrame(columns=["time_sec", "eye_contact", "posture", "lighting"]),
        }

    lighting_score = _compute_lighting_score(frame_bgr)
    eye_score = 60.0
    face_detected = 0

    model_path = _ensure_model_file("face_landmarker.task", FACE_LANDMARKER_URL)
    if model_path is not None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
            )
            with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                res = landmarker.detect(mp_image)
                if res.face_landmarks:
                    face_detected = 1
                    eye_score = _compute_eye_contact_from_landmarks(res.face_landmarks[0])
        except Exception:
            eye_score = 60.0

    posture_score = 75.0 if face_detected else 60.0
    df = pd.DataFrame([{"time_sec": 0.0, "eye_contact": eye_score, "posture": posture_score, "lighting": lighting_score}])
    return {
        "eye_contact_score": eye_score,
        "posture_score": posture_score,
        "lighting_score": lighting_score,
        "face_detection_rate": float(face_detected * 100),
        "samples_analyzed": 1,
        "timeline_df": df,
    }


def _build_markdown_report(eval_data: dict[str, Any]) -> str:
    role = eval_data["role"]
    difficulty = eval_data["difficulty"]
    question = eval_data["question"]
    scoring = eval_data["scoring"]
    vision = eval_data["vision"]
    audio = eval_data["audio"]
    nlp = eval_data["nlp"]
    pillars = scoring.get("pillars", {})

    lines = [
        "# MyInterview.Assessor — Evaluation Report",
        f"- **Role:** {role} ({difficulty})",
        f"- **Question:** {question}",
        f"- **Predictive Score:** {scoring['predictive_score']}% (Target: {scoring['target_score']}%, Delta: {scoring['delta_vs_target']:+.1f}%)",
        f"- **Readiness:** {scoring['readiness_label']}",
        "",
        "## Pillar Scores",
        f"- Context & STAR Alignment (40%): {pillars.get('context_40', 0)}%",
        f"- Speech & Audio Delivery (25%): {pillars.get('speech_25', 0)}%",
        f"- Visual & Behavioral Presence (25%): {pillars.get('visual_25', 0)}%",
        f"- Environment Quality (10%): {pillars.get('environment_10', 0)}%",
        "",
        "## Behavioral & Speech Metrics",
        f"- Eye Contact Score: {vision.get('eye_contact_score', 0)}% (Face Detection: {vision.get('face_detection_rate', 0)}%)",
        f"- Posture Score: {vision.get('posture_score', 0)}%",
        f"- Lighting Score: {vision.get('lighting_score', 0)}%",
        f"- Speaking Pace: {audio.get('wpm', 0)} WPM | Filler Words: {audio.get('filler_count', 0)}",
        f"- Pitch Variation: {audio.get('pitch_std_hz', 0)} Hz | Audio Clarity: {audio.get('audio_clarity_score', 0)}%",
        "",
        "## Transcript",
        audio.get("transcript", "_No transcript captured._"),
        "",
        "## AI Evaluation & Feedback",
        nlp.get("summary_feedback", ""),
        "",
        "### Strengths",
        *[f"- {s}" for s in nlp.get("strengths", [])],
        "",
        "### Areas for Improvement",
        *[f"- {imp}" for imp in nlp.get("improvements", [])],
    ]
    return "\n".join(lines)


def _run_live_audio_pipeline(
    audio_bytes: bytes,
    role: str,
    difficulty: str,
    question: str,
    camera_buffer: Any = None,
) -> dict[str, Any]:
    with st.status("Scoring live spoken response...", expanded=True) as status:
        st.write("1/4 Transcribing speech & measuring WPM, fillers, pitch, and clarity...")
        tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            tmp_audio.write(audio_bytes)
            tmp_audio.flush()
            tmp_audio.close()
            audio_res = analyze_audio_file(tmp_audio.name)
        finally:
            if Path(tmp_audio.name).exists():
                try:
                    Path(tmp_audio.name).unlink()
                except OSError:
                    pass

        st.write("2/4 Checking visual & lighting signals...")
        if camera_buffer is not None:
            vision_res = _analyze_live_snapshot(camera_buffer)
        else:
            vision_res = {
                "eye_contact_score": 70.0,
                "posture_score": 72.0,
                "lighting_score": 70.0,
                "face_detection_rate": 0.0,
                "samples_analyzed": 0,
                "timeline_df": pd.DataFrame(columns=["time_sec", "eye_contact", "posture", "lighting"]),
            }

        st.write("3/4 Grading answer for role alignment & STAR structure via Gemini...")
        nlp_res = evaluate_transcript_nlp(
            transcript=audio_res.get("transcript", ""),
            role=role,
            difficulty=difficulty,
            question=question,
        )

        st.write("4/4 Computing weighted Predictive Score & saving session...")
        score_res = compute_predictive_score(
            vision_result=vision_res,
            audio_result=audio_res,
            nlp_result=nlp_res,
            role=role,
            difficulty=difficulty,
        )

        session_id = save_session(
            role=role,
            difficulty=difficulty,
            question=question,
            vision_result=vision_res,
            audio_result=audio_res,
            nlp_result=nlp_res,
            score_result=score_res,
        )
        status.update(label=f"Live evaluation complete (Session #{session_id} saved)", state="complete")

    timeline_df = vision_res.get("timeline_df")
    timeline_records = (
        timeline_df.to_dict(orient="records")
        if timeline_df is not None and hasattr(timeline_df, "to_dict")
        else []
    )

    return {
        "session_id": session_id,
        "role": role,
        "difficulty": difficulty,
        "question": question,
        "vision": {k: v for k, v in vision_res.items() if k != "timeline_df"},
        "timeline": timeline_records,
        "audio": audio_res,
        "nlp": nlp_res,
        "scoring": score_res,
    }


def _run_video_pipeline(video_path: Path, role: str, difficulty: str, question: str) -> dict[str, Any]:
    with st.status("Running full multimodal video analysis...", expanded=True) as status:
        st.write("1/4 Analyzing visual cues (eye contact, posture, lighting)...")
        vision_res = analyze_video_visuals(str(video_path), sample_fps=2.0)

        st.write("2/4 Extracting audio, transcribing speech & measuring vocal dynamics...")
        audio_res = analyze_video_audio(str(video_path))

        st.write("3/4 Evaluating STAR structure & role relevance with Gemini...")
        nlp_res = evaluate_transcript_nlp(
            transcript=audio_res.get("transcript", ""),
            role=role,
            difficulty=difficulty,
            question=question,
        )

        st.write("4/4 Computing weighted Predictive Score & saving to history...")
        score_res = compute_predictive_score(
            vision_result=vision_res,
            audio_result=audio_res,
            nlp_result=nlp_res,
            role=role,
            difficulty=difficulty,
        )

        session_id = save_session(
            role=role,
            difficulty=difficulty,
            question=question,
            vision_result=vision_res,
            audio_result=audio_res,
            nlp_result=nlp_res,
            score_result=score_res,
        )
        status.update(label=f"Analysis complete (Session #{session_id} saved)", state="complete")

    timeline_df = vision_res.get("timeline_df")
    timeline_records = (
        timeline_df.to_dict(orient="records")
        if timeline_df is not None and hasattr(timeline_df, "to_dict")
        else []
    )

    return {
        "session_id": session_id,
        "role": role,
        "difficulty": difficulty,
        "question": question,
        "vision": {k: v for k, v in vision_res.items() if k != "timeline_df"},
        "timeline": timeline_records,
        "audio": audio_res,
        "nlp": nlp_res,
        "scoring": score_res,
    }


st.title("MyInterview.Assessor 🎯")
st.caption("Live Mock Interviewer & Predictive Readiness Evaluator")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["1. Live Mock Interview", "2. Upload / Sample Video", "3. Assessment Report", "4. Session History"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("API Key Status")
    st.write({
        "Gemini": bool(os.getenv("GEMINI_API_KEY")),
        "Groq (Whisper)": bool(os.getenv("GROQ_API_KEY")),
    })


# ==============================================================================
# PAGE 1: LIVE MOCK INTERVIEW (MIC + OPTIONAL WEBCAM SNAPSHOT)
# ==============================================================================
if page == "1. Live Mock Interview":
    col_q, col_rec = st.columns([1.05, 1.25], gap="large")

    with col_q:
        st.subheader("Role & Interview Prompt")
        role_options = list(SAMPLE_QUESTIONS.keys())
        default_role_idx = role_options.index(st.session_state.role) if st.session_state.role in role_options else 0
        role = st.selectbox("Target Role", role_options, index=default_role_idx)

        diff_options = ["Junior", "Mid-Level", "Senior", "Lead"]
        default_diff_idx = diff_options.index(st.session_state.difficulty) if st.session_state.difficulty in diff_options else 1
        difficulty = st.select_slider("Experience Level", options=diff_options, value=diff_options[default_diff_idx])

        target = get_target_score(role, difficulty)
        st.metric("Target Readiness Score", f"{target:.1f}%", f"{role} · {difficulty}")

        preset_q = st.selectbox("Sample Questions", SAMPLE_QUESTIONS[role])
        question = st.text_area("Active Interview Question (Editable for Panel)", value=preset_q, height=95)

        st.session_state.role = role
        st.session_state.difficulty = difficulty
        st.session_state.question = question

    with col_rec:
        st.subheader("Live Spoken Response")
        st.caption("Ask the candidate or panel member to answer aloud via the microphone below, then click **Score Live Response**.")

        live_audio = st.audio_input("Record live spoken answer")
        enable_cam = st.checkbox("Include webcam eye-contact & lighting snapshot (optional)")
        cam_snap = st.camera_input("Optional live webcam snapshot") if enable_cam else None

        if st.button(
            "Score Live Response",
            type="primary",
            use_container_width=True,
            disabled=live_audio is None,
        ):
            if live_audio is not None:
                try:
                    result = _run_live_audio_pipeline(
                        audio_bytes=live_audio.getvalue(),
                        role=role,
                        difficulty=difficulty,
                        question=question,
                        camera_buffer=cam_snap,
                    )
                    st.session_state.latest_evaluation = result
                    st.success(
                        f"Predictive Score: **{result['scoring']['predictive_score']}%** "
                        f"({result['scoring']['readiness_label']}) · Pace: **{result['audio']['wpm']} WPM** · "
                        f"Fillers: **{result['audio']['filler_count']}**. View **3. Assessment Report** for full feedback."
                    )
                except Exception as exc:
                    st.error(f"Live evaluation failed: {exc}")


# ==============================================================================
# PAGE 2: UPLOAD OR SAMPLE VIDEO PIPELINE
# ==============================================================================
elif page == "2. Upload / Sample Video":
    col_cfg, col_vid = st.columns([1, 1.3], gap="large")

    with col_cfg:
        st.subheader("Session Context")
        role = st.selectbox("Target Role", list(SAMPLE_QUESTIONS.keys()), key="vid_role")
        difficulty = st.select_slider("Experience Level", options=["Junior", "Mid-Level", "Senior", "Lead"], value="Mid-Level", key="vid_diff")
        question = st.text_area("Interview Question Evaluated", value=SAMPLE_QUESTIONS[role][0], height=90, key="vid_q")

    with col_vid:
        st.subheader("Upload Video or Use Sample Clip")
        uploaded_file = st.file_uploader(
            "Upload an interview recording (.mp4, .mov, .webm, .avi)",
            type=["mp4", "mov", "webm", "avi", "mkv"],
        )

        sample_clip = Path("assets/test_clip.mp4")
        use_sample = False
        if uploaded_file is None and sample_clip.exists():
            use_sample = st.checkbox("Use local sample clip (`assets/test_clip.mp4`)", value=True)

        active_video_path: Path | None = None
        temp_file_obj = None

        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix or ".mp4"
            temp_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file_obj.write(uploaded_file.getbuffer())
            temp_file_obj.flush()
            temp_file_obj.close()
            active_video_path = Path(temp_file_obj.name)
            st.video(str(active_video_path))
        elif use_sample and sample_clip.exists():
            active_video_path = sample_clip
            st.video(str(active_video_path))
        else:
            st.info("Upload a video file or place a clip at `assets/test_clip.mp4` to run analysis.")

        if st.button("Run Full Video Assessment", type="primary", use_container_width=True, disabled=active_video_path is None):
            if active_video_path is not None:
                try:
                    result = _run_video_pipeline(active_video_path, role, difficulty, question)
                    st.session_state.latest_evaluation = result
                    st.success(
                        f"Predictive Score: **{result['scoring']['predictive_score']}%** "
                        f"({result['scoring']['readiness_label']}). Open **3. Assessment Report** to inspect full details."
                    )
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                finally:
                    if temp_file_obj is not None and Path(temp_file_obj.name).exists():
                        try:
                            Path(temp_file_obj.name).unlink()
                        except OSError:
                            pass


# ==============================================================================
# PAGE 3: ASSESSMENT REPORT
# ==============================================================================
elif page == "3. Assessment Report":
    eval_data = st.session_state.latest_evaluation
    if not eval_data:
        past = get_sessions(limit=1)
        if past and past[0].get("report"):
            rep = past[0]["report"]
            eval_data = {
                "session_id": past[0]["id"],
                "role": past[0]["role"],
                "difficulty": past[0]["difficulty"],
                "question": past[0]["question"],
                "vision": rep.get("vision", {}),
                "timeline": rep.get("timeline", []),
                "audio": rep.get("audio", {}),
                "nlp": rep.get("nlp", {}),
                "scoring": rep.get("scoring", {}),
            }

    if not eval_data or not eval_data.get("scoring"):
        st.info("No evaluation available yet. Record a live answer on **1. Live Mock Interview** or analyze a video first.")
    else:
        scoring = eval_data["scoring"]
        vision = eval_data["vision"]
        audio = eval_data["audio"]
        nlp = eval_data["nlp"]
        pillars = scoring.get("pillars", {})

        st.subheader(f"Assessment Report — {eval_data['role']} ({eval_data['difficulty']})")
        st.caption(f"Prompt: {eval_data['question']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predictive Score", f"{scoring['predictive_score']}%", f"{scoring['delta_vs_target']:+.1f}% vs Target")
        c2.metric("Speaking Pace", f"{audio.get('wpm', 0)} WPM", f"{audio.get('filler_count', 0)} filler words")
        c3.metric("Eye Contact", f"{vision.get('eye_contact_score', 0)}%", f"Posture: {vision.get('posture_score', 0)}%")
        c4.metric("Context & Relevance", f"{pillars.get('context_40', 0)}%", scoring.get("readiness_label", ""))

        st.divider()

        st.markdown("#### Weighted Pillar Breakdown")
        p1, p2, p3, p4 = st.columns(4)
        p1.progress(min(100, max(0, int(round(pillars.get("context_40", 0))))), text=f"Context (40%): {pillars.get('context_40', 0)}%")
        p2.progress(min(100, max(0, int(round(pillars.get("speech_25", 0))))), text=f"Speech (25%): {pillars.get('speech_25', 0)}%")
        p3.progress(min(100, max(0, int(round(pillars.get("visual_25", 0))))), text=f"Visual (25%): {pillars.get('visual_25', 0)}%")
        p4.progress(min(100, max(0, int(round(pillars.get("environment_10", 0))))), text=f"Environment (10%): {pillars.get('environment_10', 0)}%")

        tab_vis, tab_nlp, tab_exp = st.tabs(["📈 Delivery & Visuals", "🗣️ Transcript & STAR Feedback", "📥 Export Report"])

        with tab_vis:
            timeline = eval_data.get("timeline", [])
            if len(timeline) > 1:
                t_df = pd.DataFrame(timeline).set_index("time_sec")
                st.line_chart(t_df[["eye_contact", "posture", "lighting"]])

            m1, m2, m3 = st.columns(3)
            m1.metric("Lighting Score", f"{vision.get('lighting_score', 0)}%")
            m2.metric("Audio Clarity", f"{audio.get('audio_clarity_score', 0)}%")
            m3.metric("Pitch Variation", f"{audio.get('pitch_std_hz', 0)} Hz")

            if audio.get("filler_breakdown"):
                st.write("**Filler Words Detected:**", audio["filler_breakdown"])

        with tab_nlp:
            st.markdown("**Summary Evaluation**")
            st.write(nlp.get("summary_feedback", ""))

            star = nlp.get("star_breakdown", {})
            if isinstance(star, dict) and star:
                st.markdown("**STAR Method Breakdown**")
                for key in ["situation", "task", "action", "result"]:
                    info = star.get(key, {})
                    present = "✅" if info.get("present") else "⚠️"
                    st.write(f"- {present} **{key.capitalize()}:** {info.get('note', '')}")

            s_col, i_col = st.columns(2)
            with s_col:
                st.markdown("**Strengths**")
                for item in nlp.get("strengths", []):
                    st.write(f"- {item}")
            with i_col:
                st.markdown("**Actionable Improvements**")
                for item in nlp.get("improvements", []):
                    st.write(f"- {item}")

            with st.expander("View Full Transcript"):
                st.write(audio.get("transcript", ""))

        with tab_exp:
            md_report = _build_markdown_report(eval_data)
            st.download_button(
                label="Download Markdown Report (.md)",
                data=md_report,
                file_name=f"interview_report_session_{eval_data.get('session_id', 'latest')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
            st.download_button(
                label="Download Raw JSON Evaluation (.json)",
                data=json.dumps(eval_data, indent=2, default=str),
                file_name=f"interview_evaluation_session_{eval_data.get('session_id', 'latest')}.json",
                mime="application/json",
                use_container_width=True,
            )


# ==============================================================================
# PAGE 4: SESSION HISTORY
# ==============================================================================
elif page == "4. Session History":
    st.subheader("Saved Interview Sessions")

    sessions = get_sessions(limit=50)
    if not sessions:
        st.info("No saved sessions yet. Complete a live or video assessment to record your first session.")
    else:
        history_df = pd.DataFrame(
            [
                {
                    "ID": s["id"],
                    "Timestamp": s["created_at"],
                    "Role": s["role"],
                    "Level": s["difficulty"],
                    "Predictive %": s["predictive_score"],
                    "Target %": s["target_score"],
                    "WPM": s["wpm"],
                    "Fillers": s["filler_count"],
                    "Eye Contact %": s["eye_contact_score"],
                    "Status": s["readiness_label"],
                }
                for s in sessions
            ]
        )

        chronological_df = history_df.iloc[::-1].set_index("ID")
        st.line_chart(chronological_df[["Predictive %", "Target %"]])
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        st.divider()
        selected_id = st.selectbox("Load a past session into Assessment Report", options=[s["id"] for s in sessions])
        if st.button("Load Selected Session"):
            chosen = next((s for s in sessions if s["id"] == selected_id), None)
            if chosen and chosen.get("report"):
                rep = chosen["report"]
                st.session_state.latest_evaluation = {
                    "session_id": chosen["id"],
                    "role": chosen["role"],
                    "difficulty": chosen["difficulty"],
                    "question": chosen["question"],
                    "vision": rep.get("vision", {}),
                    "timeline": rep.get("timeline", []),
                    "audio": rep.get("audio", {}),
                    "nlp": rep.get("nlp", {}),
                    "scoring": rep.get("scoring", {}),
                }
                st.success(f"Loaded Session #{selected_id}. Switch to **3. Assessment Report** to view.")

        if st.button("Clear All Saved Sessions", type="secondary"):
            clear_sessions()
            st.session_state.latest_evaluation = None
            st.rerun()