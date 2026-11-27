import html
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from pypdf import PdfReader

from database.db import clear_sessions, get_sessions, init_db, save_session
from modules.audio import analyze_video_audio
from modules.nlp import evaluate_transcript_nlp, generate_dynamic_question
from modules.scoring import compute_predictive_score, get_target_score
from modules.vision import analyze_video_visuals

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
SAMPLE_CLIP_PATH = APP_DIR / "assets" / "test_clip.mp4"

st.set_page_config(
    page_title="MyInterview.Assessor",
    layout="wide",
    page_icon="🎯",
)

# High-Contrast Warm Light / Anthropic-Inspired Theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F7F5F0;
        color: #1A1918;
    }

    [data-testid="stSidebar"] {
        background-color: #EFECE4;
        border-right: 1px solid #E2DDD2;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #1A1918 !important;
    }

    h1, h2, h3, h4, p, label, .stMarkdown {
        color: #1A1918;
    }

    /* Ensure widget labels and metric text are always crisp and visible */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] label {
        color: #3D3A36 !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #1A1918 !important;
        font-weight: 700 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border: 1px solid #E3DDD2 !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
    }

    .app-subtitle {
        color: #5C5750;
        font-size: 0.98rem;
        margin-top: -0.35rem;
        margin-bottom: 1rem;
    }

    .prompt-card {
        background-color: #FAF7F2;
        border: 1px solid #E3DDD2;
        border-left: 4px solid #C96442;
        border-radius: 10px;
        padding: 14px 16px;
        margin: 10px 0 14px 0;
    }

    .prompt-meta {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #C96442;
        margin-bottom: 6px;
    }

    .prompt-text {
        font-size: 1.02rem;
        font-weight: 600;
        color: #1A1918;
        line-height: 1.5;
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #C96442;
        color: #FFFFFF !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #B05334;
        color: #FFFFFF !important;
    }

    div[data-testid="stButton"] > button[kind="secondary"] {
        background-color: #FFFFFF;
        color: #1A1918 !important;
        border: 1px solid #D6CFC2;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

init_db()

DIFFICULTY_LEVELS = ["Intern", "Junior", "Mid-Level", "Senior", "Lead", "Senior Manager"]
ROLE_OPTIONS = ["Software Engineer", "Product Manager", "Data Analyst", "Marketing Executive"]

LIVE_RECORDER_HTML = """
<div style="background:#FAF7F2;border:1px solid #E3DDD2;border-radius:10px;padding:14px;color:#1A1918;font-family:sans-serif;">
  <video id="preview" autoplay playsinline muted style="width:100%;max-height:285px;border-radius:8px;background:#1A1918;"></video>
  <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
    <button id="startBtn" type="button" style="padding:8px 14px;border-radius:8px;border:none;background:#2F6F4E;color:#FFFFFF;font-weight:600;cursor:pointer;">Start Camera & Record</button>
    <button id="stopBtn" type="button" disabled style="padding:8px 14px;border-radius:8px;border:none;background:#B53A2D;color:#FFFFFF;font-weight:600;cursor:pointer;">Stop Recording</button>
    <a id="downloadLink" style="display:none;padding:8px 14px;border-radius:8px;background:#C96442;color:#FFFFFF;text-decoration:none;font-size:13px;font-weight:600;">Download Clip</a>
    <span id="timerText" style="font-size:13px;font-weight:600;color:#5C5750;margin-left:auto;">00:00</span>
  </div>
  <div id="statusText" style="margin-top:8px;font-size:12.5px;color:#5C5750;">Ready for short answers or multi-minute sessions.</div>
</div>

<script>
let mediaRecorder;
let recordedChunks = [];
let streamRef;
let timerInterval;
let elapsedSec = 0;

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const preview = document.getElementById("preview");
const downloadLink = document.getElementById("downloadLink");
const statusText = document.getElementById("statusText");
const timerText = document.getElementById("timerText");

function formatTime(sec) {
  const m = String(Math.floor(sec / 60)).padStart(2, "0");
  const s = String(sec % 60).padStart(2, "0");
  return `${m}:${s}`;
}

startBtn.onclick = async () => {
  try {
    recordedChunks = [];
    elapsedSec = 0;
    timerText.textContent = "00:00";

    streamRef = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    preview.srcObject = streamRef;

    const candidates = ["video/webm;codecs=vp8,opus", "video/webm", "video/mp4"];
    const supported = candidates.find((t) => MediaRecorder.isTypeSupported(t));
    mediaRecorder = supported
      ? new MediaRecorder(streamRef, { mimeType: supported })
      : new MediaRecorder(streamRef);
    const mimeType = mediaRecorder.mimeType || supported || "video/webm";
    const fileExt = mimeType.includes("mp4") ? "mp4" : "webm";

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      clearInterval(timerInterval);
      const blob = new Blob(recordedChunks, { type: mimeType });
      const url = URL.createObjectURL(blob);
      downloadLink.href = url;
      downloadLink.download = "live_interview_session." + fileExt;
      downloadLink.style.display = "inline-block";
      statusText.textContent = "Recording ready — download the clip and select it below to evaluate.";
      if (streamRef) {
        streamRef.getTracks().forEach((t) => t.stop());
      }
    };

    mediaRecorder.start(1000);
    timerInterval = setInterval(() => {
      elapsedSec += 1;
      timerText.textContent = formatTime(elapsedSec);
    }, 1000);

    startBtn.disabled = true;
    stopBtn.disabled = false;
    downloadLink.style.display = "none";
    statusText.textContent = "Recording camera & microphone...";
  } catch (err) {
    statusText.textContent = "Camera/microphone access issue: " + err.message;
  }
};

stopBtn.onclick = () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
};
</script>
"""

if "role" not in st.session_state:
    st.session_state.role = "Software Engineer"
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "Mid-Level"
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "resume_name" not in st.session_state:
    st.session_state.resume_name = ""
if "question" not in st.session_state:
    st.session_state.question = (
        "Upload a resume and click 'Generate Question' for a prompt tailored to your experience."
    )
if "q_number" not in st.session_state:
    st.session_state.q_number = 1
if "latest_evaluation" not in st.session_state:
    st.session_state.latest_evaluation = None
if "q_editor_rev" not in st.session_state:
    # Bumped whenever a new question is generated so the keyed text_area is
    # re-created with the new value instead of keeping its stale widget state.
    st.session_state.q_editor_rev = 0


def _write_upload_to_temp(uploaded: Any, default_suffix: str) -> Path:
    """Persist an uploaded file to a named temp file and return its path."""
    suffix = Path(uploaded.name).suffix or default_suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
    return Path(tmp.name)


def _render_html_component(markup: str, height: int) -> None:
    """Render trusted inline HTML/JS, preferring st.iframe on newer Streamlit."""
    if hasattr(st, "iframe"):
        st.iframe(markup, height=height)
    else:
        components.html(markup, height=height)


def _extract_resume_text(uploaded_resume: Any) -> str:
    """Extract readable plain text from an uploaded PDF or TXT resume."""
    if uploaded_resume is None:
        return ""
    name = (uploaded_resume.name or "").lower()
    raw_bytes = uploaded_resume.getvalue()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        cleaned = "\n".join(pages_text)
    else:
        cleaned = raw_bytes.decode("utf-8", errors="ignore")

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned[:6000]


def _extract_drive_file_id(url_or_id: str) -> str | None:
    """Extract a Google Drive file ID from a share URL or raw ID."""
    text = (url_or_id or "").strip()
    if not text:
        return None
    if re.fullmatch(r"[A-Za-z0-9_-]{15,}", text):
        return text
    patterns = [
        r"/file/d/([A-Za-z0-9_-]+)",
        r"[?&]id=([A-Za-z0-9_-]+)",
        r"/d/([A-Za-z0-9_-]+)",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group(1)
    return None


def _download_drive_video_to_temp(drive_input: str, max_mb: int = 150) -> Path:
    """Download a publicly shared Google Drive video into a temporary .mp4 file."""
    file_id = _extract_drive_file_id(drive_input)
    if not file_id:
        raise ValueError("Please enter a valid Google Drive share link or File ID.")

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    resp = session.get(download_url, stream=True, timeout=30)

    for key, val in resp.cookies.items():
        if key.startswith("download_warning"):
            resp = session.get(
                f"{download_url}&confirm={val}",
                stream=True,
                timeout=60,
            )
            break

    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        # Large files return an HTML "virus scan warning" page with a confirm form.
        page = resp.text
        form = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', page)
        if form:
            params = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', page))
            resp = session.get(html.unescape(form.group(1)), params=params, stream=True, timeout=60)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        raise RuntimeError(
            "Google Drive returned an HTML page instead of a video. "
            "Check that sharing is set to 'Anyone with the link'."
        )

    max_bytes = max_mb * 1024 * 1024
    downloaded = 0
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > max_bytes:
                raise ValueError(f"Video exceeds the {max_mb} MB limit for lightweight processing.")
            tmp.write(chunk)
        tmp.flush()
        tmp.close()
        return Path(tmp.name)
    except Exception:
        tmp.close()
        if Path(tmp.name).exists():
            Path(tmp.name).unlink(missing_ok=True)
        raise


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
        audio.get("transcript") or "_No transcript captured._",
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


def _run_assessment_pipeline(
    video_path: Path,
    role: str,
    difficulty: str,
    question: str,
    resume_text: str = "",
    sample_fps: float = 2.0,
) -> dict[str, Any]:
    with st.status("Running full multimodal assessment...", expanded=True) as status:
        st.write("1/4 Analyzing motion frames (eye contact, posture, lighting timeline)...")
        vision_res = analyze_video_visuals(str(video_path), sample_fps=sample_fps)

        st.write("2/4 Extracting audio, transcribing speech & measuring vocal metrics...")
        audio_res = analyze_video_audio(str(video_path))

        st.write("3/4 Evaluating role alignment, resume consistency & STAR structure...")
        nlp_res = evaluate_transcript_nlp(
            transcript=audio_res.get("transcript", ""),
            role=role,
            difficulty=difficulty,
            question=question,
            resume_text=resume_text or None,
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
        status.update(label=f"Assessment complete (Session #{session_id} saved)", state="complete")

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


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("### MIA Assessor")
    page = st.radio(
        "Workspace",
        ["1. Live Interview", "2. Drive / Clip Eval", "3. Report", "4. History"],
        label_visibility="collapsed",
    )
    st.divider()
    resume_status = st.session_state.resume_name if st.session_state.resume_text else "Not uploaded"
    st.caption(f"Resume: {resume_status}")
    gemini_ok = "Connected" if os.getenv("GEMINI_API_KEY") else "Missing key"
    groq_ok = "Connected" if os.getenv("GROQ_API_KEY") else "Missing key"
    st.caption(f"Gemini: {gemini_ok} · Groq: {groq_ok}")


# ==============================================================================
# PAGE 1: LIVE INTERVIEW
# ==============================================================================
if page == "1. Live Interview":
    st.title("MyInterview.Assessor")
    st.markdown(
        '<div class="app-subtitle">Resume-grounded mock interviews, motion & speech evaluation, and predictive readiness scoring.</div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.0, 1.25], gap="large")

    with left_col:
        with st.container(border=True):
            st.markdown("### 1. Candidate & Prompt Setup")

            resume_file = st.file_uploader(
                "Upload Resume (.pdf or .txt)",
                type=["pdf", "txt"],
                key="live_resume_uploader",
            )
            if resume_file is not None:
                try:
                    extracted_resume = _extract_resume_text(resume_file)
                    st.session_state.resume_text = extracted_resume
                    st.session_state.resume_name = resume_file.name
                    with st.expander("Preview Extracted Resume"):
                        st.text(extracted_resume[:1200] + ("..." if len(extracted_resume) > 1200 else ""))
                except Exception as exc:
                    st.error(f"Could not read resume: {exc}")

            c_role, c_lvl = st.columns(2)
            with c_role:
                default_role_idx = ROLE_OPTIONS.index(st.session_state.role) if st.session_state.role in ROLE_OPTIONS else 0
                role = st.selectbox("Target Role", ROLE_OPTIONS, index=default_role_idx)
            with c_lvl:
                default_diff_idx = (
                    DIFFICULTY_LEVELS.index(st.session_state.difficulty)
                    if st.session_state.difficulty in DIFFICULTY_LEVELS
                    else 2
                )
                difficulty = st.selectbox("Level", DIFFICULTY_LEVELS, index=default_diff_idx)

            target = get_target_score(role, difficulty)
            st.metric("Target Readiness", f"{target:.1f}%", f"{role} · {difficulty}")

            st.markdown(
                f"""
                <div class="prompt-card">
                  <div class="prompt-meta">Question #{st.session_state.q_number} · Resume-Tailored</div>
                  <div class="prompt-text">"{html.escape(st.session_state.question)}"</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            act1, act2 = st.columns(2)
            with act1:
                if st.button("Generate Question", type="primary", use_container_width=True):
                    with st.spinner("Generating question from resume..."):
                        st.session_state.question = generate_dynamic_question(
                            role=role,
                            difficulty=difficulty,
                            resume_text=st.session_state.resume_text or None,
                            question_number=st.session_state.q_number,
                        )
                        st.session_state.q_editor_rev += 1
                        st.rerun()

            with act2:
                prev_eval = st.session_state.latest_evaluation
                has_prev_answer = bool(prev_eval and (prev_eval.get("audio", {}).get("transcript") or "").strip())
                if st.button("Adaptive Follow-Up", use_container_width=True, disabled=not has_prev_answer):
                    with st.spinner("Generating follow-up..."):
                        next_num = st.session_state.q_number + 1
                        st.session_state.question = generate_dynamic_question(
                            role=role,
                            difficulty=difficulty,
                            resume_text=st.session_state.resume_text or None,
                            previous_question=prev_eval.get("question", st.session_state.question),
                            previous_answer=prev_eval.get("audio", {}).get("transcript", ""),
                            question_number=next_num,
                        )
                        st.session_state.q_number = next_num
                        st.session_state.q_editor_rev += 1
                        st.rerun()

            question = st.text_area(
                "Refine Question (Optional)",
                value=st.session_state.question,
                height=80,
                key=f"live_q_editor_{st.session_state.q_editor_rev}",
            )

            sample_fps = st.slider(
                "Frame Sampling (FPS)",
                min_value=0.5,
                max_value=3.0,
                value=1.5,
                step=0.5,
                help="1.0–1.5 FPS is recommended for multi-minute clips.",
            )

            st.session_state.role = role
            st.session_state.difficulty = difficulty
            st.session_state.question = question

    with right_col:
        with st.container(border=True):
            st.markdown("### 2. Record Answer & Evaluate")
            _render_html_component(LIVE_RECORDER_HTML, height=390)

            live_upload = st.file_uploader(
                "Select Recorded Clip (.webm, .mp4, .mov, .mkv)",
                type=["webm", "mp4", "mov", "avi", "mkv"],
                key="live_clip_uploader",
            )

            if live_upload is not None:
                st.video(live_upload.getvalue())

            if st.button(
                "Evaluate Response",
                type="primary",
                use_container_width=True,
                disabled=live_upload is None,
            ):
                if live_upload is not None:
                    active_live_path = _write_upload_to_temp(live_upload, ".webm")
                    try:
                        result = _run_assessment_pipeline(
                            video_path=active_live_path,
                            role=role,
                            difficulty=difficulty,
                            question=question,
                            resume_text=st.session_state.resume_text,
                            sample_fps=sample_fps,
                        )
                        st.session_state.latest_evaluation = result
                        st.success(
                            f"Saved Session **#{result['session_id']}** — "
                            f"Score: **{result['scoring']['predictive_score']}%** ({result['scoring']['readiness_label']})"
                        )
                    except Exception as exc:
                        st.error(f"Assessment failed: {exc}")
                    finally:
                        active_live_path.unlink(missing_ok=True)


# ==============================================================================
# PAGE 2: DRIVE / CLIP EVALUATION
# ==============================================================================
elif page == "2. Drive / Clip Eval":
    st.title("Drive & Clip Evaluation")
    st.markdown(
        '<div class="app-subtitle">Assess an interview clip from a Google Drive link or local video file.</div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.0, 1.25], gap="large")

    with left_col:
        with st.container(border=True):
            st.markdown("### Session Context")

            vid_resume = st.file_uploader(
                "Resume (.pdf or .txt, optional)",
                type=["pdf", "txt"],
                key="vid_resume_uploader",
            )
            if vid_resume is not None:
                try:
                    st.session_state.resume_text = _extract_resume_text(vid_resume)
                    st.session_state.resume_name = vid_resume.name
                except Exception as exc:
                    st.error(f"Could not read resume: {exc}")

            role = st.selectbox("Target Role", ROLE_OPTIONS, key="vid_role")
            difficulty = st.selectbox("Level", DIFFICULTY_LEVELS, index=2, key="vid_diff")

            if st.button("Generate Resume Question", key="vid_gen_q", use_container_width=True):
                with st.spinner("Generating question..."):
                    st.session_state["vid_q"] = generate_dynamic_question(
                        role=role,
                        difficulty=difficulty,
                        resume_text=st.session_state.resume_text or None,
                        question_number=1,
                    )

            if "vid_q" not in st.session_state:
                st.session_state["vid_q"] = (
                    f"Walk me through a key project on your resume relevant to a {difficulty} {role} role."
                )
            question = st.text_area("Evaluated Question", height=85, key="vid_q")
            sample_fps = st.slider("Frame Sampling (FPS)", min_value=0.5, max_value=3.0, value=2.0, step=0.5, key="vid_fps")

    with right_col:
        with st.container(border=True):
            st.markdown("### Video Source")

            source_mode = st.radio(
                "Source Type",
                ["Google Drive Link", "Local File / Sample"],
                horizontal=True,
            )

            drive_link = ""
            uploaded_file = None
            use_sample = False

            if source_mode == "Google Drive Link":
                drive_link = st.text_input(
                    "Google Drive Share Link or File ID",
                    placeholder="https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing",
                )
                st.caption("Sharing must be set to **Anyone with the link**.")
            else:
                uploaded_file = st.file_uploader(
                    "Upload Clip (.mp4, .mov, .webm, .avi)",
                    type=["mp4", "mov", "webm", "avi", "mkv"],
                    key="batch_clip_uploader",
                )
                if uploaded_file is None and SAMPLE_CLIP_PATH.exists():
                    use_sample = st.checkbox("Use local sample (`assets/test_clip.mp4`)", value=True)

            active_video_path: Path | None = None

            if uploaded_file is not None:
                st.video(uploaded_file.getvalue())
            elif use_sample and SAMPLE_CLIP_PATH.exists():
                active_video_path = SAMPLE_CLIP_PATH
                st.video(str(active_video_path))

            has_drive_input = source_mode == "Google Drive Link" and bool(drive_link.strip())
            can_run = has_drive_input or uploaded_file is not None or active_video_path is not None

            if st.button("Run Full Assessment", type="primary", use_container_width=True, disabled=not can_run):
                downloaded_drive_path: Path | None = None
                uploaded_tmp_path: Path | None = None
                try:
                    if uploaded_file is not None:
                        uploaded_tmp_path = _write_upload_to_temp(uploaded_file, ".mp4")
                        active_video_path = uploaded_tmp_path
                    elif source_mode == "Google Drive Link":
                        with st.spinner("Fetching video from Google Drive..."):
                            downloaded_drive_path = _download_drive_video_to_temp(drive_link)
                        active_video_path = downloaded_drive_path

                    if active_video_path is not None:
                        result = _run_assessment_pipeline(
                            video_path=active_video_path,
                            role=role,
                            difficulty=difficulty,
                            question=question,
                            resume_text=st.session_state.resume_text,
                            sample_fps=sample_fps,
                        )
                        st.session_state.latest_evaluation = result
                        st.success(
                            f"Score: **{result['scoring']['predictive_score']}%** ({result['scoring']['readiness_label']}). "
                            "Open **3. Report** to inspect full results."
                        )
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                finally:
                    for tmp_p in (uploaded_tmp_path, downloaded_drive_path):
                        if tmp_p is not None and tmp_p.exists():
                            try:
                                tmp_p.unlink()
                            except OSError:
                                pass


# ==============================================================================
# PAGE 3: ASSESSMENT REPORT
# ==============================================================================
elif page == "3. Report":
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

    st.title("Assessment Report")
    if not eval_data or not eval_data.get("scoring"):
        with st.container(border=True):
            st.info("No evaluation available yet. Complete a session on **1. Live Interview** or **2. Drive / Clip Eval**.")
    else:
        scoring = eval_data["scoring"]
        vision = eval_data["vision"]
        audio = eval_data["audio"]
        nlp = eval_data["nlp"]
        pillars = scoring.get("pillars", {})

        st.markdown(
            f'<div class="app-subtitle">{html.escape(str(eval_data["role"]))} ({html.escape(str(eval_data["difficulty"]))}) · '
            f'Prompt: "{html.escape(str(eval_data["question"]))}"</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predictive Score", f"{scoring['predictive_score']}%", f"{scoring['delta_vs_target']:+.1f}% vs Target")
        c2.metric("Speaking Pace", f"{audio.get('wpm', 0)} WPM", f"{audio.get('filler_count', 0)} fillers")
        c3.metric("Eye Contact", f"{vision.get('eye_contact_score', 0)}%", f"Posture: {vision.get('posture_score', 0)}%")
        c4.metric("Context & Relevance", f"{pillars.get('context_40', 0)}%", scoring.get("readiness_label", ""))

        with st.container(border=True):
            st.markdown("#### Weighted Pillar Breakdown")
            p1, p2, p3, p4 = st.columns(4)
            p1.progress(min(100, max(0, int(round(pillars.get("context_40", 0))))), text=f"Context (40%): {pillars.get('context_40', 0)}%")
            p2.progress(min(100, max(0, int(round(pillars.get("speech_25", 0))))), text=f"Speech (25%): {pillars.get('speech_25', 0)}%")
            p3.progress(min(100, max(0, int(round(pillars.get("visual_25", 0))))), text=f"Visual (25%): {pillars.get('visual_25', 0)}%")
            p4.progress(min(100, max(0, int(round(pillars.get("environment_10", 0))))), text=f"Environment (10%): {pillars.get('environment_10', 0)}%")

        tab_vis, tab_nlp, tab_exp = st.tabs(["Delivery & Visuals", "Transcript & STAR", "Export"])

        with tab_vis:
            with st.container(border=True):
                timeline = eval_data.get("timeline", [])
                if len(timeline) > 1:
                    t_df = pd.DataFrame(timeline).set_index("time_sec")
                    st.line_chart(t_df[["eye_contact", "posture", "lighting"]])
                else:
                    st.info("Not enough sampled frames for a multi-point timeline chart.")

                m1, m2, m3 = st.columns(3)
                m1.metric("Lighting Score", f"{vision.get('lighting_score', 0)}%")
                m2.metric("Audio Clarity", f"{audio.get('audio_clarity_score', 0)}%")
                m3.metric("Pitch Variation", f"{audio.get('pitch_std_hz', 0)} Hz")

                if audio.get("filler_breakdown"):
                    st.write("**Filler Words Detected:**", audio["filler_breakdown"])

        with tab_nlp:
            with st.container(border=True):
                st.markdown("**Summary Evaluation**")
                st.write(nlp.get("summary_feedback", ""))

                star = nlp.get("star_breakdown", {})
                if isinstance(star, dict) and star:
                    st.markdown("**STAR Breakdown**")
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
                    st.markdown("**Improvements**")
                    for item in nlp.get("improvements", []):
                        st.write(f"- {item}")

                with st.expander("View Full Transcript"):
                    st.write(audio.get("transcript", ""))

        with tab_exp:
            with st.container(border=True):
                md_report = _build_markdown_report(eval_data)
                st.download_button(
                    label="Download Markdown Report (.md)",
                    data=md_report,
                    file_name=f"interview_report_session_{eval_data.get('session_id', 'latest')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
                st.download_button(
                    label="Download Raw JSON (.json)",
                    data=json.dumps(eval_data, indent=2, default=str),
                    file_name=f"interview_evaluation_session_{eval_data.get('session_id', 'latest')}.json",
                    mime="application/json",
                    use_container_width=True,
                )


# ==============================================================================
# PAGE 4: SESSION HISTORY
# ==============================================================================
elif page == "4. History":
    st.title("Session History")
    st.markdown(
        '<div class="app-subtitle">Review past mock assessments and track readiness over time.</div>',
        unsafe_allow_html=True,
    )

    sessions = get_sessions(limit=50)
    with st.container(border=True):
        if not sessions:
            st.info("No saved sessions yet.")
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
            selected_id = st.selectbox("Load Session into Report", options=[s["id"] for s in sessions])
            b1, b2 = st.columns([1, 1])
            with b1:
                if st.button("Load Selected Session", type="primary", use_container_width=True):
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
                        st.success(f"Loaded Session #{selected_id}. Open **3. Report** to inspect.")
            with b2:
                if st.button("Clear All Sessions", type="secondary", use_container_width=True):
                    clear_sessions()
                    st.session_state.latest_evaluation = None
                    st.rerun()
                    st.rerun()