================================================================================
MyInterview.Assessor — Complete Project README, Setup Guide & Git Handoff
================================================================================

1. OVERVIEW
MyInterview.Assessor is an AI-powered mock interview trainer and predictive
readiness evaluator built with Streamlit. It evaluates spoken and video
interview responses across four weighted pillars and tracks session history
over time using SQLite:
- Context & STAR Alignment (40%)
- Speech & Audio Delivery (25%)
- Visual & Behavioral Presence (25%)
- Environment Quality (10%)

--------------------------------------------------------------------------------
2. CORE CAPABILITIES
- Live Mock Interviewer (1. Live Mock Interview):
  Select a target role and experience level, choose or edit an interview
  question, record a live spoken answer via browser microphone (st.audio_input),
  optionally capture a live webcam snapshot (st.camera_input) for eye-contact
  and lighting checks, and score the response in real time.

- Video Upload & Sample Clip Pipeline (2. Upload / Sample Video):
  Upload .mp4, .mov, .webm, .avi, or .mkv interview recordings—or use a local
  sample clip (assets/test_clip.mp4)—to run full multimodal analysis.

- Visual & Behavioral Analysis (modules/vision.py):
  Samples video frames (default 2.0 FPS) using OpenCV and MediaPipe Tasks
  (FaceLandmarker / PoseLandmarker, with Haar cascade fallback) to estimate
  eye contact %, posture alignment %, lighting quality %, and frame-by-frame
  timeline trends.

- Speech & Audio Diagnostics (modules/audio.py):
  Extracts audio via ffmpeg, transcribes speech using Groq Whisper
  (whisper-large-v3-turbo), computes Words Per Minute (WPM) and filler word
  counts/breakdown, and measures pitch variation (Hz) and audio clarity using
  librosa.

- Context & STAR Evaluation (modules/nlp.py):
  Evaluates transcripts against the target role, difficulty level, and STAR
  method (Situation, Task, Action, Result) using the Gemini API
  (gemini-3.8-flash with gemini-3.5-flash-lite fallback) to return structured
  JSON feedback, matched keywords, strengths, and actionable improvements.

- Weighted Predictive Scoring (modules/scoring.py):
  Combines Context (40%), Speech (25%), Visual (25%), and Environment (10%)
  into a Predictive Score and compares it against role- and difficulty-adjusted
  target scores.

- Assessment Report & Session History (3. Assessment Report, 4. Session History,
  database/db.py):
  Displays pillar progress bars, behavioral timeline charts, STAR feedback,
  full transcripts, Markdown/JSON report downloads, and SQLite-backed session
  history with score trend charts.

--------------------------------------------------------------------------------
3. PROJECT STRUCTURE
MyInterview_Assessor/ (or MIA/)
├── app.py                  # Main Streamlit UI, live mic/video workflows, report & history
├── requirements.txt        # Pinned Python dependencies
├── packages.txt            # Linux system packages (ffmpeg, libgl1, libglib2.0-0)
├── .gitignore              # Keeps secrets, venv, media, and local DB out of Git
├── .env                    # Local API keys (do NOT commit to Git)
├── interview_sessions.db   # Auto-created local SQLite database (ignored by Git)
├── assets/
│   └── test_clip.mp4       # Optional sample video clip for testing
├── database/
│   ├── __init__.py
│   └── db.py               # SQLite initialization, save_session(), get_sessions(), clear_sessions()
└── modules/
    ├── __init__.py
    ├── vision.py           # Eye contact, posture, lighting & timeline analysis
    ├── audio.py            # Audio extraction, Whisper transcription & vocal metrics
    ├── nlp.py              # Gemini STAR & role-relevance evaluation
    └── scoring.py          # Target score & weighted Predictive Score calculator

--------------------------------------------------------------------------------
4. PREREQUISITES & LOCAL SETUP
Step 1: System Requirements
- Python 3.10 to 3.13
- ffmpeg installed and available on your system PATH (required for audio
  extraction/processing; on Windows you can install via:
  winget install Gyan.FFmpeg
  and restart your terminal).

Step 2: Clone Repository & Create Virtual Environment
  git clone https://github.com/Abinash805/MyInterview.Assesor.git
  cd MyInterview.Assesor

  python -m venv venv
  # Windows (PowerShell):
  .\venv\Scripts\Activate.ps1
  # macOS / Linux:
  # source venv/bin/activate

Step 3: Install Python Dependencies
  pip install -r requirements.txt
  (If st.audio_input is missing on an older Streamlit version, run:
   pip install --upgrade streamlit)

Step 4: Obtain API Keys & Create .env
1. Gemini API Key (GEMINI_API_KEY):
   Create a key at https://aistudio.google.com/ (API Keys section).
2. Groq API Key (GROQ_API_KEY):
   Create a key at https://console.groq.com/ (API Keys section).

Create a file named .env in the project root folder with:
  GEMINI_API_KEY=your_gemini_api_key_here
  GROQ_API_KEY=your_groq_api_key_here

SECURITY NOTE: .env is listed in .gitignore so your secret keys are never
committed to GitHub. Share keys privately with teammates so they can create
their own local .env file.

--------------------------------------------------------------------------------
5. RUNNING THE APPLICATION & DEMO WORKFLOW
Start Streamlit from the project root with your virtual environment activated:
  streamlit run app.py

Demo Workflow 1 — Live Spoken Mock Interview:
1. Open "1. Live Mock Interview" in the sidebar.
2. Select Target Role (e.g., Software Engineer), Experience Level, and a preset
   or custom interview question.
3. Click the microphone in "Record live spoken answer", speak for 15–45 seconds,
   stop recording, and optionally capture a webcam snapshot.
4. Click "Score Live Response".

Demo Workflow 2 — Video File / Sample Clip Assessment:
1. Open "2. Upload / Sample Video".
2. Upload a video file (.mp4, .mov, .webm, .avi, .mkv) or check
   "Use local sample clip (assets/test_clip.mp4)" if present.
3. Click "Run Full Video Assessment".

Viewing & Exporting Results:
- Open "3. Assessment Report" to inspect the Predictive Score, pillar breakdown,
  timeline chart, STAR breakdown, strengths/improvements, and full transcript,
  and download Markdown (.md) or JSON (.json) reports.
- Open "4. Session History" to view past sessions, score trend charts, or
  reload an earlier session.

Optional CLI Module Checks:
  python modules/vision.py assets/test_clip.mp4
  python modules/audio.py assets/test_clip.mp4
  python modules/nlp.py
  python modules/scoring.py

--------------------------------------------------------------------------------
6. TROUBLESHOOTING & CLOUD DEPLOYMENT NOTES
- MediaPipe model downloads: On first run, modules/vision.py downloads .task
  model files into assets/mp_models/ and falls back to OpenCV Haar face
  detection if offline.
- Browser permissions: Allow microphone (and camera, if enabled) permissions
  when prompted during live sessions.
- Cloud deployment (Streamlit Community Cloud / Hugging Face Spaces):
  Use requirements.txt (with opencv-python-headless) and packages.txt
  (ffmpeg, libgl1, libglib2.0-0), and add GEMINI_API_KEY and GROQ_API_KEY in
  your hosting platform's Secrets settings.

================================================================================