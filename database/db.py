import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "interview_sessions.db"


def _get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create the sessions table if it does not already exist."""
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                role TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                question TEXT NOT NULL,
                predictive_score REAL NOT NULL,
                target_score REAL NOT NULL,
                delta_vs_target REAL NOT NULL,
                readiness_label TEXT NOT NULL,
                context_score REAL NOT NULL,
                speech_score REAL NOT NULL,
                visual_score REAL NOT NULL,
                environment_score REAL NOT NULL,
                wpm REAL NOT NULL,
                filler_count INTEGER NOT NULL,
                eye_contact_score REAL NOT NULL,
                posture_score REAL NOT NULL,
                lighting_score REAL NOT NULL,
                audio_clarity_score REAL NOT NULL,
                transcript TEXT NOT NULL,
                report_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_session(
    role: str,
    difficulty: str,
    question: str,
    vision_result: dict[str, Any],
    audio_result: dict[str, Any],
    nlp_result: dict[str, Any],
    score_result: dict[str, Any],
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Persist a completed evaluation session and return its row ID."""
    init_db(db_path)

    timeline_df = vision_result.get("timeline_df")
    timeline_records = (
        timeline_df.to_dict(orient="records")
        if timeline_df is not None and hasattr(timeline_df, "to_dict")
        else []
    )

    report_payload = {
        "vision": {k: v for k, v in vision_result.items() if k != "timeline_df"},
        "timeline": timeline_records,
        "audio": audio_result,
        "nlp": nlp_result,
        "scoring": score_result,
    }

    pillars = score_result.get("pillars", {})
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO sessions (
                created_at, role, difficulty, question,
                predictive_score, target_score, delta_vs_target, readiness_label,
                context_score, speech_score, visual_score, environment_score,
                wpm, filler_count, eye_contact_score, posture_score,
                lighting_score, audio_clarity_score, transcript, report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_str,
                role,
                difficulty,
                question,
                float(score_result.get("predictive_score", 0.0)),
                float(score_result.get("target_score", 80.0)),
                float(score_result.get("delta_vs_target", 0.0)),
                str(score_result.get("readiness_label", "Evaluated")),
                float(pillars.get("context_40", 0.0)),
                float(pillars.get("speech_25", 0.0)),
                float(pillars.get("visual_25", 0.0)),
                float(pillars.get("environment_10", 0.0)),
                float(audio_result.get("wpm", 0.0)),
                int(audio_result.get("filler_count", 0)),
                float(vision_result.get("eye_contact_score", 0.0)),
                float(vision_result.get("posture_score", 0.0)),
                float(vision_result.get("lighting_score", 0.0)),
                float(audio_result.get("audio_clarity_score", 0.0)),
                str(audio_result.get("transcript", "")),
                json.dumps(report_payload),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_sessions(limit: int = 50, db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Fetch recent sessions ordered newest-first."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, role, difficulty, question,
                   predictive_score, target_score, delta_vs_target, readiness_label,
                   context_score, speech_score, visual_score, environment_score,
                   wpm, filler_count, eye_contact_score, posture_score,
                   lighting_score, audio_clarity_score, transcript, report_json
            FROM sessions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["report"] = json.loads(item.pop("report_json"))
        except Exception:
            item["report"] = {}
        results.append(item)
    return results


def clear_sessions(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Delete all saved sessions."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        conn.execute("DELETE FROM sessions")
        conn.commit()