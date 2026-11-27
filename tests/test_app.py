"""Regression tests for MyInterview.Assessor. Run with: python -m pytest tests"""
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from database import db  # noqa: E402
from modules import nlp, scoring, vision  # noqa: E402


# ---------------------------------------------------------------- nlp parsing
def test_extract_json_handles_fences_and_wrapping():
    assert nlp._extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert nlp._extract_json_object('Here you go: {"a": 2} thanks') == {"a": 2}
    assert nlp._extract_json_object('[{"a": 3}]') == {"a": 3}
    with pytest.raises(ValueError):
        nlp._extract_json_object('"just a string"')


@pytest.mark.parametrize(
    "raw,expected",
    [(None, 65.0), ("82", 82.0), ("n/a", 65.0), (150, 100.0), (-3, 0.0), (float("nan"), 65.0)],
)
def test_clamp_score(raw, expected):
    assert nlp._clamp_score(raw) == expected


def test_evaluate_transcript_tolerates_null_scores(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    fake_resp = SimpleNamespace(
        text='{"content_score": null, "relevance_score": "71", "strengths": "Clear", '
        '"star_breakdown": {"Situation": true}}'
    )
    with mock.patch.object(nlp.genai, "Client"), mock.patch.object(
        nlp, "_generate_with_fallback", return_value=fake_resp
    ):
        out = nlp.evaluate_transcript_nlp("I profiled the slow queries and added an index.")
    assert out["content_score"] == 65.0
    assert out["relevance_score"] == 71.0
    assert out["strengths"] == ["Clear"]
    assert out["star_breakdown"]["situation"]["present"] is True


# --------------------------------------------------------------------- vision
def test_posture_respects_zero_visibility():
    lm = [SimpleNamespace(x=0.5, y=0.5, visibility=0.9) for _ in range(13)]
    lm[11] = SimpleNamespace(x=0.3, y=0.5, visibility=0.0)
    lm[12] = SimpleNamespace(x=0.7, y=0.5, visibility=0.0)
    assert vision._compute_posture_from_landmarks(lm) == 60.0  # treated as not visible


# -------------------------------------------------------------------- scoring
def test_predictive_score_weights():
    res = scoring.compute_predictive_score(
        {"eye_contact_score": 100, "posture_score": 100, "lighting_score": 100},
        {"speech_delivery_score": 100, "audio_clarity_score": 100},
        {"content_score": 100, "relevance_score": 100},
    )
    assert res["predictive_score"] == 100.0
    assert res["readiness_label"] == "Strong Hire / Ready"


# ------------------------------------------------------------------- database
def test_db_roundtrip_and_connections_closed(tmp_path):
    path = tmp_path / "t.db"
    opened = []
    real_connect = sqlite3.connect

    def tracking_connect(*a, **kw):
        conn = real_connect(*a, **kw)
        opened.append(conn)
        return conn

    with mock.patch.object(db.sqlite3, "connect", tracking_connect):
        sid = db.save_session(
            "Software Engineer", "Mid-Level", "q",
            {"eye_contact_score": 50.0}, {"transcript": "hi", "wpm": 120.0}, {}, {"predictive_score": 70.0},
            db_path=path,
        )
        rows = db.get_sessions(db_path=path)
        db.clear_sessions(db_path=path)

    assert sid == 1 and rows[0]["transcript"] == "hi"
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")  # closed


# ------------------------------------------------------------------------ app
@pytest.fixture
def app_test(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    real_get_connection = db._get_connection
    monkeypatch.setattr(db, "_get_connection", lambda db_path=None: real_get_connection(tmp_path / "app.db"))
    monkeypatch.setattr(nlp, "generate_dynamic_question", lambda **kw: "Generated <b>question</b>?")
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)


def test_generate_question_updates_live_editor(app_test):
    at = app_test.run()
    assert not at.exception
    at.button[0].click().run()
    assert at.session_state.question == "Generated <b>question</b>?"
    assert at.text_area[0].value == "Generated <b>question</b>?"
    # User/LLM text must be escaped in the HTML prompt card.
    assert any("&lt;b&gt;question&lt;/b&gt;" in m.value for m in at.markdown)


def test_generate_question_updates_clip_eval_editor(app_test):
    at = app_test.run()
    at.sidebar.radio[0].set_value("2. Drive / Clip Eval").run()
    at.button(key="vid_gen_q").click().run()
    assert at.text_area(key="vid_q").value == "Generated <b>question</b>?"
    assert not at.exception


def test_history_and_report_pages_render(app_test):
    at = app_test.run()
    for page in ("3. Report", "4. History"):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception
