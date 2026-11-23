import json
import os
import re
from typing import Any
from google import genai
from google.genai import types


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse JSON cleanly even if wrapped in markdown fences."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Model did not return valid JSON: {raw_text[:300]}")


def evaluate_transcript_nlp(
    transcript: str,
    role: str = "Software Engineer",
    difficulty: str = "Mid-Level",
    question: str = "Tell me about a time you solved a difficult problem or led a project.",
) -> dict[str, Any]:
    """Evaluate interview transcript for STAR structure, role relevance, and feedback."""
    cleaned_transcript = (transcript or "").strip()
    if len(cleaned_transcript) < 10:
        return {
            "content_score": 15.0,
            "relevance_score": 15.0,
            "star_breakdown": {
                "situation": {"present": False, "note": "Insufficient speech captured."},
                "task": {"present": False, "note": "Insufficient speech captured."},
                "action": {"present": False, "note": "Insufficient speech captured."},
                "result": {"present": False, "note": "Insufficient speech captured."},
            },
            "keywords_matched": [],
            "strengths": ["Attempted response"],
            "improvements": ["Speak for at least 30–60 seconds with clear examples and measurable outcomes."],
            "summary_feedback": "Transcript was too short for a full STAR and role-relevance evaluation.",
        }

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set in environment or .env")

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an objective technical and behavioral interview evaluator.
Evaluate the candidate's response for a {difficulty} {role} role.

Interview Question:
"{question}"

Candidate Transcript:
\"\"\"{cleaned_transcript}\"\"\"

Return ONLY a valid JSON object matching this exact schema:
{{
  "content_score": <number 0 to 100 reflecting clarity, structure, and depth>,
  "relevance_score": <number 0 to 100 reflecting alignment with {role} at {difficulty} level>,
  "star_breakdown": {{
    "situation": {{"present": <bool>, "note": "<brief 1-sentence observation>"}},
    "task": {{"present": <bool>, "note": "<brief 1-sentence observation>"}},
    "action": {{"present": <bool>, "note": "<brief 1-sentence observation>"}},
    "result": {{"present": <bool>, "note": "<brief 1-sentence observation>"}}
  }},
  "keywords_matched": ["<keyword1>", "<keyword2>"],
  "strengths": ["<strength 1>", "<strength 2>"],
  "improvements": ["<actionable improvement 1>", "<actionable improvement 2>"],
  "summary_feedback": "<concise 2-sentence overall evaluation>"
}}
""".strip()

    models_to_try = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    response = None
    last_err: Exception | None = None

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            break
        except Exception as exc:
            last_err = exc

    if response is None:
        raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")

    data = _extract_json_object(response.text or "{}")

    content_score = round(max(0.0, min(100.0, float(data.get("content_score", 65.0)))), 1)
    relevance_score = round(max(0.0, min(100.0, float(data.get("relevance_score", 65.0)))), 1)

    return {
        "content_score": content_score,
        "relevance_score": relevance_score,
        "star_breakdown": data.get("star_breakdown", {}),
        "keywords_matched": data.get("keywords_matched", []),
        "strengths": data.get("strengths", []),
        "improvements": data.get("improvements", []),
        "summary_feedback": str(data.get("summary_feedback", "")).strip(),
    }


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    sample_text = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else (
            "In my previous internship, our API response times spiked during peak traffic. "
            "I was tasked with diagnosing the bottleneck, profiled slow SQL queries, added "
            "composite indexes and caching, and cut p95 latency from 900ms to 180ms."
        )
    )
    out = evaluate_transcript_nlp(sample_text, role="Software Engineer", difficulty="Mid-Level")
    print(json.dumps(out, indent=2))