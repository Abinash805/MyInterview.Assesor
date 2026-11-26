import json
import os
import re
from typing import Any
from google import genai
from google.genai import types

GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
]


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


def _generate_with_fallback(
    client: genai.Client,
    contents: str,
    config: types.GenerateContentConfig,
) -> Any:
    """Try candidate Gemini models in order until one succeeds."""
    last_err: Exception | None = None
    for model_name in GEMINI_MODELS:
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")


def evaluate_transcript_nlp(
    transcript: str,
    role: str = "Software Engineer",
    difficulty: str = "Mid-Level",
    question: str = "Tell me about a time you solved a difficult problem or led a project.",
    resume_text: str | None = None,
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

    cleaned_resume = (resume_text or "").strip()[:4500]
    resume_section = (
        f'\nCandidate Resume Context:\n"""\n{cleaned_resume}\n"""\n'
        if cleaned_resume
        else ""
    )

    prompt = f"""
You are an objective technical and behavioral interview evaluator.
Evaluate the candidate's response for a {difficulty} {role} role.
{resume_section}
Interview Question:
"{question}"

Candidate Transcript:
\"\"\"{cleaned_transcript}\"\"\"

Return ONLY a valid JSON object matching this exact schema:
{{
  "content_score": <number 0 to 100 reflecting clarity, structure, and depth>,
  "relevance_score": <number 0 to 100 reflecting alignment with {role} at {difficulty} level and resume claims if provided>,
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

    response = _generate_with_fallback(
        client=client,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

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


def generate_dynamic_question(
    role: str = "Software Engineer",
    difficulty: str = "Mid-Level",
    resume_text: str | None = None,
    previous_question: str | None = None,
    previous_answer: str | None = None,
    question_number: int = 1,
) -> str:
    """Generate an interview question grounded in the candidate's resume, role, level, and prior answer."""
    fallback_q = (
        f"Walk me through a project or experience on your resume relevant to a {difficulty} {role} "
        f"role, including the key technical or strategic decisions you made and the measurable outcome."
    )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return fallback_q

    client = genai.Client(api_key=api_key)

    cleaned_resume = (resume_text or "").strip()[:4500]
    resume_block = (
        f'Candidate Resume:\n"""\n{cleaned_resume}\n"""\n'
        if cleaned_resume
        else "Candidate Resume: Not provided (calibrate to role and experience level).\n"
    )

    if previous_question and previous_answer and len(previous_answer.strip()) >= 10:
        task_block = f"""
Previous Question: "{previous_question}"
Candidate's Previous Answer:
\"\"\"{previous_answer.strip()}\"\"\"

Generate question #{question_number} as a realistic, targeted follow-up question.
- Reference specific projects, tools, responsibilities, or claims from the candidate's resume and their previous answer.
- Probe deeper into trade-offs, architecture/execution choices, ownership, or missing metrics/impact.
- Match the depth and expectations of a {difficulty} {role}.
"""
    else:
        task_block = f"""
Generate interview question #{question_number} for a {difficulty} {role} candidate.
- Directly reference a concrete project, role, skill, or experience from the candidate's resume when available.
- Calibrate scope and depth strictly to {difficulty} level (e.g., foundational implementation/debugging for Intern/Junior, system ownership/trade-offs for Mid/Senior, org/strategy/leadership for Lead/Senior Manager).
"""

    prompt = f"""
You are a technical and behavioral interviewer hiring for a {difficulty} {role} role.
{resume_block}
{task_block}
Return ONLY the single interview question as plain text—no bullet points, no quotes around the whole string, and no extra commentary.
""".strip()

    try:
        resp = _generate_with_fallback(
            client=client,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.5),
        )
        generated = (resp.text or "").strip().strip('"')
        if generated:
            return generated
    except Exception:
        pass

    return fallback_q


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