from typing import Any


def get_target_score(role: str, difficulty: str) -> float:
    """Return baseline readiness target score (0-100) by role and difficulty."""
    role_base = {
        "Software Engineer": 80.0,
        "Product Manager": 82.0,
        "Data Analyst": 78.0,
        "Marketing Executive": 76.0,
    }.get(role, 78.0)

    difficulty_offset = {
        "Junior": -3.0,
        "Mid-Level": 0.0,
        "Senior": 5.0,
        "Lead": 8.0,
    }.get(difficulty, 0.0)

    return round(max(60.0, min(95.0, role_base + difficulty_offset)), 1)


def compute_predictive_score(
    vision_result: dict[str, Any],
    audio_result: dict[str, Any],
    nlp_result: dict[str, Any],
    role: str = "Software Engineer",
    difficulty: str = "Mid-Level",
) -> dict[str, Any]:
    """Calculate weighted Predictive Score (40% Context, 25% Speech, 25% Visual, 10% Env)."""
    # 1. Context / NLP pillar (40%)
    content_score = float(nlp_result.get("content_score", 60.0))
    relevance_score = float(nlp_result.get("relevance_score", 60.0))
    context_pillar = round(0.5 * content_score + 0.5 * relevance_score, 1)

    # 2. Speech & Audio delivery pillar (25%)
    speech_pillar = round(float(audio_result.get("speech_delivery_score", 60.0)), 1)

    # 3. Visual & Behavioral pillar (25%)
    eye_score = float(vision_result.get("eye_contact_score", 60.0))
    posture_score = float(vision_result.get("posture_score", 60.0))
    visual_pillar = round(0.55 * eye_score + 0.45 * posture_score, 1)

    # 4. Environment pillar (10%): lighting + audio clarity
    lighting_score = float(vision_result.get("lighting_score", 60.0))
    clarity_score = float(audio_result.get("audio_clarity_score", 60.0))
    environment_pillar = round(0.5 * lighting_score + 0.5 * clarity_score, 1)

    predictive_score = round(
        0.40 * context_pillar
        + 0.25 * speech_pillar
        + 0.25 * visual_pillar
        + 0.10 * environment_pillar,
        1,
    )

    target_score = get_target_score(role, difficulty)
    delta_vs_target = round(predictive_score - target_score, 1)

    if predictive_score >= target_score:
        readiness_label = "Strong Hire / Ready"
    elif predictive_score >= target_score - 8.0:
        readiness_label = "Borderline / Needs Polish"
    else:
        readiness_label = "High Risk / Practice More"

    return {
        "predictive_score": predictive_score,
        "target_score": target_score,
        "delta_vs_target": delta_vs_target,
        "readiness_label": readiness_label,
        "pillars": {
            "context_40": context_pillar,
            "speech_25": speech_pillar,
            "visual_25": visual_pillar,
            "environment_10": environment_pillar,
        },
    }


if __name__ == "__main__":
    sample_vision = {"eye_contact_score": 78.9, "posture_score": 83.1, "lighting_score": 56.9}
    sample_audio = {"speech_delivery_score": 75.3, "audio_clarity_score": 60.8}
    sample_nlp = {"content_score": 80.0, "relevance_score": 78.0}
    print(compute_predictive_score(sample_vision, sample_audio, sample_nlp))