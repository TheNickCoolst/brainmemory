"""Turn network recall into an answer.

The answer is composed from neurons that actually fired. An optional LLM may
only phrase those concepts — it is not allowed to be the memory.
"""

from __future__ import annotations

import os
from typing import Any


def render_answer(question: str, recall: dict[str, Any], language: str = "de") -> str:
    cues = list(recall.get("cue_concepts") or [])
    assoc = list(recall.get("associated_concepts") or [])
    conf = float(recall.get("confidence") or 0.0)
    n_active = int(recall.get("engram_size") or 0)
    q = question.strip()
    if language == "de":
        if not cues:
            return f"Aus „{q}“ wurden keine Begriffe erkannt, die das Netz kennt."
        if not assoc or conf < 0.12:
            return (
                f"Cue {', '.join(cues[:5])} aktiviert {n_active} Neuronen, "
                f"aber es gibt noch keine stabile Ergänzung (Konfidenz {conf:.2f})."
            )
        return (
            f"Frage „{q}“ legt {', '.join(cues[:4])} an. "
            f"Das Netz vervollständigt: {', '.join(assoc[:8])}. "
            f"Konfidenz {conf:.2f}, {n_active} aktive Neuronen."
        )
    if not cues:
        return f"No known concepts in: {q}"
    if not assoc or conf < 0.12:
        return f"Cue {', '.join(cues[:5])} fired {n_active} neurons without a stable completion (conf {conf:.2f})."
    return (
        f"Query “{q}” clamped {', '.join(cues[:4])}. "
        f"Network completed: {', '.join(assoc[:8])}. "
        f"confidence {conf:.2f}, {n_active} active neurons."
    )


def maybe_verbalize(question: str, recall: dict[str, Any], draft: str) -> str:
    if os.environ.get("BRAINMEMORY_SPEAK") != "1":
        return draft
    api_key = os.environ.get("XAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return draft
    concepts = list(recall.get("recalled_concepts") or [])
    try:
        import json
        import urllib.request

        prompt = (
            "Formuliere EINE kurze Antwort auf Deutsch. "
            "Du darfst NUR diese vom Netzwerk erinnerten Begriffe verwenden, nichts erfinden:\n"
            + ", ".join(concepts[:16])
            + f"\nFrage: {question}"
        )
        url = os.environ.get("BRAINMEMORY_LLM_URL", "https://api.x.ai/v1/chat/completions")
        model = os.environ.get("BRAINMEMORY_LLM_MODEL", "grok-4")
        body = json.dumps(
            {
                "model": model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "Nur Netzwerkbegriffe verbalisieren, kein Weltwissen."},
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"].strip()
        return text or draft
    except Exception:
        return draft
