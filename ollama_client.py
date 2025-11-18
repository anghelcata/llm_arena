# ollama_client.py
# Minimal HTTP client for Ollama (/api/generate), non-stream by default.
from __future__ import annotations
import json
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from config import SETTINGS

@dataclass
class GenerateResult:
    text: str
    latency_ms: Optional[int] = None
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None

def _duration_to_ms(obj: Dict[str, Any]) -> Optional[int]:
    for k in ("total_duration", "eval_duration", "prompt_eval_duration", "load_duration"):
        if k in obj and isinstance(obj[k], (int, float)):
            try:
                return int(round(float(obj[k]) / 1000.0))  # microseconds -> ms
            except Exception:
                continue
    return None

def generate(
    *,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_ctx: int = 8192,
    num_predict: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    seed: Optional[int] = None,
    extra_options: Optional[Dict[str, Any]] = None,
) -> GenerateResult:
    url = SETTINGS.ollama_base_url.rstrip("/") + "/api/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": num_ctx,
        },
    }
    if num_predict is not None:
        payload["options"]["num_predict"] = int(num_predict)
    if stop is not None:
        payload["options"]["stop"] = [stop] if isinstance(stop, str) else list(stop)
    if seed is not None:
        payload["options"]["seed"] = int(seed)
    if extra_options:
        payload["options"].update(extra_options)

    resp = requests.post(url, json=payload, timeout=SETTINGS.timeout_s)
    # Some Ollama builds may still stream; handle ndjson fallback
    if "application/x-ndjson" in resp.headers.get("Content-Type", "") or resp.headers.get("Transfer-Encoding") == "chunked":
        # simple streaming parse
        text_parts: List[str] = []
        final_packet: Dict[str, Any] = {}
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "response" in obj and obj["response"]:
                text_parts.append(obj["response"])
            if obj.get("done"):
                final_packet = obj
                break
        text = "".join(text_parts)
        return GenerateResult(
            text=text,
            latency_ms=_duration_to_ms(final_packet),
            prompt_tokens=final_packet.get("prompt_eval_count"),
            output_tokens=final_packet.get("eval_count"),
            raw=final_packet or None,
        )

    resp.raise_for_status()
    data = resp.json()
    return GenerateResult(
        text=(data.get("response", "") or "").strip(),
        latency_ms=_duration_to_ms(data),
        prompt_tokens=data.get("prompt_eval_count"),
        output_tokens=data.get("eval_count"),
        raw=data,
    )
