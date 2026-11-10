"""Probe GitHub Copilot's auto-mode session API with your own token.

Spike 1 and 2 of COPILOT_GPT61_AUTO_PLAN.md (Part 3): does ``POST /models/session``
answer for a token issued the way the Portal issues them, what pool does it
return, how long does the session token live, and does ``/responses`` accept
``Copilot-Session-Token`` with a pool model?

Run it yourself; the token never leaves your machine except to GitHub:

    PowerShell:
      $env:EFP_PROBE_TOKEN = "<ghu_/ghp_/github_pat_ source token, or a Copilot token>"
      python copilot_auto_probe.py                 # token exchange + /models/session + GET /models
      python copilot_auto_probe.py --chat          # + one tiny /responses call with the session token
      python copilot_auto_probe.py --chat --bad-model claude-sonnet-5   # + capture the non-pool error
      python copilot_auto_probe.py --auto-v2 --prompt "..."             # which model /auto picks for a prompt
      python copilot_auto_probe.py --discount-test                      # is the auto pick billed at the auto discount?
      python copilot_auto_probe.py --discount-test --variants efp --repeat 20   # one header set, to find in the usage report

    Options: --model gpt-6.1-sol (pool model for --chat), --api-base https://... (override),
             --no-models (skip GET /models), --json (dump raw JSON of the session response,
             token redacted), --hints / --hints-wide (model_hints sweeps), --intent (intent
             router), --vscode-ids (VS Code's id headers), --editor-version / --plugin-version /
             --user-agent (send your real VS Code and extension versions)

Prints statuses, model ids, limits, expiry and error text only. Tokens are
redacted from anything printed. Proxy settings come from HTTPS_PROXY/NO_PROXY
like the runtime's default behaviour.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

GITHUB_API = "https://api.github.com"
DEFAULT_API_BASE = "https://api.githubcopilot.com"
PLUGIN_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.41.0",
    "Editor-Version": "vscode/1.133.0",
    "Editor-Plugin-Version": "copilot-chat/0.41.0",
    "Copilot-Integration-Id": "vscode-chat",
}
SOURCE_PREFIXES = ("ghu_", "ghp_", "gho_", "github_pat_")
# The runtime's Copilot allowlist after the GPT-5.6 removal (plan Part 2).
EFP_COPILOT_MODELS = ("gpt-6-astra", "gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol")

_SECRETS: list[str] = []


def redact(text: str) -> str:
    for secret in _SECRETS:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


def http(method: str, url: str, *, token: str, body: dict | None = None, extra: dict | None = None, timeout: float = 60.0):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json", **PLUGIN_HEADERS}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    if extra:
        headers.update(extra)
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, raw, time.monotonic() - started
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return exc.code, raw, time.monotonic() - started
    except urllib.error.URLError as exc:
        return 0, f"URLError: {exc.reason}", time.monotonic() - started


def exchange_source_token(source: str) -> tuple[str, str | None]:
    """ghu_/ghp_/github_pat_ -> Copilot token, like the runtime's token exchange."""
    status, raw, elapsed = http("GET", f"{GITHUB_API}/copilot_internal/v2/token", token=source)
    print(f"[exchange] GET /copilot_internal/v2/token -> {status} ({elapsed:.1f}s)")
    if status != 200:
        print(redact(raw[:600]))
        sys.exit("token exchange failed; an exchanged Copilot token is required for the rest")
    payload = json.loads(raw)
    token = payload.get("token")
    if not isinstance(token, str) or not token:
        sys.exit("token exchange returned no token")
    _SECRETS.append(token)
    expires_at = payload.get("expires_at")
    if isinstance(expires_at, (int, float)):
        print(f"[exchange] copilot token expires in {int(expires_at - time.time())}s; sku={payload.get('sku')!r}")
    endpoints = payload.get("endpoints") if isinstance(payload.get("endpoints"), dict) else {}
    api_base = endpoints.get("api") if isinstance(endpoints.get("api"), str) else None
    derived = api_base_from_token(token)
    print(f"[exchange] endpoints.api={api_base!r} proxy-ep-derived={derived!r} (the runtimes use the derived one)")
    if not api_base:
        api_base = derived
    return token, api_base


def api_base_from_token(token: str) -> str | None:
    # Copilot tokens look like "tid=...;exp=...;proxy-ep=proxy.individual.githubcopilot.com;...".
    for part in token.split(";"):
        key, _, value = part.partition("=")
        if key.strip() == "proxy-ep" and value.strip():
            host = value.strip()
            if host.startswith("proxy."):
                host = "api." + host[len("proxy."):]
            return f"https://{host}"
    return None


def probe_session(
    api_base: str,
    token: str,
    *,
    dump_json: bool,
    api_version: str | None,
    hints: list[str] | None = None,
    compact: bool = False,
    body_override: dict | None = None,
) -> dict | None:
    hints = [h for h in (hints or []) if h] or ["auto"]
    body = body_override if body_override is not None else {"auto_mode": {"model_hints": hints}}
    # VS Code's CAPI client sends X-GitHub-Api-Version on every CAPI request
    # (2026-08-01 in @vscode/copilot-api 0.5.2); the runtimes did not, and a
    # route can 404 for an unknown or missing API version.
    extra = {"X-GitHub-Api-Version": api_version} if api_version else {}
    status, raw, elapsed = http("POST", f"{api_base}/models/session", token=token, body=body, extra=extra)
    label = json.dumps(body) if body_override is not None else f"hints={hints}"
    if compact:
        if status != 200:
            print(f"[sweep] {label} -> {status} {redact(raw[:160]).strip()!r}")
            return None
        payload = json.loads(raw)
        if isinstance(payload.get("session_token"), str):
            _SECRETS.append(payload["session_token"])
        print(
            f"[sweep] {label} -> pool={payload.get('available_models')!r} selected={payload.get('selected_model')!r} "
            f"discounts={json.dumps(payload.get('discounted_costs'), sort_keys=True)}"
        )
        return payload
    print(
        f"[session] POST {api_base}/models/session {label} (X-GitHub-Api-Version={api_version or 'none'}) -> {status} ({elapsed:.1f}s)"
    )
    if status != 200:
        print("[session] body:", redact(raw[:800]) or "<empty>")
        return None
    payload = json.loads(raw)
    session_token = payload.get("session_token")
    if isinstance(session_token, str):
        _SECRETS.append(session_token)
    available = payload.get("available_models") or []
    expires_at = payload.get("expires_at")
    lifetime = int(expires_at - time.time()) if isinstance(expires_at, (int, float)) else None
    print(f"[session] session_token present: {isinstance(session_token, str) and bool(session_token)}; expires in {lifetime}s")
    print(f"[session] available_models ({len(available)}): {', '.join(map(str, available))}")
    pool = [m for m in EFP_COPILOT_MODELS if m in available]
    print(f"[session] intersection with EFP Copilot list: {pool or 'EMPTY'}")
    discounts = payload.get("discounted_costs")
    if isinstance(discounts, dict):
        print(f"[session] discounted_costs: {json.dumps(discounts, sort_keys=True)}")
    if "selected_model" in payload:
        print(f"[session] selected_model: {payload.get('selected_model')!r}")
    extra_keys = sorted(set(payload) - {"session_token", "available_models", "expires_at", "discounted_costs", "selected_model"})
    if extra_keys:
        print(f"[session] other keys: {extra_keys}")
    if dump_json:
        shown = dict(payload)
        if "session_token" in shown:
            shown["session_token"] = "<redacted>"
        print(json.dumps(shown, indent=2, sort_keys=True))
    return payload


def _pricing_tiers(row: dict) -> str:
    """``billing.token_prices`` as VS Code reads it: the default tier's prompt cap and the long-context tier.

    VS Code offers two context sizes per model, the default tier's
    ``max_prompt_tokens`` (GPT-6: 272k, the regular window GitHub bills at
    the base rate) and the full ``capabilities.limits`` window (1M, billed at
    the long-context tier above the threshold); nothing else is sent on the
    wire, the client just uses more of the window.
    """

    billing = row.get("billing") if isinstance(row.get("billing"), dict) else {}
    prices = billing.get("token_prices") if isinstance(billing.get("token_prices"), dict) else {}
    if not prices:
        return "none"
    parts = []
    for tier_name, tier in prices.items():
        if not isinstance(tier, dict):
            continue
        parts.append(
            f"{tier_name}(prompt_cap={tier.get('max_prompt_tokens')} in={tier.get('input_price')} out={tier.get('output_price')})"
        )
    return " ".join(parts) or json.dumps(prices, sort_keys=True)[:200]


def probe_models(
    api_base: str,
    token: str,
    *,
    detail_ids: tuple[str, ...] = (),
    api_version: str | None = None,
) -> None:
    # The API version header selects the schema: VS Code's CAPI client sends
    # it on /models too, and the 2026-10-20 /auto answer (which needs it)
    # described the GPT-6 line with OpenAI's 1M windows where the headerless
    # /models call had reported 400k.
    extra = {"X-GitHub-Api-Version": api_version} if api_version else {}
    status, raw, elapsed = http("GET", f"{api_base}/models", token=token, extra=extra)
    print(f"[models] GET {api_base}/models (api-version={api_version or 'none'}) -> {status} ({elapsed:.1f}s)")
    if status != 200:
        print("[models] body:", redact(raw[:600]) or "<empty>")
        return
    payload = json.loads(raw)
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        print("[models] unexpected shape:", type(rows).__name__)
        return
    for row in rows:
        if isinstance(row, dict) and str(row.get("id") or "") in detail_ids:
            # The whole entry: limits, supports, policy and anything that hints
            # at an extended context mode.
            print(f"[models:detail] {json.dumps(row, sort_keys=True)}")
    ids = [str(row.get("id") or "") for row in rows if isinstance(row, dict)]
    auto_like = [model_id for model_id in ids if "auto" in model_id.lower()]
    print(f"[models] {len(ids)} models listed; auto-like ids: {auto_like or 'none'}")
    print("[models] id | picker | policy | ctx/prompt/out | tool_calls | vision | reasoning_efforts")
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or "")
        if not (model_id.startswith("gpt-") or model_id in EFP_COPILOT_MODELS or "auto" in model_id.lower()):
            continue
        caps = row.get("capabilities") if isinstance(row.get("capabilities"), dict) else {}
        limits = caps.get("limits") if isinstance(caps.get("limits"), dict) else {}
        supports = caps.get("supports") if isinstance(caps.get("supports"), dict) else {}
        policy = row.get("policy") if isinstance(row.get("policy"), dict) else {}
        efforts = supports.get("reasoning_effort") or limits.get("reasoning_efforts") or row.get("supported_reasoning_efforts")
        print(
            f"[models] {model_id} | picker={row.get('model_picker_enabled')} | policy={policy.get('state')} | "
            f"{limits.get('max_context_window_tokens')}/{limits.get('max_prompt_tokens')}/{limits.get('max_output_tokens')} | "
            f"tools={supports.get('tool_calls')} | vision={supports.get('vision')} | efforts={efforts} | "
            f"tiers={_pricing_tiers(row)}"
        )
    long_context = [str(row.get("id")) for row in rows if isinstance(row, dict) and "long_context" in _pricing_tiers(row)]
    print(f"[models] models with a long_context pricing tier: {long_context or 'none'}")


def probe_intent(
    api_base: str,
    token: str,
    session_token: str,
    *,
    prompt: str,
    available_models: list[str],
    api_version: str | None,
) -> None:
    """VS Code's intent router (POST /models/session/intent): ranks the session's pool for one prompt.

    Its input is the pool the session returned (``available_models``); it can
    only answer with a subset of what it is given, so the prompt never widens
    the pool. VS Code runs it behind an experiment flag and only for panel chat.
    """

    body = {
        "prompt": prompt,
        "available_models": available_models,
        "prompt_char_count": len(prompt),
        "turn_number": 1,
    }
    extra = {"Copilot-Session-Token": session_token}
    if api_version:
        extra["X-GitHub-Api-Version"] = api_version
    status, raw, elapsed = http("POST", f"{api_base}/models/session/intent", token=token, body=body, extra=extra)
    print(f"[intent] POST /models/session/intent available_models={available_models} prompt_chars={len(prompt)} -> {status} ({elapsed:.1f}s)")
    if status != 200:
        print("[intent] body:", redact(raw[:800]) or "<empty>")
        return
    payload = json.loads(raw)
    print(
        f"[intent] predicted_label={payload.get('predicted_label')!r} confidence={payload.get('confidence')!r} "
        f"candidate_models={payload.get('candidate_models')!r} sticky_override={payload.get('sticky_override')!r}"
    )


def probe_auto_v2(
    api_base: str,
    token: str,
    *,
    prompt: str,
    tier: str | None,
    api_version: str | None,
    dump_json: bool = False,
) -> dict | None:
    """Auto v2, what current VS Code uses: ``POST /auto`` with the prompt and a tier.

    From microsoft/vscode extensions/copilot (autoV2Fetcher.ts, autoModeTiers.ts):
    body ``{prompt, tier?, has_image?, multi_turn?}``, tiers efficiency /
    balance / intelligence / fast (balance is the picker default), response
    ``{session_token, expires_at, selected_model: {id, ...model metadata},
    discounted_costs?}``. The chat request then sends ``selected_model.id`` with
    ``Copilot-Session-Token``. The server routes on the prompt text, unlike the
    older /models/session pool.
    """

    body: dict = {"prompt": prompt}
    if tier:
        body["tier"] = tier
    extra = {"X-GitHub-Api-Version": api_version} if api_version else {}
    status, raw, elapsed = http("POST", f"{api_base}/auto", token=token, body=body, extra=extra)
    print(f"[auto-v2] POST {api_base}/auto tier={tier or 'server default'} prompt={prompt[:40]!r} -> {status} ({elapsed:.1f}s)")
    if status != 200:
        print("[auto-v2] body:", redact(raw[:800]) or "<empty>")
        return None
    payload = json.loads(raw)
    session_token = payload.get("session_token")
    if isinstance(session_token, str):
        _SECRETS.append(session_token)
    selected = payload.get("selected_model") if isinstance(payload.get("selected_model"), dict) else {}
    caps = selected.get("capabilities") if isinstance(selected.get("capabilities"), dict) else {}
    limits = caps.get("limits") if isinstance(caps.get("limits"), dict) else {}
    supports = caps.get("supports") if isinstance(caps.get("supports"), dict) else {}
    expires_at = payload.get("expires_at")
    lifetime = int(expires_at - time.time()) if isinstance(expires_at, (int, float)) else None
    print(
        f"[auto-v2] selected_model={selected.get('id')!r} name={selected.get('name')!r} vendor={selected.get('vendor')!r} "
        f"limits={limits.get('max_context_window_tokens')}/{limits.get('max_prompt_tokens')}/{limits.get('max_output_tokens')} "
        f"efforts={supports.get('reasoning_effort')!r} tools={supports.get('tool_calls')!r} "
        f"discounts={json.dumps(payload.get('discounted_costs'), sort_keys=True)} session={'yes' if session_token else 'no'} expires_in={lifetime}s"
    )
    extra_keys = sorted(set(payload) - {"session_token", "expires_at", "selected_model", "discounted_costs"})
    if extra_keys:
        print(f"[auto-v2] other keys: {extra_keys}")
    # Undeclared in VS Code's AutoV2Response type but present on this seat
    # (2026-10-20 run): multi_turn / multi_turn_mode / hydra_scores. Their
    # shape decides whether the runtimes can hand the state back per turn.
    for key in ("multi_turn_mode", "multi_turn", "hydra_scores"):
        if key in payload:
            print(f"[auto-v2] {key}={redact(json.dumps(payload[key], sort_keys=True))[:600]}")
    selected_keys = sorted(set(selected) - {"id", "name", "vendor", "version", "capabilities", "model_picker_enabled"})
    if selected_keys:
        print(f"[auto-v2] selected_model extra keys: {selected_keys}")
    print(
        f"[auto-v2] selected_model endpoints={selected.get('supported_endpoints')!r} "
        f"category={selected.get('model_picker_category')!r} price={selected.get('model_picker_price_category')!r} "
        f"billing={json.dumps(selected.get('billing'), sort_keys=True)} preview={selected.get('preview')!r}"
    )
    if dump_json:
        print(redact(json.dumps(payload, indent=2, sort_keys=True)))
    return payload


def probe_context_test(api_base: str, token: str, *, model: str, k_tokens: int) -> None:
    """Does ``/responses`` accept a prompt above the catalog's ``max_prompt_tokens``?

    ``GET /models`` reports 272,000 prompt tokens for the GPT-6 line while
    ``POST /auto`` embeds OpenAI's 872k / 922k. This sends about ``k_tokens``
    thousand tokens of filler (``the quick brown fox jumps over the lazy dog.``
    is ten o200k tokens) and reports the status, the error text or the usage.
    It costs input tokens at the model's rate (GPT-6: about $2 per million,
    more above the long-context threshold), so run it once with a number just
    above the cap, e.g. ``--context-test 300``.
    """

    # Two characters per o200k token (" a", " b", ... are single tokens), so
    # 300k tokens travel as ~600 KB: the first run's 4.5-chars-per-token prose
    # (1.35 MB) made the gateway give up reading the body after 60 s (408
    # user_request_timeout) before it ever judged the length. tiktoken, when
    # installed, counts the filler exactly; otherwise the estimate is printed.
    unit = " " + " ".join("abcdefghijklmnopqrstuvwxyz")
    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding("o200k_base")
        tokens_per_unit = len(encoding.encode(unit))
    except Exception:  # noqa: BLE001 - tiktoken is optional; the estimate is close enough.
        encoding = None
        tokens_per_unit = 26
    repeats = max(1, -(-int(k_tokens) * 1000 // tokens_per_unit))
    filler = unit * repeats
    counted = len(encoding.encode(filler)) if encoding is not None else repeats * tokens_per_unit
    print(
        f"[context-test] filler {len(filler):,} chars ~{counted:,} tokens "
        f"({'tiktoken' if encoding is not None else 'estimated'}; base tier cap 272,000)"
    )
    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": filler + "\n\nReply with the single word pong."}],
            }
        ],
        "stream": False,
        "reasoning": {"effort": "low"},
        "max_output_tokens": 32,
    }
    print(f"[context-test] POST {api_base}/responses model={model} ...")
    status, raw, elapsed = http("POST", f"{api_base}/responses", token=token, body=body, timeout=600)
    print(f"[context-test] -> {status} ({elapsed:.1f}s, ~{len(filler) / 1024 / max(elapsed, 0.1):.0f} KB/s if the time was the upload)")
    if status != 200:
        print("[context-test] body:", redact(raw[:800]) or "<empty>")
        return
    payload = json.loads(raw)
    text = ""
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text += part["text"]
    print(
        f"[context-test] response.model={payload.get('model')!r} text={text.strip()[:40]!r} "
        f"usage={json.dumps(payload.get('usage'), sort_keys=True)}"
    )


def probe_chat(
    api_base: str,
    token: str,
    session_token: str | None,
    model: str,
    *,
    label: str,
    endpoint: str = "responses",
    reasoning: bool = True,
) -> None:
    """One tiny request on /responses (default) or /chat/completions with the session token."""

    if endpoint == "chat":
        path = "/chat/completions"
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word pong."}],
            "max_tokens": 16,
            "stream": False,
        }
    else:
        path = "/responses"
        body = {
            "model": model,
            "input": "Reply with the single word pong.",
            "max_output_tokens": 16,
            "stream": False,
        }
        if reasoning:
            body["reasoning"] = {"effort": "low"}
    extra = {"Openai-Intent": "conversation-edits", "x-initiator": "agent"}
    if session_token:
        extra["Copilot-Session-Token"] = session_token
    status, raw, elapsed = http("POST", f"{api_base}{path}", token=token, body=body, extra=extra)
    print(f"[chat:{label}] POST {path} model={model} session_header={'yes' if session_token else 'no'} -> {status} ({elapsed:.1f}s)")
    if status != 200:
        print(f"[chat:{label}] body:", redact(raw[:800]) or "<empty>")
        return
    payload = json.loads(raw)
    text = ""
    if endpoint == "chat":
        for choice in payload.get("choices") or []:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict):
                text += str(message.get("content") or "")
    else:
        for item in payload.get("output") or []:
            if isinstance(item, dict) and item.get("type") == "message":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text += str(part.get("text") or "")
    usage = payload.get("usage") or {}
    print(f"[chat:{label}] response.model={payload.get('model')!r} text={text.strip()!r} usage={json.dumps(usage)}")


# Header sets for --discount-test, each sent with the model /auto picked. The
# question: which of them makes Copilot bill the request in the auto pool (10%
# off, "Auto: <model>" in the usage report) instead of as the plain model.
# VS Code's HTTP chat POST (extensions/copilot chatMLFetcher.ts -> networking.ts
# networkRequest) carries X-Request-Id, OpenAI-Intent, X-GitHub-Api-Version
# 2026-01-09, X-Interaction-Id, X-Initiator, X-Interaction-Type and
# X-Agent-Task-Id next to Copilot-Session-Token; the runtimes send only the
# session token on top of their usual chat headers.
EFP_CHAT_HEADERS = {"Openai-Intent": "conversation-edits", "x-initiator": "agent"}
DISCOUNT_VARIANTS: dict[str, dict] = {
    "baseline": {"session": False, "headers": {}, "about": "no session token (the plain model's price)"},
    "efp": {"session": True, "headers": {}, "about": "session token only (what the runtimes send today)"},
    "version-0801": {
        "session": True,
        "headers": {"X-GitHub-Api-Version": "2026-08-01"},
        "about": "session token + the API version the runtimes send on /auto",
    },
    "version-0109": {
        "session": True,
        "headers": {"X-GitHub-Api-Version": "2026-01-09"},
        "about": "session token + the API version VS Code sends on chat",
    },
    "vscode": {
        "session": True,
        "headers": {
            "X-GitHub-Api-Version": "2026-01-09",
            "Openai-Intent": "conversation-agent",
            "X-Interaction-Type": "conversation-agent",
            "x-initiator": "user",
        },
        "request_ids": True,
        "about": "session token + VS Code's chat headers (version, request / interaction ids, interaction type)",
    },
}


def _find_key(value, key: str):
    """The first ``key`` found anywhere in a decoded JSON value, or None."""

    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_key(child, key)
            if found is not None:
                return found
    return None


# Response headers printed next to a charge: the request ids a support ticket
# needs and the quota / usage snapshots Copilot stamps on chat answers.
BILLING_HEADER_MARKS = ("quota", "usage", "request-id", "ratelimit")


def responses_stream(url: str, *, token: str, body: dict, extra: dict, timeout: float = 120.0):
    """One streamed ``/responses`` call, the way the runtimes send it.

    Returns ``(status, final event, raw error body, seconds, seen)``. The final
    event is the last one carrying a ``response`` and is kept whole: Copilot
    can put its charge (``copilot_usage.total_nano_aiu``) next to ``response``
    instead of inside ``response.usage``. ``seen`` holds the event types in
    order, the last ``copilot_usage`` found on any event with the event type
    it came on, and the billing-related response headers.
    """

    headers = {"Authorization": f"Bearer {token}", **PLUGIN_HEADERS, "Content-Type": "application/json"}
    headers.update(extra)
    headers["Accept"] = "text/event-stream"
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    started = time.monotonic()
    final = None
    seen: dict = {"event_types": [], "copilot_usage": None, "copilot_usage_event": None, "headers": {}}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            seen["headers"] = {
                name: value
                for name, value in response.headers.items()
                if any(mark in name.lower() for mark in BILLING_HEADER_MARKS)
            }
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                event_type = str(event.get("type") or "<untyped>")
                if event_type not in seen["event_types"]:
                    seen["event_types"].append(event_type)
                copilot_usage = _find_key(event, "copilot_usage")
                if copilot_usage is not None:
                    seen["copilot_usage"] = copilot_usage
                    seen["copilot_usage_event"] = event_type
                if isinstance(event.get("response"), dict):
                    final = event
            return status, final, "", time.monotonic() - started, seen
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return exc.code, None, raw, time.monotonic() - started, seen
    except urllib.error.URLError as exc:
        return 0, None, f"URLError: {exc.reason}", time.monotonic() - started, seen


def _default_tier_prices(*entries) -> dict | None:
    """``billing.token_prices.default`` (credits per million tokens) from the first model entry that has it."""

    for entry in entries:
        billing = entry.get("billing") if isinstance(entry, dict) else None
        prices = billing.get("token_prices") if isinstance(billing, dict) else None
        tier = prices.get("default") if isinstance(prices, dict) else None
        if isinstance(tier, dict) and tier.get("input_price") is not None and tier.get("output_price") is not None:
            return tier
    return None


def _list_price_credits(usage: dict, prices: dict | None) -> float | None:
    """What the usage costs at the model's published default-tier prices, in credits."""

    if not prices:
        return None
    try:
        input_price = float(prices["input_price"])
        output_price = float(prices["output_price"])
    except (KeyError, TypeError, ValueError):
        return None
    cached_price = None
    for key in ("cached_input_price", "cache_read_price", "cached_price", "input_cached_price", "cached_input"):
        if prices.get(key) is not None:
            cached_price = float(prices[key])
            break
    if cached_price is None:
        cached_price = input_price / 10.0
    details = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
    prompt = int(usage.get("input_tokens") or 0)
    cached = min(int(details.get("cached_tokens") or 0), prompt)
    output = int(usage.get("output_tokens") or 0)
    return ((prompt - cached) * input_price + cached * cached_price + output * output_price) / 1_000_000.0


def probe_discount_test(
    api_base: str,
    token: str,
    *,
    prompt: str,
    tier: str | None,
    api_version: str | None,
    variants: list[str],
    repeat: int,
) -> None:
    """Is a request sent with the auto session token billed at the auto discount?

    Routes once (``POST /auto``), then sends the same small ``/responses``
    request with the picked model under each header set of DISCOUNT_VARIANTS
    and compares what Copilot says it charged (``copilot_usage.total_nano_aiu``
    on the stream's last event, 1 credit = 10^9 nano-AIU) with the model's list price
    for the tokens used. ``baseline`` (no session token) is the plain model;
    a variant whose charge per list-price credit is ~10% below the baseline's
    is billed in the auto pool. Each request starts with a fresh nonce and
    stays under 1,024 prompt tokens, so no prompt cache read or write skews
    the comparison. Costs a few thousandths of a credit per request on Luna.
    """

    import uuid

    decision = probe_auto_v2(api_base, token, prompt=prompt, tier=tier, api_version=api_version)
    if decision is None:
        sys.exit("[discount] /auto did not answer; nothing to compare")
    selected = decision.get("selected_model") if isinstance(decision.get("selected_model"), dict) else {}
    model = str(selected.get("id") or "")
    session_token = decision.get("session_token")
    if not model or not isinstance(session_token, str) or not session_token:
        sys.exit("[discount] /auto answered without a model or a session token")
    discounts = decision.get("discounted_costs") if isinstance(decision.get("discounted_costs"), dict) else {}
    try:
        expected_discount = float(discounts.get(model))
    except (TypeError, ValueError):
        expected_discount = 0.1
        print(f"[discount] /auto names no discount for {model}; assuming the documented 10%")
    prices = _default_tier_prices(selected)
    if prices is None:
        # /auto leaves token pricing unset on some seats; the catalog row has it.
        extra = {"X-GitHub-Api-Version": api_version} if api_version else {}
        status, raw, _ = http("GET", f"{api_base}/models", token=token, extra=extra)
        if status == 200:
            rows = json.loads(raw).get("data") or []
            prices = _default_tier_prices(*[row for row in rows if isinstance(row, dict) and row.get("id") == model])
    print(
        f"[discount] model={model} expected auto discount={expected_discount:.0%} "
        f"list prices (credits per 1M tokens)={json.dumps(prices, sort_keys=True) if prices else 'unknown'}"
    )
    print(f"[discount] started {time.strftime('%Y-%m-%d %H:%M:%S %z')}; variants: {', '.join(variants)}; repeat={repeat}")

    filler = " ".join(["The quick brown fox jumps over the lazy dog."] * 25)
    results: dict[str, list[tuple[float, float | None]]] = {}
    for name in variants:
        variant = DISCOUNT_VARIANTS[name]
        print(f"[discount:{name}] {variant['about']}")
        for attempt in range(repeat):
            extra = dict(EFP_CHAT_HEADERS)
            extra.update(variant["headers"])
            request_id = str(uuid.uuid4())
            if variant.get("request_ids"):
                extra["X-Request-Id"] = request_id
                extra["X-Agent-Task-Id"] = request_id
                extra["X-Interaction-Id"] = str(uuid.uuid4())
            if variant["session"]:
                extra["Copilot-Session-Token"] = session_token
            body = {
                "model": model,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                # The nonce comes first so no request shares a cacheable prefix with another.
                                "text": f"nonce {uuid.uuid4()}\n{filler}\n\nReply with the single word pong.",
                            }
                        ],
                    }
                ],
                "stream": True,
                "reasoning": {"effort": "low"},
                "max_output_tokens": 64,
            }
            status, final_event, raw, elapsed, seen = responses_stream(
                f"{api_base}/responses", token=token, body=body, extra=extra
            )
            label = f"[discount:{name}#{attempt + 1}]"
            if status != 200 or final_event is None:
                print(f"{label} -> {status} ({elapsed:.1f}s) {redact(raw[:400]).strip() or '<no final event>'}")
                continue
            final = final_event["response"]
            usage = final.get("usage") if isinstance(final.get("usage"), dict) else {}
            copilot_usage = seen["copilot_usage"]
            nano = copilot_usage.get("total_nano_aiu") if isinstance(copilot_usage, dict) else None
            if attempt == 0:
                # Once per header set: where the charge sits in the stream (the
                # runtimes read response.usage only) and what else Copilot says.
                if copilot_usage is None:
                    where = "absent"
                elif "copilot_usage" in usage:
                    where = "response.usage"
                elif "copilot_usage" in final_event:
                    where = f"top level of the {seen['copilot_usage_event']} event"
                else:
                    where = f"inside the {seen['copilot_usage_event']} event"
                print(
                    f"{label} copilot_usage: {where}; final event keys={sorted(final_event)} "
                    f"events={seen['event_types']} headers={redact(json.dumps(seen['headers'], sort_keys=True))}"
                )
            details = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
            tokens = (
                f"in={usage.get('input_tokens')} cached={details.get('cached_tokens', 0)} out={usage.get('output_tokens')}"
            )
            if not isinstance(nano, (int, float)) or isinstance(nano, bool):
                print(
                    f"{label} -> 200 ({elapsed:.1f}s) response.model={final.get('model')!r} {tokens} "
                    f"no copilot_usage.total_nano_aiu; usage={json.dumps(usage, sort_keys=True)}"
                )
                continue
            charged = float(nano) / 1e9
            listed = _list_price_credits(usage, prices)
            ratio = f"{charged / listed:.4f}" if listed else "n/a"
            print(
                f"{label} -> 200 ({elapsed:.1f}s) response.model={final.get('model')!r} {tokens} "
                f"charged={charged:.9f} credits list={'%.9f' % listed if listed else 'n/a'} charged/list={ratio}"
            )
            results.setdefault(name, []).append((charged, listed))

    def charged_per_list(name: str) -> float | None:
        rows = [(charged, listed) for charged, listed in results.get(name, []) if listed]
        if not rows:
            return None
        return sum(charged for charged, _ in rows) / sum(listed for _, listed in rows)

    print("[discount] summary (charged per list-price credit; the baseline is the plain model)")
    baseline = charged_per_list("baseline")
    for name in variants:
        value = charged_per_list(name)
        if value is None:
            print(f"[discount]   {name:<13} no usable answer")
            continue
        if name == "baseline" or baseline is None:
            verdict = "" if name == "baseline" else "no baseline to compare with (run it with baseline)"
            print(f"[discount]   {name:<13} {value:.4f} {verdict}".rstrip())
            continue
        relative = value / baseline
        if abs(relative - (1.0 - expected_discount)) <= 0.02:
            verdict = "AUTO DISCOUNT APPLIED"
        elif abs(relative - 1.0) <= 0.02:
            verdict = "no discount (billed as the plain model)"
        else:
            verdict = "unclear"
        print(f"[discount]   {name:<13} {value:.4f} = {relative:.3f} x baseline -> {verdict}")
    if not results:
        print(
            "[discount] Copilot returned no charge (copilot_usage.total_nano_aiu) on any request, so nothing can be "
            "compared here; look for these requests in the usage report instead"
        )
    elif not any(listed for rows in results.values() for _, listed in rows):
        print("[discount] no list price found for the model, so only the raw charges above can be compared by hand")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chat", action="store_true", help="also send one tiny /responses request with the session token")
    parser.add_argument("--model", default=None, help="pool model for --chat (default: first EFP model in available_models)")
    parser.add_argument("--bad-model", default=None, help="also send a /responses request with this non-pool model and the session token")
    parser.add_argument("--api-base", default=None, help="override the Copilot API base URL")
    parser.add_argument("--no-models", action="store_true", help="skip GET /models")
    parser.add_argument("--json", action="store_true", help="dump the raw /models/session JSON (token redacted)")
    parser.add_argument(
        "--api-version",
        default="2026-08-01",
        help="X-GitHub-Api-Version sent on /models/session, as VS Code's CAPI client does (default 2026-08-01); pass '' to omit",
    )
    parser.add_argument("--hint", action="append", default=None, help="model_hints value for the session (repeatable; default auto)")
    parser.add_argument(
        "--hints",
        action="store_true",
        help="sweep a set of model_hints values and print each pool: auto, efficiency, balance, balanced, intelligence, quality, gpt-6.1-sol, gpt-6-sol, gpt",
    )
    parser.add_argument("--chat-completions", action="store_true", help="with --chat: also try /chat/completions with the pool model and the session token")
    parser.add_argument("--models-detail", action="store_true", help="print the full GET /models entries for gpt-6.1-sol, gpt-6-sol and mai-code-1.1-flash")
    parser.add_argument(
        "--intent",
        nargs="?",
        const=(
            "Refactor the payment service: split the 2,000-line PaymentProcessor into bounded modules, keep the public API, "
            "add retries with idempotency keys for the gateway calls, migrate the integration tests, and explain the trade-offs."
        ),
        default=None,
        help="also call the intent router (/models/session/intent) with this prompt (default: a long, complex coding task)",
    )
    parser.add_argument(
        "--intent-models",
        default=None,
        help="comma-separated available_models to hand the router instead of the session's pool (e.g. gpt-6.1-sol,gpt-6-sol,mai-code-1.1-flash)",
    )
    parser.add_argument(
        "--hints-wide",
        action="store_true",
        help="sweep ~40 model_hints candidates (picker categories, tiers, combos, empty) and print one line per pool",
    )
    parser.add_argument(
        "--vscode-ids",
        action="store_true",
        help="add the VScode-SessionId / VScode-MachineId / Editor-Device-Id headers VS Code's CAPI client sends (random ids)",
    )
    parser.add_argument(
        "--auto-v2",
        action="store_true",
        help="call POST /auto (Auto v2, what current VS Code uses) with --prompt for each of --tiers, then one chat with the selected model",
    )
    parser.add_argument(
        "--context-test",
        type=int,
        default=None,
        metavar="K_TOKENS",
        help="send about K thousand tokens of filler to /responses with --model (default gpt-6.1-sol) to learn whether the "
        "gateway enforces the catalog's 272k prompt cap; costs input tokens, run once (e.g. 300)",
    )
    parser.add_argument(
        "--discount-test",
        action="store_true",
        help="route once on --prompt (POST /auto), then send the same small /responses request with the picked model under "
        "several header sets and compare Copilot's own charge (copilot_usage.total_nano_aiu) with the list price: "
        "tells whether the session token earns the auto discount, and which headers it needs",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DISCOUNT_VARIANTS),
        help="comma-separated header sets for --discount-test (default: all): " + ", ".join(DISCOUNT_VARIANTS),
    )
    parser.add_argument("--repeat", type=int, default=1, help="requests per variant for --discount-test (default 1)")
    parser.add_argument("--tier", default="balance", help="auto tier for --discount-test (default balance; '' lets the server pick)")
    parser.add_argument("--prompt", default="hey", help="prompt text for --auto-v2 and --discount-test (default: hey)")
    parser.add_argument("--tiers", default="balance,efficiency,intelligence,fast,", help="comma-separated tiers for --auto-v2; an empty item means no tier (server default)")
    parser.add_argument("--editor-version", default=None, help="override Editor-Version (e.g. vscode/1.117.2, your VS Code's real version)")
    parser.add_argument("--plugin-version", default=None, help="override Editor-Plugin-Version (e.g. copilot-chat/0.39.1, your Copilot Chat extension's version)")
    parser.add_argument("--user-agent", default=None, help="override User-Agent (e.g. GitHubCopilotChat/0.39.1)")
    args = parser.parse_args()
    api_version = (args.api_version or "").strip() or None
    if args.editor_version:
        PLUGIN_HEADERS["Editor-Version"] = args.editor_version
    if args.plugin_version:
        PLUGIN_HEADERS["Editor-Plugin-Version"] = args.plugin_version
    if args.user_agent:
        PLUGIN_HEADERS["User-Agent"] = args.user_agent
    if args.vscode_ids:
        import uuid

        PLUGIN_HEADERS["VScode-SessionId"] = f"{uuid.uuid4()}{int(time.time() * 1000)}"
        PLUGIN_HEADERS["VScode-MachineId"] = uuid.uuid4().hex + uuid.uuid4().hex
        PLUGIN_HEADERS["Editor-Device-Id"] = str(uuid.uuid4())
    print(f"[headers] {json.dumps({k: v for k, v in PLUGIN_HEADERS.items()}, sort_keys=True)}")

    raw_token = (os.environ.get("EFP_PROBE_TOKEN") or "").strip()
    if not raw_token:
        sys.exit("set EFP_PROBE_TOKEN to a GitHub source token (ghu_/ghp_/github_pat_) or a Copilot token")
    _SECRETS.append(raw_token)

    api_base = None
    if raw_token.lower().startswith(SOURCE_PREFIXES):
        token, api_base = exchange_source_token(raw_token)
    else:
        token = raw_token
        api_base = api_base_from_token(token)
        print("[exchange] skipped: EFP_PROBE_TOKEN already looks like a Copilot token")
    api_base = (args.api_base or api_base or DEFAULT_API_BASE).rstrip("/")
    print(f"[api] base url: {api_base}")

    if args.context_test:
        probe_context_test(api_base, token, model=args.model or "gpt-6.1-sol", k_tokens=args.context_test)
        return

    if args.discount_test:
        variants = [name.strip() for name in args.variants.split(",") if name.strip()]
        unknown = [name for name in variants if name not in DISCOUNT_VARIANTS]
        if unknown or not variants:
            sys.exit(f"--variants: unknown {unknown}; choose from {', '.join(DISCOUNT_VARIANTS)}")
        probe_discount_test(
            api_base,
            token,
            prompt=args.prompt,
            tier=(args.tier or "").strip() or None,
            api_version=api_version,
            variants=variants,
            repeat=max(1, args.repeat),
        )
        return

    if args.auto_v2:
        tiers = [t.strip() for t in args.tiers.split(",")]
        decisions: list[tuple[str | None, dict]] = []
        for tier in tiers:
            decision = probe_auto_v2(
                api_base, token, prompt=args.prompt, tier=tier or None, api_version=api_version, dump_json=args.json
            )
            if decision is not None:
                decisions.append((tier or None, decision))
        # A longer, clearly agentic prompt: does the routing change with the text?
        long_prompt = (
            "Refactor the payment service: split the 2,000-line PaymentProcessor into bounded modules, keep the public API, "
            "add retries with idempotency keys for the gateway calls, migrate the integration tests, and explain the trade-offs."
        )
        probe_auto_v2(api_base, token, prompt=long_prompt, tier="balance", api_version=api_version)
        if decisions:
            tier, decision = decisions[0]
            selected = decision.get("selected_model") or {}
            model = str(selected.get("id") or "")
            session_token = decision.get("session_token")
            if model and isinstance(session_token, str):
                probe_chat(api_base, token, session_token, model, label=f"auto-v2:{tier or 'default'}")
        return

    session = probe_session(api_base, token, dump_json=args.json, api_version=api_version, hints=args.hint)
    if session is None and api_version:
        # Tell apart "wrong API version" from "no auto for this token/plan".
        probe_session(api_base, token, dump_json=False, api_version=None, hints=args.hint)
    if args.hints:
        # Does the hint steer the pool? VS Code's default is "auto"; the tiers in
        # the docs (efficiency / balance / intelligence) and plain model ids are
        # the obvious things to try.
        for hint in ("efficiency", "balance", "balanced", "intelligence", "quality", "gpt-6.1-sol", "gpt-6-sol", "gpt"):
            probe_session(api_base, token, dump_json=False, api_version=api_version, hints=[hint])
    if args.hints_wide:
        # VS Code's hint is an experiment treatment variable we cannot see; try
        # the /models picker categories, tier spellings, combos and the empty
        # list, one compact line each.
        candidates: list[list[str]] = [[h] for h in (
            "auto", "lightweight", "versatile", "powerful", "standard", "premium", "default", "none",
            "fast", "smart", "cheap", "reasoning", "coding", "chat", "agent", "editor", "panel", "inline",
            "auto-v2", "auto_v2", "autov2", "auto2", "v2", "auto-balanced", "auto-efficiency", "auto-intelligence",
            "auto:balance", "auto/balance", "tier-balance", "balance-tier", "gpt-6", "gpt6", "openai", "copilot",
            "Auto", "AUTO", "auto-model", "automode", "auto_mode",
        )]
        candidates += [["auto", "balance"], ["balance", "auto"], ["auto", "gpt-6.1-sol"], ["gpt-6.1-sol", "auto"], ["auto", "powerful"], []]
        for hint_list in candidates:
            probe_session(api_base, token, dump_json=False, api_version=api_version, hints=hint_list, compact=True,
                          body_override={"auto_mode": {"model_hints": hint_list}})
        for body in ({"auto_mode": {}}, {}, {"auto_mode": {"model_hints": ["auto"], "tier": "balance"}}, {"auto_mode": {"model_hints": ["auto"], "location": "panel"}}):
            probe_session(api_base, token, dump_json=False, api_version=api_version, compact=True, body_override=body)
    if not args.no_models:
        detail = ("gpt-6.1-sol", "gpt-6-sol", "mai-code-1.1-flash") if args.models_detail else ()
        probe_models(api_base, token, detail_ids=detail, api_version=api_version)

    if args.intent and isinstance(session, dict) and isinstance(session.get("session_token"), str):
        pool_models = [str(m) for m in (session.get("available_models") or [])]
        router_models = [m.strip() for m in args.intent_models.split(",") if m.strip()] if args.intent_models else pool_models
        probe_intent(api_base, token, session["session_token"], prompt=args.intent, available_models=router_models, api_version=api_version)
        if not args.intent_models:
            # And with the GPT-6 line offered alongside: does the router ever
            # pick outside the session's pool?
            widened = pool_models + [m for m in EFP_COPILOT_MODELS if m not in pool_models]
            probe_intent(api_base, token, session["session_token"], prompt=args.intent, available_models=widened, api_version=api_version)

    if args.chat:
        session_token = session.get("session_token") if isinstance(session, dict) else None
        session_token = session_token if isinstance(session_token, str) else None
        available = list(session.get("available_models") or []) if isinstance(session, dict) else []
        selected = session.get("selected_model") if isinstance(session, dict) else None
        # What the server picked, else our preference, else whatever is in the pool.
        model = (
            args.model
            or (selected if isinstance(selected, str) and selected else None)
            or next((m for m in EFP_COPILOT_MODELS if m in available), None)
            or (available[0] if available else None)
            or "gpt-6-sol"
        )
        probe_chat(api_base, token, session_token, model, label="pool")
        if model not in EFP_COPILOT_MODELS:
            # A non-GPT pool model may not accept a reasoning block on /responses.
            probe_chat(api_base, token, session_token, model, label="pool-no-reasoning", reasoning=False)
            # The question that decides the design: does the auto session accept a
            # GPT-6 model it did not list? If so, "auto session + GPT-6" keeps the
            # discount; if not, auto on this plan means the pool model or nothing.
            probe_chat(api_base, token, session_token, "gpt-6.1-sol", label="gpt-6.1-sol-with-session")
            probe_chat(api_base, token, None, "gpt-6.1-sol", label="gpt-6.1-sol-no-session")
        if args.chat_completions:
            probe_chat(api_base, token, session_token, model, label="pool", endpoint="chat")
        if args.bad_model:
            probe_chat(api_base, token, session_token, args.bad_model, label="non-pool")
            if args.chat_completions:
                probe_chat(api_base, token, session_token, args.bad_model, label="non-pool", endpoint="chat")


if __name__ == "__main__":
    main()
