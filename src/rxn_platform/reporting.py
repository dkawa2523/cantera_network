"""HTML report rendering utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import html
import json
from typing import Any

from rxn_platform.core import ArtifactManifest


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)


def _escape_json(payload: Any) -> str:
    return html.escape(_json_dump(payload), quote=False)


def _escape_text(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def _render_inputs_list(inputs: Sequence[Mapping[str, str]]) -> str:
    if not inputs:
        return '<p class="muted">No input artifacts provided.</p>'
    items = []
    for item in inputs:
        kind = _escape_text(item.get("kind"))
        artifact_id = _escape_text(item.get("id"))
        items.append(
            "<li>"
            f'<span class="pill">{kind}</span>'
            f"<code>{artifact_id}</code>"
            "</li>"
        )
    return "<ul class=\"inputs-list\">\n" + "\n".join(items) + "\n</ul>"


def render_report_html(
    *,
    title: str,
    dashboard: str,
    created_at: str,
    manifest: ArtifactManifest,
    inputs: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
    placeholders: Sequence[str],
) -> str:
    manifest_payload = manifest.to_dict()
    input_count = len(inputs)
    inputs_html = _render_inputs_list(inputs)
    config_json = _escape_json(config)
    manifest_json = _escape_json(manifest_payload)
    manifest_raw = _json_dump(manifest_payload)
    config_raw = _json_dump(config)

    placeholder_cards = []
    for label in placeholders:
        placeholder_cards.append(
            "<div class=\"slot\">"
            "<div class=\"slot-header\">Placeholder</div>"
            f"<div class=\"slot-title\">{_escape_text(label)}</div>"
            "<div class=\"slot-body\">"
            "Reserved for upcoming visualizations."
            "</div>"
            "</div>"
        )
    slots_html = "\n".join(placeholder_cards)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape_text(title)}</title>
  <style>
    :root {{
      --bg: #f5f0e7;
      --bg-accent: #dfe9ef;
      --panel: #ffffff;
      --ink: #1f2a33;
      --muted: #5f6c77;
      --accent: #0f6f68;
      --border: #d5d7d9;
      --shadow: 0 18px 40px rgba(19, 36, 48, 0.08);
      --font-display: "Trebuchet MS", "Lucida Sans Unicode", "Lucida Grande", sans-serif;
      --font-body: "Palatino Linotype", "Book Antiqua", Palatino, serif;
      --font-mono: "Courier New", Courier, monospace;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: var(--font-body);
      color: var(--ink);
      background:
        radial-gradient(circle at top right, #fefaf2 0%, transparent 55%),
        radial-gradient(circle at 15% 15%, #e5f2ef 0%, transparent 45%),
        linear-gradient(140deg, var(--bg) 0%, var(--bg-accent) 100%);
    }}

    code {{
      font-family: var(--font-mono);
      font-size: 0.9em;
    }}

    pre {{
      margin: 0;
      padding: 16px;
      background: #f7f5f0;
      border-radius: 12px;
      border: 1px solid var(--border);
      font-size: 12px;
      overflow-x: auto;
      font-family: var(--font-mono);
    }}

    .page {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 32px 20px 60px;
    }}

    .hero {{
      display: flex;
      flex-wrap: wrap;
      gap: 24px;
      align-items: center;
      justify-content: space-between;
      padding: 28px;
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.85);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      animation: slide-up 0.6s ease-out;
    }}

    .eyebrow {{
      font-family: var(--font-display);
      letter-spacing: 0.18em;
      text-transform: uppercase;
      font-size: 11px;
      color: var(--accent);
      margin: 0 0 8px;
    }}

    h1 {{
      font-family: var(--font-display);
      font-size: 32px;
      margin: 0 0 12px;
    }}

    .subtitle {{
      margin: 0;
      color: var(--muted);
    }}

    .meta {{
      display: grid;
      gap: 12px;
      min-width: 220px;
      padding: 16px;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: #fdfcf9;
    }}

    .meta-label {{
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      margin-bottom: 6px;
    }}

    .panel {{
      margin-top: 26px;
      padding: 22px;
      border-radius: 20px;
      background: var(--panel);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      animation: fade-in 0.7s ease-out;
    }}

    .panel h2 {{
      margin: 0 0 14px;
      font-family: var(--font-display);
      font-size: 20px;
    }}

    .inputs-list {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 10px;
    }}

    .inputs-list li {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 12px;
      border-radius: 12px;
      background: #fbfaf7;
      border: 1px solid var(--border);
    }}

    .pill {{
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      border-radius: 999px;
      background: #e7f3f0;
      color: var(--accent);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-family: var(--font-display);
    }}

    .grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}

    .slot {{
      border-radius: 18px;
      border: 1px dashed #b9c5c7;
      padding: 18px;
      background: #fcfdfb;
      min-height: 140px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}

    .slot-header {{
      font-size: 11px;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      color: var(--muted);
      font-family: var(--font-display);
    }}

    .slot-title {{
      font-family: var(--font-display);
      font-size: 16px;
    }}

    .slot-body {{
      color: var(--muted);
      font-size: 13px;
    }}

    .muted {{
      color: var(--muted);
      margin: 0;
    }}

    @keyframes slide-up {{
      from {{
        opacity: 0;
        transform: translateY(12px);
      }}
      to {{
        opacity: 1;
        transform: translateY(0);
      }}
    }}

    @keyframes fade-in {{
      from {{
        opacity: 0;
      }}
      to {{
        opacity: 1;
      }}
    }}

    .panel:nth-of-type(2) {{
      animation-delay: 0.08s;
    }}

    .panel:nth-of-type(3) {{
      animation-delay: 0.16s;
    }}

    .panel:nth-of-type(4) {{
      animation-delay: 0.24s;
    }}

    @media (max-width: 720px) {{
      .hero {{
        padding: 22px;
      }}

      h1 {{
        font-size: 26px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <div>
        <p class="eyebrow">rxn platform report</p>
        <h1>{_escape_text(title)}</h1>
        <p class="subtitle">Dashboard: {_escape_text(dashboard)} · Created: {_escape_text(created_at)}</p>
      </div>
      <div class="meta">
        <div>
          <span class="meta-label">Report ID</span>
          <code>{_escape_text(manifest.id)}</code>
        </div>
        <div>
          <span class="meta-label">Inputs</span>
          {_escape_text(input_count)}
        </div>
      </div>
    </header>

    <section class="panel">
      <h2>Input Artifacts</h2>
      {inputs_html}
    </section>

    <section class="panel">
      <h2>Config Summary</h2>
      <pre>{config_json}</pre>
    </section>

    <section class="panel">
      <h2>Report Manifest</h2>
      <pre>{manifest_json}</pre>
    </section>

    <section class="panel">
      <h2>Dashboard Slots</h2>
      <div class="grid">
        {slots_html}
      </div>
    </section>
  </div>

  <script type="application/json" id="report-config">
{config_raw}
  </script>
  <script type="application/json" id="report-manifest">
{manifest_raw}
  </script>
</body>
</html>
"""


__all__ = ["render_report_html"]
