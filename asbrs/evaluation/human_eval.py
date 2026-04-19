"""evaluation/human_eval.py — Human evaluation sheet generator.

Produces a self-contained, browser-renderable HTML file that presents
a random sample of test sessions together with:

  - The items the user browsed (numbered list).
  - The inferred purchase intent (LLM summary + confidence).
  - The top-5 recommendations with per-item explanations.
  - 5-star rating fields for Relevance and Explanation Quality.

All CSS is embedded inline so the file is fully portable.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List

from agent.interfaces import IntentResult, RecommendationOutput

logger = logging.getLogger(__name__)

# Output path used when no explicit destination is passed.
_DEFAULT_OUTPUT = Path("evaluation") / "human_eval_sheet.html"

# Maximum recommendations displayed per session.
_TOP_N_RECS = 5


class HumanEvalExporter:
    """Generate a human-evaluation HTML sheet from ASBRS model outputs.

    Attributes:
        output_path: Destination path for the generated HTML file.
        seed:        Random seed for reproducible session sampling.
    """

    def __init__(
        self,
        output_path: Path | str = _DEFAULT_OUTPUT,
        seed: int = 42,
    ) -> None:
        """Initialise the exporter.

        Args:
            output_path: Where to write the HTML file.
            seed:        Random seed for session sampling reproducibility.
        """
        self.output_path = Path(output_path)
        self.seed = seed

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_eval_sheet(
        self,
        sessions: List[List[str]],
        recommendations: List[List[RecommendationOutput]],
        intents: List[IntentResult],
        n_sessions: int = 10,
    ) -> None:
        """Sample sessions and write a human-evaluation HTML file.

        Args:
            sessions:        All test sessions; each is a list of item-ID
                             strings (ASINs or titles).
            recommendations: One list of RecommendationOutput per session,
                             aligned with *sessions*.
            intents:         One IntentResult per session, aligned with
                             *sessions*.
            n_sessions:      Number of sessions to include in the sheet.

        Raises:
            ValueError: If the three lists have different lengths.
        """
        if not (len(sessions) == len(recommendations) == len(intents)):
            raise ValueError(
                "sessions, recommendations, and intents must have the same length. "
                f"Got {len(sessions)}, {len(recommendations)}, {len(intents)}."
            )

        total = len(sessions)
        n_sessions = min(n_sessions, total)

        rng = random.Random(self.seed)
        indices = rng.sample(range(total), n_sessions)

        sampled_sessions = [sessions[i] for i in indices]
        sampled_recs = [recommendations[i] for i in indices]
        sampled_intents = [intents[i] for i in indices]

        html = self._build_html(sampled_sessions, sampled_recs, sampled_intents)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(html, encoding="utf-8")
        logger.info("Human eval sheet written to %s", self.output_path)

    # ── HTML construction ─────────────────────────────────────────────────────

    def _build_html(
        self,
        sessions: List[List[str]],
        recommendations: List[List[RecommendationOutput]],
        intents: List[IntentResult],
    ) -> str:
        """Assemble the full HTML document.

        Args:
            sessions:        Sampled session lists.
            recommendations: Corresponding recommendation lists.
            intents:         Corresponding intent results.

        Returns:
            Complete HTML string.
        """
        session_blocks = "\n".join(
            self._render_session_block(idx + 1, sess, recs, intent)
            for idx, (sess, recs, intent) in enumerate(
                zip(sessions, recommendations, intents)
            )
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ASBRS Human Evaluation Sheet</title>
  <style>
    /* ── Base ────────────────────────────────────────────────────────── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #f4f6f9;
      color: #2d3748;
      padding: 2rem;
      line-height: 1.6;
    }}
    h1 {{
      text-align: center;
      font-size: 1.8rem;
      margin-bottom: 0.4rem;
      color: #1a202c;
    }}
    .subtitle {{
      text-align: center;
      color: #718096;
      font-size: 0.95rem;
      margin-bottom: 2.5rem;
    }}
    /* ── Session card ────────────────────────────────────────────────── */
    .session-card {{
      background: #fff;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 1.6rem 2rem;
      margin-bottom: 2.2rem;
      box-shadow: 0 2px 6px rgba(0,0,0,.07);
    }}
    .session-header {{
      font-size: 1.15rem;
      font-weight: 700;
      color: #2b6cb0;
      border-bottom: 2px solid #bee3f8;
      padding-bottom: 0.5rem;
      margin-bottom: 1.2rem;
    }}
    /* ── Section headings ────────────────────────────────────────────── */
    .section-label {{
      font-weight: 600;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: .06em;
      color: #4a5568;
      margin-bottom: 0.5rem;
    }}
    .section-block {{ margin-bottom: 1.4rem; }}
    /* ── Browsing history list ───────────────────────────────────────── */
    ol.browsing-list {{
      padding-left: 1.4rem;
      color: #4a5568;
      font-size: 0.93rem;
    }}
    ol.browsing-list li {{ margin-bottom: 0.2rem; }}
    /* ── Intent box ──────────────────────────────────────────────────── */
    .intent-box {{
      background: #ebf8ff;
      border-left: 4px solid #4299e1;
      border-radius: 6px;
      padding: 0.9rem 1.1rem;
      font-size: 0.93rem;
    }}
    .intent-box .summary {{ font-style: italic; margin-bottom: 0.4rem; }}
    .intent-box .meta {{ color: #718096; font-size: 0.82rem; }}
    .keywords {{ display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.4rem; }}
    .keyword-badge {{
      background: #bee3f8;
      color: #2b6cb0;
      border-radius: 999px;
      padding: 0.15rem 0.7rem;
      font-size: 0.78rem;
      font-weight: 600;
    }}
    /* ── Recommendation table ────────────────────────────────────────── */
    table.recs-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    table.recs-table th {{
      background: #edf2f7;
      text-align: left;
      padding: 0.55rem 0.75rem;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: .05em;
      color: #4a5568;
      border-bottom: 1px solid #e2e8f0;
    }}
    table.recs-table td {{
      padding: 0.6rem 0.75rem;
      border-bottom: 1px solid #f0f4f8;
      vertical-align: top;
    }}
    table.recs-table tr:last-child td {{ border-bottom: none; }}
    .score-badge {{
      display: inline-block;
      background: #f0fff4;
      color: #276749;
      border: 1px solid #c6f6d5;
      border-radius: 5px;
      padding: 0.15rem 0.5rem;
      font-size: 0.8rem;
      font-weight: 600;
    }}
    /* ── Rating area ─────────────────────────────────────────────────── */
    .rating-section {{
      background: #fafafa;
      border: 1px dashed #cbd5e0;
      border-radius: 8px;
      padding: 1rem 1.2rem;
    }}
    .rating-row {{
      display: flex;
      align-items: center;
      gap: 1.2rem;
      margin-bottom: 0.6rem;
    }}
    .rating-row:last-child {{ margin-bottom: 0; }}
    .rating-label {{ font-size: 0.9rem; min-width: 190px; }}
    .stars {{ display: flex; gap: 0.3rem; }}
    .stars input[type=radio] {{ display: none; }}
    .stars label {{
      font-size: 1.5rem;
      color: #cbd5e0;
      cursor: pointer;
      transition: color .15s;
    }}
    .stars input[type=radio]:checked ~ label,
    .stars label:hover,
    .stars label:hover ~ label {{ color: #f6ad55; }}
    /* Reverse sibling hack for CSS-only stars */
    .stars {{ direction: rtl; }}
    .stars label:hover,
    .stars label:hover ~ label {{ color: #f6ad55; }}
    /* ── Footer ──────────────────────────────────────────────────────── */
    footer {{
      text-align: center;
      color: #a0aec0;
      font-size: 0.82rem;
      margin-top: 3rem;
    }}
  </style>
</head>
<body>
  <h1>ASBRS Human Evaluation Sheet</h1>
  <p class="subtitle">
    Agentic Session-Based Recommender System — Qualitative Assessment
  </p>

  {session_blocks}

  <footer>Generated by ASBRS HumanEvalExporter &bull; {len(sessions)} sessions sampled</footer>
</body>
</html>"""

    def _render_session_block(
        self,
        number: int,
        session: List[str],
        recs: List[RecommendationOutput],
        intent: IntentResult,
    ) -> str:
        """Render a single session card as an HTML string.

        Args:
            number:  1-based session number used in the heading.
            session: List of ASIN / title strings for browsed items.
            recs:    Top recommendations produced by ASBRS.
            intent:  Inferred intent for this session.

        Returns:
            HTML string for this session card.
        """
        browsing_items = "\n".join(
            f"      <li>{self._escape(item)}</li>"
            for item in session
        )

        keywords_html = ""
        if intent.keywords:
            badges = " ".join(
                f'<span class="keyword-badge">{self._escape(kw)}</span>'
                for kw in intent.keywords
            )
            keywords_html = f'<div class="keywords">{badges}</div>'

        top_recs = recs[:_TOP_N_RECS]
        rec_rows = "\n".join(
            f"""        <tr>
          <td>{rec.rank}</td>
          <td>{self._escape(rec.item_title)}</td>
          <td><span class="score-badge">{rec.final_score:.4f}</span></td>
          <td>{self._escape(rec.explanation)}</td>
        </tr>"""
            for rec in top_recs
        )

        # Unique prefix ensures no radio-group collisions between sessions.
        uid = f"s{number}"
        rel_stars = self._star_field(f"{uid}_rel", "relevance")
        exp_stars = self._star_field(f"{uid}_exp", "explanation")

        return f"""  <div class="session-card">
    <div class="session-header">Session #{number}</div>

    <div class="section-block">
      <div class="section-label">Browsing History ({len(session)} items)</div>
      <ol class="browsing-list">
{browsing_items}
      </ol>
    </div>

    <div class="section-block">
      <div class="section-label">Inferred Intent</div>
      <div class="intent-box">
        <div class="summary">"{self._escape(intent.intent_summary)}"</div>
        <div class="meta">Confidence: {intent.confidence:.2f}</div>
        {keywords_html}
      </div>
    </div>

    <div class="section-block">
      <div class="section-label">Top Recommendations</div>
      <table class="recs-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Item</th>
            <th>Score</th>
            <th>Explanation</th>
          </tr>
        </thead>
        <tbody>
{rec_rows}
        </tbody>
      </table>
    </div>

    <div class="section-block">
      <div class="section-label">Ratings</div>
      <div class="rating-section">
        <div class="rating-row">
          <span class="rating-label">Relevance (1–5)</span>
          {rel_stars}
        </div>
        <div class="rating-row">
          <span class="rating-label">Explanation Quality (1–5)</span>
          {exp_stars}
        </div>
      </div>
    </div>
  </div>"""

    @staticmethod
    def _star_field(name: str, label: str) -> str:
        """Build a CSS-only 5-star radio group.

        Args:
            name:  Unique name attribute for the radio inputs.
            label: Accessible label (used in aria attributes).

        Returns:
            HTML string for the star field.
        """
        stars = "".join(
            f'<input type="radio" id="{name}_{i}" name="{name}" value="{i}" />'
            f'<label for="{name}_{i}" title="{i} star">&#9733;</label>'
            for i in range(5, 0, -1)
        )
        return f'<div class="stars" role="radiogroup" aria-label="{label}">{stars}</div>'

    @staticmethod
    def _escape(text: str) -> str:
        """HTML-escape a string.

        Args:
            text: Raw string.

        Returns:
            HTML-safe string.
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
