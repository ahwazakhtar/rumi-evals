"""Server-rendered inline-SVG charts.

No build step, no CDN, no JavaScript required — every chart is a static <svg>
string that inherits its colours from CSS custom properties defined in
``static/style.css`` (so it adapts to light/dark automatically). Tables always
accompany the charts in the templates, so if SVG fails the data still reads.

Design follows the repo's dataviz palette: fixed status colours (good/critical/
warning) and a fixed categorical order for models. Marks are thin, data-ends are
rounded, reference lines are dashed and labelled.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field


@dataclass
class Row:
    label: str
    value: float
    color: str = "var(--series-1)"
    display: str | None = None      # override the value label text
    sub: str | None = None          # small secondary label under the row label


@dataclass
class Ref:
    value: float
    label: str
    color: str = "var(--muted)"


@dataclass
class BarChart:
    rows: list[Row]
    vmin: float
    vmax: float
    refs: list[Ref] = field(default_factory=list)
    diverging: bool = False
    unit: str = ""
    decimals: int = 2
    label_w: int = 150
    chart_w: int = 560
    bar_h: int = 22
    gap: int = 12
    pad_top: int = 22
    pad_bottom: int = 30
    pad_right: int = 58

    # ---- geometry helpers -------------------------------------------------
    def _x(self, v: float) -> float:
        v = max(self.vmin, min(self.vmax, v))
        frac = (v - self.vmin) / (self.vmax - self.vmin) if self.vmax != self.vmin else 0
        return self.label_w + frac * self.chart_w

    def _fmt(self, r: Row) -> str:
        if r.display is not None:
            return r.display
        return f"{r.value:.{self.decimals}f}{self.unit}"

    # ---- render -----------------------------------------------------------
    def svg(self) -> str:
        n = len(self.rows)
        total_w = self.label_w + self.chart_w + self.pad_right
        total_h = self.pad_top + n * (self.bar_h + self.gap) + self.pad_bottom
        baseline_v = 0.0 if (self.vmin <= 0 <= self.vmax) else self.vmin
        x0 = self._x(baseline_v)

        parts: list[str] = [
            f'<svg viewBox="0 0 {total_w} {total_h}" width="100%" '
            f'preserveAspectRatio="xMinYMin meet" role="img" class="chart-svg" '
            f'font-family="system-ui,-apple-system,Segoe UI,sans-serif">'
        ]

        # plot frame baseline
        plot_top = self.pad_top - 6
        plot_bottom = self.pad_top + n * (self.bar_h + self.gap) - self.gap + 6
        parts.append(
            f'<line x1="{x0:.1f}" y1="{plot_top}" x2="{x0:.1f}" y2="{plot_bottom}" '
            f'stroke="var(--axis)" stroke-width="1"/>'
        )

        # reference lines
        for ref in self.refs:
            rx = self._x(ref.value)
            parts.append(
                f'<line x1="{rx:.1f}" y1="{plot_top - 8}" x2="{rx:.1f}" y2="{plot_bottom}" '
                f'stroke="{ref.color}" stroke-width="1.5" stroke-dasharray="4 3"/>'
            )
            parts.append(
                f'<text x="{rx:.1f}" y="{plot_top - 11}" text-anchor="middle" '
                f'font-size="10.5" fill="{ref.color}" font-weight="600">'
                f'{html.escape(ref.label)}</text>'
            )

        # bars
        y = self.pad_top
        for r in self.rows:
            bx = self._x(r.value)
            left = min(x0, bx)
            w = abs(bx - x0)
            w = max(w, 1.5)
            parts.append(
                f'<rect x="{left:.1f}" y="{y}" width="{w:.1f}" height="{self.bar_h}" '
                f'rx="3" ry="3" fill="{r.color}"/>'
            )
            # row label (left gutter)
            label_y = y + self.bar_h / 2 + (0 if not r.sub else -3)
            parts.append(
                f'<text x="{self.label_w - 10}" y="{label_y:.1f}" text-anchor="end" '
                f'font-size="12" fill="var(--text-secondary)" dominant-baseline="middle" '
                f'font-weight="500">{html.escape(r.label)}</text>'
            )
            if r.sub:
                parts.append(
                    f'<text x="{self.label_w - 10}" y="{label_y + 12:.1f}" text-anchor="end" '
                    f'font-size="9.5" fill="var(--muted)" dominant-baseline="middle">'
                    f'{html.escape(r.sub)}</text>'
                )
            # value label at bar end
            val = self._fmt(r)
            if bx >= x0:
                vx, anchor = bx + 6, "start"
            else:
                vx, anchor = bx - 6, "end"
            parts.append(
                f'<text x="{vx:.1f}" y="{y + self.bar_h / 2:.1f}" text-anchor="{anchor}" '
                f'font-size="11.5" fill="var(--text-primary)" dominant-baseline="middle" '
                f'font-weight="600" style="font-variant-numeric:tabular-nums">{html.escape(val)}</text>'
            )
            y += self.bar_h + self.gap

        # x-axis min/max ticks
        for tick_v in (self.vmin, self.vmax):
            tx = self._x(tick_v)
            parts.append(
                f'<text x="{tx:.1f}" y="{plot_bottom + 16}" text-anchor="middle" '
                f'font-size="10" fill="var(--muted)" style="font-variant-numeric:tabular-nums">'
                f'{tick_v:.{self.decimals}f}{self.unit}</text>'
            )

        parts.append("</svg>")
        return "".join(parts)


# Fixed categorical colours for the study models (dataviz slot order).
MODEL_COLORS = {
    "gpt-5.1": "var(--series-1)",
    "minimaxai/minimax-m2.7": "var(--series-2)",
    "deepseek-ai/deepseek-v4-pro": "var(--series-3)",
    "kimi": "var(--series-5)",
    "mistral": "var(--series-6)",
    "nemotron": "var(--series-8)",
}


def model_color(name: str) -> str:
    for key, col in MODEL_COLORS.items():
        if key in name:
            return col
    return "var(--series-1)"
