from __future__ import annotations

import csv
import json
from pathlib import Path

from jinja2 import Template

from driver_safety.core.models import DetectionEvent, SessionSummary


def export_run_artifacts(
    output_dir: str | Path,
    *,
    events: list[DetectionEvent],
    summary: SessionSummary,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events_path = output / "events.json"
    summary_path = output / "summary.json"
    csv_path = output / "events.csv"
    html_path = output / "report.html"
    
    events_path.write_text(
        json.dumps([event.to_dict() for event in events], indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    write_events_csv(csv_path, events)
    write_html_report(html_path, events=events, summary=summary)
    
    return {
        "events": events_path,
        "summary": summary_path,
        "csv": csv_path,
        "html": html_path,
    }


def write_events_csv(path: str | Path, events: list[DetectionEvent]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_id",
                "timestamp",
                "frame_index",
                "signal",
                "state",
                "score",
                "severity",
                "message",
            ],
        )
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "event_id": event.event_id,
                    "timestamp": f"{event.timestamp:.3f}",
                    "frame_index": event.frame_index,
                    "signal": event.signal,
                    "state": event.state.value,
                    "score": f"{event.score:.4f}",
                    "severity": event.severity.value,
                    "message": event.message,
                }
            )


def write_html_report(
    path: str | Path,
    *,
    events: list[DetectionEvent],
    summary: SessionSummary,
) -> None:
    rendered = Template(REPORT_TEMPLATE).render(
        events=[event.to_dict() for event in events],
        summary=summary.to_dict(),
    )
    Path(path).write_text(rendered, encoding="utf-8")


REPORT_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DriveSafe-AI // Session Analytics</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      color-scheme: dark;
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    body {
      margin: 0;
      background: #0d1112;
      color: #e6e8e8;
      padding-bottom: 60px;
    }
    main {
      max-width: 1200px;
      margin: 0 auto;
      padding: 40px 24px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #1f292b;
      padding-bottom: 24px;
      margin-bottom: 32px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #f6f8f8;
    }
    h2 {
      font-size: 18px;
      font-weight: 600;
      margin-top: 0;
      margin-bottom: 20px;
      color: #f6f8f8;
    }
    .muted {
      color: #8fa09b;
      font-size: 14px;
      margin: 4px 0 0 0;
    }
    .session-badge {
      background: #181e1f;
      border: 1px solid #2d3b3d;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 13px;
      font-family: monospace;
      color: #2ec57c;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .metric {
      background: rgba(24, 30, 31, 0.6);
      backdrop-filter: blur(12px);
      border: 1px solid #1f292b;
      border-radius: 12px;
      padding: 20px;
    }
    .metric span {
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #8fa09b;
      margin-bottom: 8px;
    }
    .metric strong {
      font-size: 28px;
      font-weight: 700;
      color: #f6f8f8;
    }
    .panel {
      background: rgba(24, 30, 31, 0.6);
      backdrop-filter: blur(12px);
      border: 1px solid #1f292b;
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
    }
    .chart-container {
      position: relative;
      height: 280px;
      width: 100%;
    }
    .table-container {
      overflow-x: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      text-align: left;
    }
    th {
      padding: 12px 16px;
      border-bottom: 1px solid #1f292b;
      color: #8fa09b;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 600;
    }
    td {
      padding: 14px 16px;
      border-bottom: 1px solid #161e1f;
      color: #cbd5e1;
      vertical-align: middle;
    }
    tr:hover td {
      background: rgba(255, 255, 255, 0.01);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
    }
    .badge.low { background: rgba(76, 190, 118, 0.15); color: #4cbe76; }
    .badge.medium { background: rgba(239, 139, 50, 0.15); color: #ef8b32; }
    .badge.high { background: rgba(238, 42, 42, 0.15); color: #ee2a2a; }
    
    @media (max-width: 768px) {
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      header { flex-direction: column; align-items: flex-start; gap: 16px; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>DriveSafe-AI Run Reports</h1>
      <p class="muted">Source Telemetry: {{ summary.source }}</p>
    </div>
    <div class="session-badge">SESSION ID: #{{ summary.session_id }}</div>
  </header>

  <section class="grid">
    <div class="metric"><span>Session Duration</span><strong>{{ summary.duration_seconds }}s</strong></div>
    <div class="metric"><span>Frames Analyzed</span><strong>{{ summary.processed_frames }}</strong></div>
    <div class="metric"><span>Max Unsafe Interval</span><strong>{{ summary.longest_unsafe_interval_seconds }}s</strong></div>
    <div class="metric"><span>Total Risk Alerts</span><strong>{{ events|length }}</strong></div>
  </section>

  <section class="panel">
    <h2>Continuous Risk Overview</h2>
    <div class="chart-container">
      <canvas id="riskChart"></canvas>
    </div>
  </section>

  <section class="panel">
    <h2>Logged Detection Incidents</h2>
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>Time Offset</th>
            <th>Trigger Signal</th>
            <th>Classification State</th>
            <th>Anomaly Score</th>
            <th>System Log Message</th>
          </tr>
        </thead>
        <tbody>
        {% for event in events[:120] %}
          <tr>
            <td style="font-family: monospace;">{{ "%.2f"|format(event.timestamp) }}s</td>
            <td><code style="color: #2ec57c;">{{ event.signal }}</code></td>
            <td>
              <span class="badge {% if event.severity == 'critical' or event.severity == 'high' %}high{% elif event.severity == 'medium' %}medium{% else %}low{% endif %}">
                {{ event.state }}
              </span>
            </td>
            <td style="font-family: monospace;">{{ "%.2f"|format(event.score) }}</td>
            <td>{{ event.message }}</td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </section>
</main>

<script>
  const timelineData = {{ summary.risk_timeline | tojson }};
  
  // Extract tracking datasets dynamically
  const timestamps = timelineData.map((_, idx) => ((idx * 3) / 30).toFixed(1) + 's');
  const scores = timelineData.map(point => point.risk_score || 0);

  const ctx = document.getElementById('riskChart').getContext('2d');
  
  // Neon Cyan gradient fill for presentation
  const gradient = ctx.createLinearGradient(0, 0, 0, 240);
  gradient.addColorStop(0, 'rgba(42, 238, 238, 0.25)');
  gradient.addColorStop(1, 'rgba(42, 238, 238, 0.0)');

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: timestamps,
      datasets: [{
        label: 'Risk Progression',
        data: scores,
        borderColor: '#2aeeee',
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        backgroundColor: gradient,
        fill: true,
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.03)' },
          ticks: { color: '#8fa09b', maxTicksLimit: 12 }
        },
        y: {
          min: 0,
          max: 1,
          grid: { color: 'rgba(255, 255, 255, 0.03)' },
          ticks: { color: '#8fa09b', stepSize: 0.25 }
        }
      }
    }
  });
</script>
</body>
</html>
"""