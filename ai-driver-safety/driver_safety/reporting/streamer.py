from __future__ import annotations

import asyncio
import json
import threading
from typing import Any
import fastapi
from fastapi.responses import HTMLResponse
from uvicorn import Config, Server

from driver_safety.core.models import ProcessedFrame

_latest_data: dict[str, Any] = {}
_data_lock = threading.Lock()

app = fastapi.FastAPI()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serves the real-time glassmorphic dark dashboard straight to the browser."""
    return DASHBOARD_HTML

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: fastapi.WebSocket):
    """Streams live telemetry packets to the frontend template."""
    await websocket.accept()
    try:
        while True:
            with _data_lock:
                payload = _latest_data.copy()
            if payload:
                await websocket.send_json(payload)
            await asyncio.sleep(0.04)  # ~25 FPS telemetry stream
    except fastapi.websockets.WebSocketDisconnect:
        pass

def push_frame_telemetry(processed: ProcessedFrame) -> None:
    """Call this inside your frame loop to update live metrics."""
    global _latest_data
    payload = {
        "state": processed.state.value,
        "risk_score": float(processed.risk_score),
        "latency_ms": float(processed.latency_ms),
        "signals": {k: float(v) for k, v in processed.signals.items()}
    }
    with _data_lock:
        _latest_data = payload

def start_streamer_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Spins up the FastAPI server in a background thread."""
    def run():
        config = Config(app=app, host=host, port=port, log_level="error")
        server = Server(config)
        server.run()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()

# Standalone Glassmorphic Dashboard Template
DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DriveSafe-AI // Live Telemetry</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { background-color: #080c0d; color: #e2e8f0; font-family: system-ui, sans-serif; }
    .glass { background: rgba(17, 24, 25, 0.6); backdrop-filter: blur(12px); border: 1px solid #1c2e30; }
  </style>
</head>
<body class="p-8">
  <div class="max-w-6xl mx-auto">
    <header class="flex justify-between items-center border-b border-[#141f20] pb-6 mb-8">
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <span id="status-dot" class="w-2.5 h-2.5 rounded-full bg-red-500"></span>
          DRIVESAFE-AI // REAL-TIME MONITORING
        </h1>
      </div>
      <div id="connection-badge" class="px-4 py-1.5 rounded-full border text-xs font-mono bg-red-950/30 border-red-500/30 text-red-400">
        DISCONNECTED
      </div>
    </header>

    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      <div id="state-card" class="glass p-6 rounded-xl transition-all duration-300">
        <span class="text-xs uppercase tracking-wider text-gray-400 block mb-2">Driver Assessment</span>
        <strong id="live-state" class="text-2xl font-extrabold uppercase text-gray-500">AWAITING LINK...</strong>
      </div>
      <div class="glass p-6 rounded-xl">
        <span class="text-xs uppercase tracking-wider text-gray-400 block mb-2">Risk Factor</span>
        <strong id="live-risk" class="text-3xl font-bold text-white font-mono">0.00</strong>
      </div>
      <div class="glass p-6 rounded-xl">
        <span class="text-xs uppercase tracking-wider text-gray-400 block mb-2">Inference Latency</span>
        <strong id="live-latency" class="text-3xl font-bold text-cyan-400 font-mono">0.0 ms</strong>
      </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div class="lg:col-span-2 glass p-6 rounded-xl">
        <h2 class="text-sm font-semibold uppercase tracking-wider text-white mb-4">Risk Progression Timeline</h2>
        <div class="h-64"><canvas id="liveChart"></canvas></div>
      </div>
      <div class="glass p-6 rounded-xl">
        <h2 class="text-sm font-semibold uppercase tracking-wider text-white mb-6">Extraction Channels</h2>
        <div id="signals-container" class="space-y-4"></div>
      </div>
    </div>
  </div>

  <script>
    const ctx = document.getElementById('liveChart').getContext('2d');
    const history = Array(30).fill(0);
    const labels = Array(30).fill('');
    
    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          data: history,
          borderColor: '#06b6d4',
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          backgroundColor: 'rgba(6, 182, 212, 0.05)',
          tension: 0.3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { min: 0, max: 1 }, x: { display: false } }
      }
    });

    const ws = new WebSocket("ws://" + window.location.host + "/ws/telemetry");
    const badge = document.getElementById('connection-badge');
    const dot = document.getElementById('status-dot');

    ws.onopen = () => {
      badge.textContent = "LIVE LINK STABLE";
      badge.className = "px-4 py-1.5 rounded-full border text-xs font-mono bg-emerald-950/30 border-emerald-500/30 text-emerald-400";
      dot.className = "w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse";
    };

    ws.onclose = () => {
      badge.textContent = "DISCONNECTED";
      badge.className = "px-4 py-1.5 rounded-full border text-xs font-mono bg-red-950/30 border-red-500/30 text-red-400";
      dot.className = "w-2.5 h-2.5 rounded-full bg-red-500";
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      // Update basic cards
      document.getElementById('live-state').textContent = data.state.replace('_', ' ');
      document.getElementById('live-risk').textContent = data.risk_score.toFixed(2);
      document.getElementById('live-latency').textContent = data.latency_ms.toFixed(1) + " ms";

      // Toggle alert visual warnings
      const stateCard = document.getElementById('state-card');
      if (data.risk_score > 0.45 || data.state !== 'attentive') {
        stateCard.className = "glass p-6 rounded-xl bg-red-950/20 border-red-500/40";
        document.getElementById('live-state').className = "text-2xl font-extrabold uppercase text-red-400 animate-pulse";
      } else {
        stateCard.className = "glass p-6 rounded-xl";
        document.getElementById('live-state').className = "text-2xl font-extrabold uppercase text-emerald-400";
      }

      // Update Chart
      history.push(data.risk_score);
      history.shift();
      chart.update('none');

      // Update bars dynamically
      const container = document.getElementById('signals-container');
      container.innerHTML = '';
      Object.entries(data.signals).forEach(([name, val]) => {
        container.innerHTML += `
          <div class="space-y-1">
            <div class="flex justify-between text-xs">
              <span class="capitalize text-gray-400">${name.replace('_', ' ')}</span>
              <span class="text-cyan-400 font-mono">${(val * 100).toFixed(0)}%</span>
            </div>
            <div class="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div class="h-full ${val > 0.5 ? 'bg-red-500' : 'bg-cyan-500'}" style="width: ${val * 100}%"></div>
            </div>
          </div>`;
      });
    };
  </script>
</body>
</html>
"""