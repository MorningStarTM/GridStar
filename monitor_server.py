"""
Live status + remote stop for SafetyDataGenerator runs.

Run alongside data_main.py (separate terminal/tmux pane):
    python monitor_server.py

Then open in any browser:
    http://<EC2_PUBLIC_IP>:8000

The page auto-refreshes every 3s and shows the generator's last-written
status.json. The Stop button writes monitor/STOP, which generator.py checks
at the top of every episode loop and exits cleanly on.
"""

import json
import os

from flask import Flask, jsonify, render_template_string

MONITOR_DIR  = "monitor"
STATUS_PATH  = os.path.join(MONITOR_DIR, "status.json")
STOP_PATH    = os.path.join(MONITOR_DIR, "STOP")

os.makedirs(MONITOR_DIR, exist_ok=True)

app = Flask(__name__)

PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>GridStar — Data Generation Monitor</title>
<meta http-equiv="refresh" content="3">
<style>
  body { font-family: -apple-system, sans-serif; background:#0f172a; color:#e2e8f0; padding:2rem; }
  .card { background:#1e293b; border-radius:12px; padding:1.5rem 2rem; max-width:560px; margin:auto; }
  h1 { font-size:1.1rem; color:#94a3b8; font-weight:600; margin-bottom:1rem; }
  .row { display:flex; justify-content:space-between; padding:0.5rem 0; border-bottom:1px solid #334155; }
  .row:last-child { border-bottom:none; }
  .label { color:#94a3b8; }
  .value { font-weight:600; }
  .safe { color:#10b981; }
  .unsafe { color:#ef4444; }
  button { margin-top:1.5rem; width:100%; padding:0.8rem; border:none; border-radius:8px;
           background:#ef4444; color:white; font-weight:600; font-size:1rem; cursor:pointer; }
  button:hover { background:#dc2626; }
  .stopped { color:#f59e0b; text-align:center; margin-top:1rem; }
  .empty { color:#64748b; text-align:center; padding:2rem; }
</style>
</head>
<body>
<div class="card">
  <h1>GridStar Data Generation</h1>
  {% if status %}
    <div class="row"><span class="label">Strategy</span><span class="value">{{ status.strategy }}</span></div>
    <div class="row"><span class="label">Episode</span><span class="value">{{ status.episode }}{% if status.range %} / {{ status.range[1] - 1 }}{% endif %}</span></div>
    {% if status.line_id is defined %}
    <div class="row"><span class="label">Line ID</span><span class="value">{{ status.line_id }}</span></div>
    {% endif %}
    <div class="row"><span class="label">Samples</span><span class="value">{{ status.n_samples }}</span></div>
    <div class="row"><span class="label">Safe</span><span class="value safe">{{ status.n_safe }}</span></div>
    <div class="row"><span class="label">Unsafe</span><span class="value unsafe">{{ status.n_unsafe }}</span></div>
    <div class="row"><span class="label">Safe rate</span><span class="value">{{ "%.1f"|format(status.safe_rate * 100) }}%</span></div>
    <div class="row"><span class="label">Last update</span><span class="value">{{ status.updated_at }}</span></div>
  {% else %}
    <div class="empty">No status yet — waiting for the generator to save its first episode.</div>
  {% endif %}

  {% if stopped %}
    <div class="stopped">Stop requested — generator will halt before its next episode.</div>
  {% else %}
    <form method="post" action="/stop">
      <button type="submit">Stop Generation</button>
    </form>
  {% endif %}
</div>
</body>
</html>
"""


@app.route("/")
def index():
    status = None
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH) as f:
            status = json.load(f)
    stopped = os.path.exists(STOP_PATH)
    return render_template_string(PAGE, status=status, stopped=stopped)


@app.route("/status")
def status_json():
    if not os.path.exists(STATUS_PATH):
        return jsonify({}), 404
    with open(STATUS_PATH) as f:
        return jsonify(json.load(f))


@app.route("/stop", methods=["POST"])
def stop():
    with open(STOP_PATH, "w") as f:
        f.write("stop requested via dashboard\n")
    return index()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
