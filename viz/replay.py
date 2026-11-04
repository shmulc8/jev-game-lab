"""Build read-only, self-contained replay pages. Never imports or calls API clients.

Run: python3 -m viz.replay
"""
import html
import hashlib
import base64
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORDINGS = HERE / "recordings"

STYLE = """*{box-sizing:border-box}body{margin:0;background:#090e19;color:#e1ebf5;font:16px system-ui,sans-serif}main{max-width:1040px;margin:auto;padding:20px 24px}a{color:#6bdaca}h1{font-size:1.8rem;margin:14px 0}h2{font-size:1.1rem;margin-top:0}small{color:#b3c4d9}.layout{display:flex;gap:32px;align-items:start;flex-wrap:wrap}.player{max-width:100%}canvas{display:block;background:#101b2b;width:auto;height:auto;max-height:58vh;max-width:100%;object-fit:contain;border:1px solid #64748b}section{flex:1;min-width:250px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#142135;padding:14px;font-size:12px;max-height:350px;overflow:auto}button,input{margin:8px 8px 8px 0}button{padding:8px 14px;background:#6bdaca;color:#090e19;border:0;border-radius:6px;font-weight:600}#position{display:block;width:100%;margin-top:0}.notice{line-height:1.5;color:#b3c4d9;font-size:14px;max-width:850px;margin-bottom:22px}.cards{display:grid;grid-template-columns:repeat(2,minmax(100px,1fr));gap:10px;margin:14px 0 20px}.card{background:#142135;border:1px solid #263c54;border-radius:8px;padding:14px}.card span{display:block;color:#b3c4d9;font-size:12px;margin-bottom:6px}.card strong{font-size:1.3rem}.links{list-style:none;padding:0;display:grid;gap:12px}.links a{display:block;text-decoration:none;padding:20px;border:1px solid #263c54;border-radius:8px;background:#142135}.links a:hover{border-color:#6bdaca}details{padding:12px 0;border-top:1px solid #263c54}summary{cursor:pointer;color:#b3c4d9}@media(max-width:600px){main{padding:16px}.layout{gap:20px}.player{margin:auto}canvas{max-height:52vh}.notice{font-size:12px}h1{font-size:1.5rem}}"""
JS = r"""const recording=JSON.parse(document.getElementById('recording').textContent),frames=recording.frames||[],canvas=document.getElementById('board'),ctx=canvas.getContext('2d'),slider=document.getElementById('position');let index=0,timer=null;
function render(){const frame=frames[index],rows=frame?.board?.rows||[],height=rows.length,width=rows[0]?.length||1,cell=Math.min(canvas.width/width,canvas.height/(height||1));ctx.fillStyle='#101b2b';ctx.fillRect(0,0,canvas.width,canvas.height);for(let y=0;y<height;y++)for(let x=0;x<width;x++){const value=rows[y][x];ctx.fillStyle=value==='@'?'#e1efac':value==='*'?'#ffb347':value==='#'?'#6bdaca':'#17263a';ctx.fillRect(x*cell+1,y*cell+1,Math.max(1,cell-2),Math.max(1,cell-2))}slider.value=index;document.getElementById('label').textContent=frames.length?'Frame '+(index+1)+' / '+frames.length:'No frames';document.getElementById('details').textContent=JSON.stringify(frame||{},null,2)}
function pause(){clearInterval(timer);timer=null;document.getElementById('play').textContent='Play'}slider.max=Math.max(0,frames.length-1);slider.oninput=()=>{pause();index=Number(slider.value);render()};document.getElementById('play').onclick=()=>{if(timer){pause();return}if(index>=frames.length-1)index=0;document.getElementById('play').textContent='Pause';timer=setInterval(()=>{if(index>=frames.length-1){pause();return}index++;render()},300);render()};render();"""


def recording_label(recording):
    agent = recording.get("agent", "unknown")
    mode = recording.get("mode", "")
    names = {"jev": "JEV", "oracle": "Oracle code baseline", "flood": "Flood-fill code baseline"}
    detail = f" · {mode.title()} mode" if agent == "jev" and mode else ""
    return f"{names.get(agent, agent.title())}{detail} · Seed {recording.get('seed', 0)}"


def summary_cards(recording):
    """Display historical facts; derive latency only from positive recorded samples."""
    summary = recording.get("summary", {})
    samples = []
    for frame in recording.get("frames", []):
        value = (frame.get("answer") or {}).get("latency_ms")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
            samples.append(value)
    samples.sort()
    def latency(fraction):
        if not samples:
            return "Not recorded"
        position = (len(samples) - 1) * fraction
        lo, hi = math.floor(position), math.ceil(position)
        value = samples[lo] + (samples[hi] - samples[lo]) * (position - lo)
        return f"{value:,.0f} ms"
    return [("Moves", str(summary.get("moves", len(recording.get("frames", []))))),
            ("Food eaten" if recording["game"] == "snake" else "Lines cleared", str(summary.get("lines_or_food", "Not recorded"))),
            ("Model requests", str(summary.get("requests", "Not recorded"))),
            ("Outcome", str(summary.get("outcome", "Not recorded")).replace("_", " ").title()),
            ("Recorded latency · p50", latency(.5)),
            ("Recorded latency · p95", latency(.95))]


def render_recording(recording):
    game = recording["game"]
    label = recording["game"].title() + " · " + recording_label(recording)
    title = html.escape(label.upper())
    data = json.dumps(recording, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    summary = html.escape(json.dumps(recording.get("summary", {}), indent=2))
    cards = "".join(f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>' for label, value in summary_cards(recording))
    decision_note = "JEV by TypeSafe selects from candidates computed by code."
    if game == "tetris":
        decision_note += " Rich mode uses a code-ranked candidate shortlist."
    dimensions = 'width="400" height="800"' if game == "tetris" else 'width="600" height="600"'
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} | JEV Game Lab</title><style>{STYLE}</style><main><a href="../{game}.html">← {game.title()} recordings</a><h1>{title}</h1><p class="notice">Recorded experiment · read-only playback · seed {int(recording.get("seed", 0))}. {decision_note} Baselines run entirely in code. Recorded costs and latency are historical measurements.</p><div class="layout"><div class="player"><canvas id="board" {dimensions} aria-label="Recorded {game} board"></canvas><div><button id="play" type="button">Play</button><span id="label"></span></div><input id="position" type="range" min="0" value="0" aria-label="Replay frame"></div><section><h2>Run results</h2><div class="cards">{cards}</div><p class="notice">Latency percentiles use positive latency samples from recorded model responses. Playback runs at a fixed 300 ms per frame.</p><details><summary>Original recorded summary</summary><pre>{summary}</pre></details><details><summary>Frame, request and response</summary><pre id="details"></pre></details></section></div></main><script id="recording" type="application/json">{data}</script><script>{JS}</script></html>'''


def index_page(title, links):
    items = "".join(f'<li><a href="{html.escape(path, quote=True)}">{html.escape(label)}</a></li>' for path, label in links)
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{STYLE}</style><main><a href="index.html">All recordings</a><h1>{html.escape(title)}</h1><p>Offline recordings of JEV by TypeSafe and code baselines. Playback never calls an API.</p><ul class="links">{items}</ul></main></html>'


def build(output=None):
    output = Path(output) if output is not None else HERE
    (output / "replays").mkdir(parents=True, exist_ok=True)
    groups = {"snake": [], "tetris": []}
    generated = []
    for source in sorted(RECORDINGS.glob("*.json")):
        if source.name == "index.json":
            continue
        recording = json.loads(source.read_text())
        target = output / "replays" / f"{source.stem}.html"
        target.write_text(render_recording(recording))
        groups[recording["game"]].append((f"replays/{target.name}", recording_label(recording)))
        generated.append(target)
    for game, links in groups.items():
        target = output / f"{game}.html"
        target.write_text(index_page(f"{game.title()} recordings", links))
        generated.append(target)
    target = output / "index.html"
    target.write_text(index_page("JEV Game Lab recordings", [(f"{game}.html", f"{game.title()} recordings ({len(links)})") for game, links in groups.items()]))
    generated.append(target)
    digest = lambda text: base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode("ascii")
    policy = ("default-src 'self'; script-src 'self' 'sha256-" + digest(JS) + "'; "
              "style-src 'self' 'sha256-" + digest(STYLE) + "'; img-src 'self' data:; "
              "connect-src 'none'; object-src 'none'; base-uri 'self'; "
              "frame-ancestors 'none'; form-action 'none'")
    (output / "_headers").write_text("/*\n  Content-Security-Policy: " + policy + "\n"
        "  X-Content-Type-Options: nosniff\n  X-Frame-Options: DENY\n"
        "  Referrer-Policy: strict-origin-when-cross-origin\n"
        "  Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=()\n", encoding="utf-8")
    return generated


if __name__ == "__main__":
    for path in build():
        print(path.relative_to(HERE))
