"""
prova_mouse.py — Navigazione latent space con mouse → audio in Max/MSP

Avvia un server web locale. Apri http://localhost:8090 nel browser.
Muovi il mouse sulla mappa → Python manda OSC a Max che suona il file.

Messaggi OSC mandati a Max (porta 7400):
  /latent/play  <filepath_assoluto>   → carica e suona il file
  /latent/pos   <x_norm> <y_norm>     → posizione 0.0-1.0
  /latent/cell  <gi> <gj>             → indici cella griglia
  /latent/stop  1                     → ferma il suono

Uso:
  pip install aiohttp python-osc
  python connessioneMAX/prova_mouse2.py --grid_dir ./grid_audio \
                                        --umap_points ./umap_points.json

Argomenti opzionali:
  --grid_dir     cartella con i .wav e grid_manifest.json  [./grid_128iter_in]
  --umap_points  file JSON con i punti UMAP                [./umap_points.json]
  --port         porta server web                          [8090]
  --max_host     host Max/MSP                              [127.0.0.1]
  --max_port     porta UDP verso Max                       [7400]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from aiohttp import web
from pythonosc import udp_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Stato globale ──────────────────────────────────────────────────────────────
MANIFEST    = None
GRID_DIR    = None
OSC_CLIENT  = None
ALL_POINTS  = None   # lista [[x, y], ...]
ALL_INSTR   = None   # lista ["bass", "fagotto", ...]
INST_LIST   = None   # lista sorted dei tipi unici


def setup(grid_dir, umap_points_path, max_host, max_port):
    global MANIFEST, GRID_DIR, OSC_CLIENT, ALL_POINTS, ALL_INSTR, INST_LIST

    GRID_DIR = os.path.abspath(grid_dir)

    # Manifest griglia
    manifest_path = os.path.join(GRID_DIR, "grid_manifest.json")
    with open(manifest_path) as f:
        MANIFEST = json.load(f)
    print(f"Griglia {MANIFEST['grid_size']}×{MANIFEST['grid_size']} caricata")
    print(f"  x: [{MANIFEST['x_min']:.3f}, {MANIFEST['x_max']:.3f}]")
    print(f"  y: [{MANIFEST['y_min']:.3f}, {MANIFEST['y_max']:.3f}]")

    # Punti UMAP pre-calcolati
    with open(umap_points_path) as f:
        umap_data = json.load(f)
    ALL_POINTS = umap_data['points']
    ALL_INSTR  = umap_data['instruments']
    INST_LIST  = sorted(set(ALL_INSTR))
    print(f"Punti UMAP: {len(ALL_POINTS)} · strumenti: {INST_LIST}")

    OSC_CLIENT = udp_client.SimpleUDPClient(max_host, max_port)
    print(f"OSC → {max_host}:{max_port}")


# ── Lookup cella ──────────────────────────────────────────────────────────────

def coords_to_cell(x, y):
    """
    Coordinate UMAP (x, y) → (gi, gj, filepath, x_norm, y_norm).
    Ritorna None se la cella non esiste.
    """
    m         = MANIFEST
    grid_size = m['grid_size']

    x_norm = (x - m['x_min']) / (m['x_max'] - m['x_min'])
    y_norm = (y - m['y_min']) / (m['y_max'] - m['y_min'])

    gi = int(np.clip(x_norm * grid_size, 0, grid_size - 1))
    gj = int(np.clip(y_norm * grid_size, 0, grid_size - 1))

    cell = m['cells'].get(f"{gi},{gj}")
    if cell is None:
        return None

    filepath = os.path.join(GRID_DIR, cell['file'])
    return gi, gj, filepath, float(x_norm), float(y_norm)


# ── API handlers ───────────────────────────────────────────────────────────────

async def handle_data(request):
    """Manda tutti i dati necessari alla pagina HTML — identico a demo_mouse."""
    return web.json_response({
        'points':      ALL_POINTS,
        'instruments': ALL_INSTR,
        'inst_list':   INST_LIST,
        'manifest': {
            'grid_size': MANIFEST['grid_size'],
            'x_min': MANIFEST['x_min'],
            'x_max': MANIFEST['x_max'],
            'y_min': MANIFEST['y_min'],
            'y_max': MANIFEST['y_max'],
        }
    })


async def handle_navigate(request):
    """
    Riceve {x, y} in coordinate UMAP → lookup cella → manda OSC a Max.
    """
    data = await request.json()
    x, y = float(data['x']), float(data['y'])

    result = coords_to_cell(x, y)
    if result is None:
        return web.json_response({'error': 'cella non trovata'}, status=404)

    gi, gj, filepath, x_norm, y_norm = result

    # Strumento più vicino al punto del cursore (come demo_mouse)
    pts  = np.array(ALL_POINTS)
    idx  = int(np.argmin(np.linalg.norm(pts - np.array([x, y]), axis=1)))
    inst = ALL_INSTR[idx]

    # OSC verso Max
    OSC_CLIENT.send_message("/latent/play", filepath)
    OSC_CLIENT.send_message("/latent/pos",  [x_norm, y_norm])
    OSC_CLIENT.send_message("/latent/cell", [gi, gj])

    filename = os.path.basename(filepath)
    print(f"  [{gi:02d},{gj:02d}] {filename}  {inst}  pos=({x_norm:.3f}, {y_norm:.3f})")

    return web.json_response({
        'gi':         gi,
        'gj':         gj,
        'file':       filename,
        'instrument': inst,
        'x_norm':     x_norm,
        'y_norm':     y_norm,
    })


async def handle_stop(request):
    """Ferma il suono in Max."""
    OSC_CLIENT.send_message("/latent/stop", 1)
    return web.json_response({'ok': True})


async def handle_index(request):
    return web.Response(text=build_html(), content_type='text/html')


# ── HTML — identico a demo_mouse, senza audio browser, con badge OSC ──────────

def build_html():
    return """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>LATENT SPACE</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: #080808;
    color: #ccc;
    font-family: 'Courier New', monospace;
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100vh;
    overflow: hidden;
  }

  h1 {
    font-size: 13px;
    letter-spacing: 0.3em;
    color: #555;
    padding: 16px 0 4px;
    text-transform: uppercase;
  }

  #mode-badge {
    font-size: 10px; letter-spacing: 0.2em; color: #1a6b3a;
    background: #0a1f12; border: 1px solid #1a6b3a;
    padding: 2px 10px; margin-bottom: 6px;
  }

  #info-bar {
    font-size: 11px;
    color: #444;
    letter-spacing: 0.15em;
    margin-bottom: 10px;
    height: 16px;
    transition: color 0.3s;
  }

  #info-bar.active { color: #aaa; }

  #canvas-wrap {
    position: relative;
    cursor: crosshair;
  }

  canvas {
    display: block;
    border: 1px solid #1a1a1a;
  }

  #cursor-dot {
    position: absolute;
    width: 14px;
    height: 14px;
    border: 1.5px solid #fff;
    border-radius: 50%;
    pointer-events: none;
    transform: translate(-50%, -50%);
    display: none;
    mix-blend-mode: screen;
  }

  #cursor-dot::after {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    width: 4px; height: 4px;
    background: #fff;
    border-radius: 50%;
    transform: translate(-50%, -50%);
  }

  #footer {
    font-size: 10px;
    color: #2a2a2a;
    margin-top: 10px;
    letter-spacing: 0.1em;
  }

  #osc-status {
    font-size: 10px;
    color: #1a5c30;
    margin-top: 4px;
    letter-spacing: 0.1em;
  }
</style>
</head>
<body>

<h1>LATENT SPACE NAVIGATOR</h1>
<div id="mode-badge">● MOUSE MODE → AUDIO IN MAX/MSP</div>
<div id="info-bar">caricamento...</div>

<div id="canvas-wrap">
  <canvas id="c"></canvas>
  <div id="cursor-dot"></div>
</div>

<div id="footer">muovi il mouse sulla mappa · esci per fermare il suono</div>
<div id="osc-status">—</div>

<script>
const canvas    = document.getElementById('c');
const ctx       = canvas.getContext('2d');
const dot       = document.getElementById('cursor-dot');
const infoBar   = document.getElementById('info-bar');
const oscStatus = document.getElementById('osc-status');

const SIZE = Math.min(window.innerWidth - 40, window.innerHeight - 120);
canvas.width  = SIZE;
canvas.height = SIZE;

const PALETTE = [
  '#ff6b6b','#ffd93d','#6bcb77','#4d96ff','#ff6bdb',
  '#ff9f43','#48dbfb','#ff9ff3','#54a0ff','#5f27cd',
  '#00d2d3','#ff6348'
];

let points = [], instruments = [], instList = [], manifest = null;
let lastCell = null;
let throttleTimer = null;

// ── Carica dati dal server ────────────────────────────────────────────────────
async function loadData() {
  const res  = await fetch('/data');
  const data = await res.json();
  points      = data.points;
  instruments = data.instruments;
  instList    = data.inst_list;
  manifest    = data.manifest;
  infoBar.textContent = `${points.length} suoni · griglia ${manifest.grid_size}×${manifest.grid_size}`;
  drawMap();
}

// ── Disegna mappa ─────────────────────────────────────────────────────────────
function toCanvas(x, y) {
  const pad = 30;
  const cx = pad + (x - manifest.x_min) / (manifest.x_max - manifest.x_min) * (SIZE - pad*2);
  const cy = pad + (y - manifest.y_min) / (manifest.y_max - manifest.y_min) * (SIZE - pad*2);
  return [cx, cy];
}

function drawMap(highlightX=null, highlightY=null) {
  ctx.fillStyle = '#080808';
  ctx.fillRect(0, 0, SIZE, SIZE);

  // Griglia sottile
  ctx.strokeStyle = '#111';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 10; i++) {
    const x = 30 + i * (SIZE - 60) / 10;
    const y = 30 + i * (SIZE - 60) / 10;
    ctx.beginPath(); ctx.moveTo(x, 30); ctx.lineTo(x, SIZE-30); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(30, y); ctx.lineTo(SIZE-30, y); ctx.stroke();
  }

  // Punti dataset
  const instToI = {};
  instList.forEach((inst, i) => instToI[inst] = i);

  for (let i = 0; i < points.length; i++) {
    const [cx, cy] = toCanvas(points[i][0], points[i][1]);
    const color    = PALETTE[instToI[instruments[i]] % PALETTE.length];
    ctx.beginPath();
    ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = color + 'aa';
    ctx.fill();
  }

  // Punto corrente
  if (highlightX !== null) {
    const [hx, hy] = toCanvas(highlightX, highlightY);
    ctx.beginPath();
    ctx.arc(hx, hy, 10, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffffff44';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(hx, hy, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
  }
}

// ── Mouse → coordinate UMAP → navigate ───────────────────────────────────────
function canvasToUmap(px, py) {
  const pad = 30;
  const x   = manifest.x_min + (px - pad) / (SIZE - pad*2) * (manifest.x_max - manifest.x_min);
  const y   = manifest.y_min + (py - pad) / (SIZE - pad*2) * (manifest.y_max - manifest.y_min);
  return [x, y];
}

async function navigate(x, y) {
  const res  = await fetch('/navigate', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ x, y }),
  });
  const data = await res.json();
  if (data.error) return;

  const cellKey = `${data.gi},${data.gj}`;
  if (cellKey === lastCell) return;
  lastCell = cellKey;

  infoBar.textContent = `${data.instrument}  ·  cella (${data.gi},${data.gj})`;
  infoBar.classList.add('active');
  oscStatus.textContent = `→ OSC /latent/play  ${data.file}`;

  drawMap(x, y);
}

// ── Event listeners ───────────────────────────────────────────────────────────
canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const px   = e.clientX - rect.left;
  const py   = e.clientY - rect.top;

  dot.style.left    = px + 'px';
  dot.style.top     = py + 'px';
  dot.style.display = 'block';

  if (throttleTimer) return;
  throttleTimer = setTimeout(() => {
    throttleTimer = null;
    if (!manifest) return;
    const [x, y] = canvasToUmap(px, py);
    if (x < manifest.x_min || x > manifest.x_max ||
        y < manifest.y_min || y > manifest.y_max) return;
    navigate(x, y);
  }, 80);
});

canvas.addEventListener('mouseleave', () => {
  dot.style.display = 'none';
  infoBar.classList.remove('active');
  infoBar.textContent = '—';
  oscStatus.textContent = '—';
  lastCell = null;
  fetch('/stop', { method: 'POST' });
  if (manifest) drawMap();
});

// ── Avvio ─────────────────────────────────────────────────────────────────────
loadData();
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Latent space mouse → Max/MSP via OSC")
    parser.add_argument("--grid_dir",    default=str(PROJECT_ROOT / "grid_audio"))
    parser.add_argument("--umap_points", default=str(PROJECT_ROOT / "umap_points.json"))
    parser.add_argument("--port",        type=int, default=8090)
    parser.add_argument("--max_host",    default="127.0.0.1")
    parser.add_argument("--max_port",    type=int, default=7400)
    args = parser.parse_args()

    setup(args.grid_dir, args.umap_points, args.max_host, args.max_port)

    app = web.Application()
    app.router.add_get('/',          handle_index)
    app.router.add_get('/data',      handle_data)
    app.router.add_post('/navigate', handle_navigate)
    app.router.add_post('/stop',     handle_stop)

    print(f"\nApri nel browser: http://localhost:{args.port}")
    print("Muovi il mouse → Max suona.")
    print("Ctrl+C per fermare.\n")

    web.run_app(app, host='localhost', port=args.port, print=None)


if __name__ == "__main__":
    main()
