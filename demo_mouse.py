"""
demo_mouse.py — Navigazione real-time del latent space con mouse

Avvia un server web locale. Apri http://localhost:8080 nel browser.
Muovi il mouse sulla mappa → il suono cambia in tempo reale.

Uso:
  python demo_mouse.py --checkpoint ./checkpoints/ae_final.pt \
                       --data_dir   ./data_processed \
                       --grid_dir   ./grid_128iter

"""

import argparse
import json
import os
import numpy as np
import torch
import librosa
import umap
from pathlib import Path
from aiohttp import web

from model   import Autoencoder
from dataset import NSynthDataset, SAMPLE_RATE

ALL_Z      = None
ALL_META   = None
Z2D_CACHE  = None
MANIFEST   = None
GRID_DIR   = None


def setup(checkpoint_path, data_dir, grid_dir):
    global ALL_Z, ALL_META, Z2D_CACHE, MANIFEST, GRID_DIR

    GRID_DIR = os.path.abspath(grid_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt    = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config  = ckpt['config']
    ds_mean = ckpt['dataset_mean']
    ds_std  = ckpt['dataset_std']

    model = Autoencoder(latent_dim=config['latent_dim']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Modello caricato — epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")

    dataset = NSynthDataset(data_dir, cache=False, mean=ds_mean, std=ds_std)

    print("Calcolo coordinate latenti...")
    zs, metas = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            s    = dataset[i]
            spec = s['spectrogram'].unsqueeze(0).to(device)
            z    = model.encode(spec).cpu().numpy()[0]
            zs.append(z)
            metas.append({'instrument': s['instrument'], 'filename': s['filename']})
    ALL_Z    = np.array(zs)
    ALL_META = metas

    print("Calcolo UMAP...")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.2,
        metric='euclidean', random_state=42, low_memory=True,
    )
    reducer.fit(ALL_Z)
    Z2D_CACHE = reducer.transform(ALL_Z)
    print("UMAP completata")

    with open(os.path.join(grid_dir, "grid_manifest.json")) as f:
        MANIFEST = json.load(f)
    print(f"Griglia {MANIFEST['grid_size']}×{MANIFEST['grid_size']} caricata")



async def handle_data(request):
    """Manda tutti i dati necessari alla pagina HTML."""
    z2d = Z2D_CACHE.tolist()
    instruments = [m['instrument'] for m in ALL_META]
    inst_list   = sorted(set(instruments))
    inst_to_i   = {inst: i for i, inst in enumerate(inst_list)}

    return web.json_response({
        'points':     z2d,
        'instruments': instruments,
        'inst_list':  inst_list,
        'manifest':   {
            'grid_size': MANIFEST['grid_size'],
            'x_min': MANIFEST['x_min'],
            'x_max': MANIFEST['x_max'],
            'y_min': MANIFEST['y_min'],
            'y_max': MANIFEST['y_max'],
        }
    })


async def handle_navigate(request):
    """
    Riceve {x, y} → trova cella griglia → restituisce path audio.
    """
    data = await request.json()
    x, y = float(data['x']), float(data['y'])

    m         = MANIFEST
    grid_size = m['grid_size']
    gi = int(np.clip(
        (x - m['x_min']) / (m['x_max'] - m['x_min']) * grid_size,
        0, grid_size - 1
    ))
    gj = int(np.clip(
        (y - m['y_min']) / (m['y_max'] - m['y_min']) * grid_size,
        0, grid_size - 1
    ))

    cell = m['cells'].get(f"{gi},{gj}")
    if cell is None:
        return web.json_response({'error': 'cell not found'}, status=404)

    # Trova traccia più vicina
    distances = np.linalg.norm(Z2D_CACHE - np.array([x, y]), axis=1)
    idx       = int(np.argmin(distances))
    inst      = ALL_META[idx]['instrument']

    return web.json_response({
        'file':   f"/audio/{cell['file']}",
        'gi':     gi,
        'gj':     gj,
        'instrument': inst,
        'dist':   float(distances[idx]),
    })


async def handle_audio(request):
    """Serve i file audio dalla cartella grid."""
    filename = request.match_info['filename']
    filepath = os.path.join(GRID_DIR, filename)
    if not os.path.exists(filepath):
        return web.Response(status=404)
    return web.FileResponse(filepath)


async def handle_index(request):
    """Serve la pagina HTML."""
    html = build_html()
    return web.Response(text=html, content_type='text/html')


# ── HTML ───────────────────────────────────────────────────────────────────────

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
    padding: 16px 0 8px;
    text-transform: uppercase;
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

  #btn-toggle {
    margin-top: 10px;
    padding: 6px 24px;
    background: transparent;
    border: 1px solid #333;
    color: #555;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    letter-spacing: 0.2em;
    cursor: pointer;
    text-transform: uppercase;
    transition: border-color 0.2s, color 0.2s;
  }
  #btn-toggle:hover { border-color: #666; color: #aaa; }
  #btn-toggle.on    { border-color: #888; color: #ccc; }
</style>
</head>
<body>

<h1>LATENT SPACE NAVIGATOR</h1>
<div id="info-bar">caricamento...</div>

<div id="canvas-wrap">
  <canvas id="c"></canvas>
  <div id="cursor-dot"></div>
</div>

<div id="footer">muovi il mouse sulla mappa · esci per fermare il suono</div>
<button id="btn-toggle">SOUND OFF</button>

<script>
let soundEnabled = false;
const canvas  = document.getElementById('c');
const ctx     = canvas.getContext('2d');
const dot     = document.getElementById('cursor-dot');
const infoBar = document.getElementById('info-bar');
const wrap    = document.getElementById('canvas-wrap');

const SIZE = Math.min(window.innerWidth - 40, window.innerHeight - 120);
canvas.width  = SIZE;
canvas.height = SIZE;

const PALETTE = [
  '#ff6b6b','#ffd93d','#6bcb77','#4d96ff','#ff6bdb',
  '#ff9f43','#48dbfb','#ff9ff3','#54a0ff','#5f27cd',
  '#00d2d3','#ff6348'
];

let points = [], instruments = [], instList = [], manifest = null;
let currentAudio = null;
let nextFile = null;
let isPlaying = false;
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

// ── Audio ─────────────────────────────────────────────────────────────────────
function playFile(filepath) {
  if (currentAudio) {
    // Crossfade: fade out il corrente
    const old = currentAudio;
    fadeOut(old, 80, () => { old.pause(); old.src = ''; });
  }

  const audio = new Audio(filepath);
  audio.loop  = true;
  audio.volume = 0;
  audio.play().then(() => {
    fadeIn(audio, 80);
    currentAudio = audio;
  }).catch(e => console.warn('audio error:', e));
}

function fadeOut(audio, ms, cb) {
  const steps = 20;
  const dt    = ms / steps;
  const dv    = audio.volume / steps;
  let i = 0;
  const timer = setInterval(() => {
    audio.volume = Math.max(0, audio.volume - dv);
    if (++i >= steps) { clearInterval(timer); if (cb) cb(); }
  }, dt);
}

function fadeIn(audio, ms) {
  const steps = 20;
  const dt    = ms / steps;
  const dv    = 0.8 / steps;
  let i = 0;
  audio.volume = 0;
  const timer = setInterval(() => {
    audio.volume = Math.min(0.8, audio.volume + dv);
    if (++i >= steps) clearInterval(timer);
  }, dt);
}

function stopAudio() {
  if (currentAudio) {
    fadeOut(currentAudio, 150, () => {
      currentAudio.pause();
      currentAudio.src = '';
      currentAudio = null;
    });
  }
  lastCell = null;
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
  if (cellKey === lastCell) return;   // stessa cella, non ricaricare
  lastCell = cellKey;

  infoBar.textContent  = `${data.instrument}  ·  cella (${data.gi},${data.gj})  ·  dist ${data.dist.toFixed(2)}`;
  infoBar.classList.add('active');

  playFile(data.file);
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

  if (!soundEnabled) return;   // ← unica riga aggiunta

  if (throttleTimer) return;
  throttleTimer = setTimeout(() => {
    throttleTimer = null;
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
  stopAudio();
});

const btnToggle = document.getElementById('btn-toggle');
btnToggle.addEventListener('click', () => {
  soundEnabled = !soundEnabled;
  if (soundEnabled) {
    btnToggle.textContent = 'SOUND ON';
    btnToggle.classList.add('on');
  } else {
    btnToggle.textContent = 'SOUND OFF';
    btnToggle.classList.remove('on');
    stopAudio();
    if (throttleTimer) { clearTimeout(throttleTimer); throttleTimer = null; }
  }
});

// ── Avvio ─────────────────────────────────────────────────────────────────────
loadData();
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./checkpoints/ae_final.pt")
    parser.add_argument("--data_dir",   default="./data_processed")
    parser.add_argument("--grid_dir",   default="./grid_128iter")
    parser.add_argument("--port",       type=int, default=8080)
    args = parser.parse_args()

    setup(args.checkpoint, args.data_dir, args.grid_dir)

    app = web.Application()
    app.router.add_get('/',                handle_index)
    app.router.add_get('/data',            handle_data)
    app.router.add_post('/navigate',       handle_navigate)
    app.router.add_get('/audio/{filename}',handle_audio)

    print(f"\nApri nel browser: http://localhost:{args.port}")
    print("Muovi il mouse sulla mappa per generare suono.")
    print("Ctrl+C per fermare.\n")

    web.run_app(app, host='localhost', port=args.port, print=None)


if __name__ == "__main__":
    main()