#!/usr/bin/env python3
"""Generates the Deep-Live-Cam Colab WebRTC notebook (.ipynb)."""

import json, os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source, "execution_count": None, "outputs": []}

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": []
}

cells = notebook["cells"]

# ============================================================
# CELL 0: Title (markdown)
# ============================================================
cells.append(md([
    "# 🎭 Deep-Live-Cam — Colab WebRTC Streaming Server\n",
    "\n",
    "Real-time face swap streaming: **Local Webcam → WebRTC → Colab GPU → Deep-Live-Cam → WebRTC → Local PC**\n",
    "\n",
    "Run cells **1 → 8** sequentially. Cell **9** provides a monitoring dashboard.\n",
    "\n",
    "| Cell | Purpose |\n",
    "|------|----------|\n",
    "| 1 | Install dependencies & verify GPU |\n",
    "| 2 | Upload source face image |\n",
    "| 3 | Configure Deep-Live-Cam parameters |\n",
    "| 4 | Configure Ngrok tunnel |\n",
    "| 5 | Launch signaling server + initialize DLC |\n",
    "| 6 | Start processing pipeline |\n",
    "| 7 | Generate local client.py |\n",
    "| 8 | Monitoring dashboard |\n",
    "| 9 | Troubleshooting reference |"
]))

# ============================================================
# CELL 1: Install Dependencies + GPU Verification
# ============================================================
cells.append(code([
    '#@title 1️⃣ Install Dependencies & Verify GPU\n',
    '#@markdown Installs all required packages, clones Deep-Live-Cam, and verifies GPU availability.\n',
    '\n',
    'import subprocess, sys, os\n',
    '\n',
    'print("=" * 60)\n',
    'print("  STEP 1: Installing Dependencies")\n',
    'print("=" * 60)\n',
    '\n',
    '# System packages\n',
    'subprocess.run(["apt-get", "update", "-qq"], check=True,\n',
    '               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n',
    'subprocess.run(["apt-get", "install", "-y", "-qq",\n',
    '                "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0"],\n',
    '               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n',
    'print("[✓] System packages installed (ffmpeg, libgl1)")\n',
    '\n',
    '# Clone Deep-Live-Cam\n',
    'DLC_DIR = "/content/Deep-Live-Cam"\n',
    'if not os.path.exists(DLC_DIR):\n',
    '    subprocess.run(["git", "clone",\n',
    '                    "https://github.com/hacksider/Deep-Live-Cam.git",\n',
    '                    DLC_DIR], check=True)\n',
    '    print(f"[✓] Deep-Live-Cam cloned to {DLC_DIR}")\n',
    'else:\n',
    '    print(f"[✓] Deep-Live-Cam already exists at {DLC_DIR}")\n',
    '\n',
    '# Install DLC requirements (skip GUI / platform-specific deps)\n',
    'req_path = os.path.join(DLC_DIR, "requirements.txt")\n',
    'with open(req_path, "r") as f:\n',
    '    lines = f.readlines()\n',
    'filtered = []\n',
    'SKIP = ["pyside6", "pygrabber", "cv2_enumerate",\n',
    '        "onnxruntime-silicon", "tensorflow", "opennsfw2", "onnxruntime-gpu"]\n',
    'for line in lines:\n',
    '    s = line.strip()\n',
    '    if not s or s.startswith("#"):\n',
    '        continue\n',
    '    if any(skip in s.lower() for skip in SKIP):\n',
    '        continue\n',
    '    filtered.append(s)\n',
    'if filtered:\n',
    '    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + filtered, check=True)\n',
    '    print(f"[✓] DLC core dependencies installed ({len(filtered)} packages)")\n',
    '\n',
    '# WebRTC + server packages\n',
    'webrtc_deps = [\n',
    '    "aiortc>=1.9.0", "aiohttp>=3.9.0", "av>=12.0.0",\n',
    '    "pyngrok>=7.0.0", "opencv-python>=4.8.0", "onnxruntime-gpu==1.21.0",\n',
    ']\n',
    'subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + webrtc_deps, check=True)\n',
    'print("[✓] WebRTC / server dependencies installed")\n',
    '\n',
    '# InsightFace\n',
    'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "insightface==0.7.3"], check=True)\n',
    'print("[✓] InsightFace installed")\n',
    '\n',
    'print()\n',
    'print("=" * 60)\n',
    'print("  STEP 2: GPU Verification")\n',
    'print("=" * 60)\n',
    '\n',
    'try:\n',
    '    import torch\n',
    '    if torch.cuda.is_available():\n',
    '        gpu_name = torch.cuda.get_device_name(0)\n',
    '        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)\n',
    '        print(f"[✓] CUDA GPU: {gpu_name}")\n',
    '        print(f"    VRAM: {gpu_mem:.1f} GB")\n',
    '        print(f"    CUDA: {torch.version.cuda}  |  PyTorch: {torch.__version__}")\n',
    '    else:\n',
    '        print("[✗] No CUDA GPU! Runtime → Change runtime type → GPU")\n',
    'except ImportError:\n',
    '    print("[!] PyTorch unavailable — GPU blending will use CPU fallback")\n',
    '\n',
    'import onnxruntime\n',
    'ort_providers = onnxruntime.get_available_providers()\n',
    'print(f"\\n[✓] ONNX Runtime {onnxruntime.__version__}")\n',
    'print(f"    Providers: {ort_providers}")\n',
    'if "CUDAExecutionProvider" in ort_providers:\n',
    '    print("    [✓] CUDAExecutionProvider available")\n',
    'else:\n',
    '    print("    [✗] CUDAExecutionProvider NOT available — inference will be slow")\n',
    '\n',
    'import shutil\n',
    'if shutil.which("ffmpeg"):\n',
    '    print(f"\\n[✓] ffmpeg found at {shutil.which(\'ffmpeg\')}")\n',
    'else:\n',
    '    print("\\n[✗] ffmpeg not found!")\n',
    '\n',
    'print("\\n" + "=" * 60)\n',
    'print("  ✅ Environment ready!")\n',
    'print("=" * 60)'
]))

# ============================================================
# CELL 2: Upload Source Face
# ============================================================
cells.append(md([
    "## 2️⃣ Upload Source Face Image\n",
    "Upload a clear, front-facing photo of the face you want to swap onto the webcam stream."
]))

cells.append(code([
    '#@title Upload Source Face Image\n',
    '#@markdown Upload a clear front-facing photo (PNG / JPG).\n',
    '\n',
    'import os, sys, cv2\n',
    'import numpy as np\n',
    'from IPython.display import display, Image as IPImage\n',
    'from google.colab import files\n',
    '\n',
    'DLC_DIR = "/content/Deep-Live-Cam"\n',
    'SOURCE_PATH = os.path.join(DLC_DIR, "source_face.png")\n',
    '\n',
    'print("📸 Upload a source face image (PNG/JPG):")\n',
    'uploaded = files.upload()\n',
    '\n',
    'if uploaded:\n',
    '    filename = list(uploaded.keys())[0]\n',
    '    file_bytes = uploaded[filename]\n',
    '    with open(SOURCE_PATH, "wb") as f:\n',
    '        f.write(file_bytes)\n',
    '    print(f"[✓] Saved to {SOURCE_PATH}")\n',
    '    display(IPImage(data=file_bytes, width=300))\n',
    '\n',
    '    # Validate face detection\n',
    '    sys.path.insert(0, DLC_DIR)\n',
    '    import modules.globals\n',
    '    modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]\n',
    '    from modules.face_analyser import get_one_face\n',
    '    img = cv2.imread(SOURCE_PATH)\n',
    '    face = get_one_face(img)\n',
    '    if face is not None:\n',
    '        print(f"[✓] Face detected (confidence: {face.det_score:.3f})")\n',
    '    else:\n',
    '        print("[✗] No face detected — upload a clearer image.")\n',
    'else:\n',
    '    print("[!] No file uploaded. Run this cell again.")'
]))

# ============================================================
# CELL 3: Configuration
# ============================================================
cells.append(md([
    "## 3️⃣ Configure Deep-Live-Cam\n",
    "Edit the parameters below to control the face-swap pipeline."
]))

cells.append(code([
    '#@title Deep-Live-Cam Configuration\n',
    '#@markdown Adjust parameters to control quality, speed, and features.\n',
    '\n',
    '# === Frame Processors ===\n',
    'frame_processors = ["face_swapper"]  #@param {type:"raw"}\n',
    '#@markdown Available: `face_swapper`, `face_enhancer`, `face_enhancer_gpen256`, `face_enhancer_gpen512`\n',
    '\n',
    '# === Face Detection ===\n',
    'many_faces = False  #@param {type:"boolean"}\n',
    'face_detection_confidence = 0.5  #@param {type:"slider", min:0.1, max:1.0, step:0.05}\n',
    '\n',
    '# === Face Swap Options ===\n',
    'opacity = 1.0  #@param {type:"slider", min:0.0, max:1.0, step:0.05}\n',
    'sharpness = 0.0  #@param {type:"slider", min:0.0, max:1.0, step:0.1}\n',
    'mouth_mask = False  #@param {type:"boolean"}\n',
    'poisson_blend = False  #@param {type:"boolean"}\n',
    'color_correction = False  #@param {type:"boolean"}\n',
    '\n',
    '# === Performance ===\n',
    'processing_resolution = 640  #@param {type:"slider", min:320, max:1280, step:64}\n',
    'fps_limit = 30  #@param {type:"slider", min:10, max:60, step:5}\n',
    'execution_threads = 2  #@param {type:"slider", min:1, max:8, step:1}\n',
    'max_memory_gb = 8  #@param {type:"slider", min:4, max:16, step:2}\n',
    '\n',
    '# === Temporal Smoothing ===\n',
    'enable_interpolation = False  #@param {type:"boolean"}\n',
    'interpolation_weight = 0.0  #@param {type:"slider", min:0.0, max:1.0, step:0.1}\n',
    '\n',
    '# Store config\n',
    'CONFIG = dict(\n',
    '    frame_processors=frame_processors, many_faces=many_faces,\n',
    '    face_detection_confidence=face_detection_confidence,\n',
    '    opacity=opacity, sharpness=sharpness, mouth_mask=mouth_mask,\n',
    '    poisson_blend=poisson_blend, color_correction=color_correction,\n',
    '    processing_resolution=processing_resolution, fps_limit=fps_limit,\n',
    '    execution_threads=execution_threads, max_memory_gb=max_memory_gb,\n',
    '    enable_interpolation=enable_interpolation,\n',
    '    interpolation_weight=interpolation_weight,\n',
    ')\n',
    '\n',
    'print("Configuration:")\n',
    'for k, v in CONFIG.items():\n',
    '    print(f"  {k}: {v}")\n',
    'print("\\n[✓] Config ready. Proceed to Cell 4.")'
]))

# ============================================================
# CELL 4: Ngrok
# ============================================================
cells.append(md([
    "## 4️⃣ Configure Ngrok\n",
    "Enter your Ngrok auth token to create a public HTTPS tunnel.  \n",
    "Get a free token at [ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken)."
]))

cells.append(code([
    '#@title Ngrok Authentication\n',
    '#@markdown Your token is entered securely and never stored.\n',
    '\n',
    'import os, time, threading\n',
    'from getpass import getpass\n',
    'from pyngrok import ngrok, conf\n',
    '\n',
    'NGROK_AUTH_TOKEN = getpass("Enter your NGROK_AUTH_TOKEN: ")\n',
    'if not NGROK_AUTH_TOKEN or len(NGROK_AUTH_TOKEN) < 10:\n',
    '    raise ValueError("Invalid token — get one at https://dashboard.ngrok.com")\n',
    '\n',
    'conf.get_default().auth_token = NGROK_AUTH_TOKEN\n',
    'conf.get_default().region = "us"\n',
    'print("[✓] Ngrok authenticated")\n',
    '\n',
    'SERVER_PORT = 8080\n',
    '_ngrok_tunnel = None\n',
    '_ngrok_url = None\n',
    '_ngrok_monitor_running = False\n',
    '\n',
    'def start_ngrok_tunnel():\n',
    '    global _ngrok_tunnel, _ngrok_url\n',
    '    try:\n',
    '        ngrok.kill()\n',
    '        time.sleep(1)\n',
    '        _ngrok_tunnel = ngrok.connect(SERVER_PORT, "http", bind_tls=True)\n',
    '        _ngrok_url = _ngrok_tunnel.public_url\n',
    '        sep = "=" * 60\n',
    '        print(f"\\n{sep}")\n',
    '        print(f"  🌐 NGROK TUNNEL ACTIVE")\n',
    '        print(f"{sep}")\n',
    '        print(f"  URL: {_ngrok_url}")\n',
    '        print(f"  Health: {_ngrok_url}/health")\n',
    '        print(f"{sep}")\n',
    '        return _ngrok_url\n',
    '    except Exception as e:\n',
    '        print(f"[✗] Ngrok error: {e}")\n',
    '        return None\n',
    '\n',
    'def _ngrok_monitor():\n',
    '    global _ngrok_monitor_running\n',
    '    _ngrok_monitor_running = True\n',
    '    while _ngrok_monitor_running:\n',
    '        time.sleep(30)\n',
    '        try:\n',
    '            if not ngrok.get_tunnels():\n',
    '                print("\\n[!] Ngrok tunnel dropped — reconnecting...")\n',
    '                start_ngrok_tunnel()\n',
    '        except Exception:\n',
    '            try:\n',
    '                start_ngrok_tunnel()\n',
    '            except Exception:\n',
    '                pass\n',
    '\n',
    'url = start_ngrok_tunnel()\n',
    'if url:\n',
    '    threading.Thread(target=_ngrok_monitor, daemon=True).start()\n',
    '    print("\\n[✓] Auto-reconnect monitor active")\n',
    '    print("[✓] Ngrok ready. Proceed to Cell 5.")\n',
    'else:\n',
    '    print("[✗] Failed — check your auth token.")'
]))

# ============================================================
# CELL 5: Initialize DLC + Launch Server
# ============================================================
cells.append(md([
    "## 5️⃣ Initialize Pipeline & Launch Server\n",
    "This cell loads the DLC models, initializes the face swap pipeline with CUDA,\n",
    "and starts the WebRTC signaling server."
]))

cells.append(code([
    '#@title Initialize DLC & Launch Signaling Server\n',
    '#@markdown Loads models, warms up GPU, and starts the WebRTC server.\n',
    '\n',
    'import os, sys, time, json, asyncio, threading, traceback, logging\n',
    'from collections import deque\n',
    'import numpy as np\n',
    'import cv2\n',
    '\n',
    '# ── DLC Initialization ──────────────────────────────────────\n',
    'DLC_DIR = "/content/Deep-Live-Cam"\n',
    'sys.path.insert(0, DLC_DIR)\n',
    'os.chdir(DLC_DIR)\n',
    '\n',
    '# Pre-load NVIDIA shared libs (mirrors run.py lines 39-67)\n',
    'import ctypes, glob as _glob\n',
    '_py = f"python{sys.version_info.major}.{sys.version_info.minor}"\n',
    'for _sp in [os.path.join(DLC_DIR, "venv", "lib", _py, "site-packages"),\n',
    '            os.path.join(sys.prefix, "lib", _py, "site-packages")]:\n',
    '    _nd = os.path.join(_sp, "nvidia")\n',
    '    if not os.path.isdir(_nd):\n',
    '        continue\n',
    '    for _pkg in os.listdir(_nd):\n',
    '        _ld = os.path.join(_nd, _pkg, "lib")\n',
    '        if not os.path.isdir(_ld):\n',
    '            continue\n',
    '        _ldp = os.environ.get("LD_LIBRARY_PATH", "")\n',
    '        if _ld not in _ldp.split(os.pathsep):\n',
    '            os.environ["LD_LIBRARY_PATH"] = _ld + (os.pathsep + _ldp if _ldp else "")\n',
    '        for _so in sorted(_glob.glob(os.path.join(_ld, "lib*.so*"))):\n',
    '            try:\n',
    '                ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)\n',
    '            except OSError:\n',
    '                pass\n',
    '    break\n',
    '\n',
    'os.environ["OMP_NUM_THREADS"] = "6"\n',
    'os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"\n',
    '\n',
    '# Set globals\n',
    'import modules.globals\n',
    'modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]\n',
    'modules.globals.execution_threads = CONFIG.get("execution_threads", 2)\n',
    'modules.globals.max_memory = CONFIG.get("max_memory_gb", 8)\n',
    'modules.globals.frame_processors = CONFIG.get("frame_processors", ["face_swapper"])\n',
    'modules.globals.many_faces = CONFIG.get("many_faces", False)\n',
    'modules.globals.map_faces = False\n',
    'modules.globals.mouth_mask = CONFIG.get("mouth_mask", False)\n',
    'modules.globals.poisson_blend = CONFIG.get("poisson_blend", False)\n',
    'modules.globals.color_correction = CONFIG.get("color_correction", False)\n',
    'modules.globals.opacity = CONFIG.get("opacity", 1.0)\n',
    'modules.globals.sharpness = CONFIG.get("sharpness", 0.0)\n',
    'modules.globals.enable_interpolation = CONFIG.get("enable_interpolation", False)\n',
    'modules.globals.interpolation_weight = CONFIG.get("interpolation_weight", 0.0)\n',
    'modules.globals.headless = True\n',
    'modules.globals.source_path = "/content/Deep-Live-Cam/source_face.png"\n',
    'modules.globals.nsfw_filter = False\n',
    'modules.globals.log_level = "error"\n',
    'for ek in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):\n',
    '    modules.globals.fp_ui[ek] = ek in modules.globals.frame_processors\n',
    'print(f"[✓] Globals set  |  providers={modules.globals.execution_providers}")\n',
    'print(f"    processors={modules.globals.frame_processors}")\n',
    '\n',
    '# Load source face\n',
    'from modules.face_analyser import get_one_face, detect_one_face_fast, get_face_analyser\n',
    'SOURCE_PATH = "/content/Deep-Live-Cam/source_face.png"\n',
    'source_img = cv2.imread(SOURCE_PATH)\n',
    'if source_img is None:\n',
    '    raise FileNotFoundError(f"Source face not found at {SOURCE_PATH}. Run Cell 2.")\n',
    'print("[*] Loading face analyser (InsightFace buffalo_l)...")\n',
    'get_face_analyser()\n',
    'print("[✓] Face analyser loaded")\n',
    'source_face = get_one_face(source_img)\n',
    'if source_face is None:\n',
    '    raise ValueError("No face in source image — upload a better image in Cell 2.")\n',
    'print(f"[✓] Source face loaded (score={source_face.det_score:.3f})")\n',
    '\n',
    '# Load processors\n',
    'from modules.processors.frame.core import get_frame_processors_modules\n',
    'from modules.processors.frame.face_swapper import (\n',
    '    pre_check as swapper_pre_check, pre_start as swapper_pre_start,\n',
    '    get_face_swapper, process_frame as swapper_process_frame,\n',
    ')\n',
    'print("[*] Downloading face swap model (inswapper_128.onnx)...")\n',
    'swapper_pre_check()\n',
    'print("[✓] Model downloaded")\n',
    'print("[*] Initializing ONNX session (CUDA)...")\n',
    'swapper_pre_start()\n',
    'swapper = get_face_swapper()\n',
    'if swapper is None:\n',
    '    raise RuntimeError("Failed to load face swapper!")\n',
    'print("[✓] Face swapper ready (CUDA)")\n',
    '\n',
    'processor_modules = get_frame_processors_modules(modules.globals.frame_processors)\n',
    'for pm in processor_modules:\n',
    '    pm.pre_check()\n',
    '    print(f"[✓] Processor ready: {pm.NAME}")\n',
    '\n',
    '# GPU warm-up\n',
    'print("[*] Warming up GPU...")\n',
    'dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)\n',
    'try:\n',
    '    _ = swapper_process_frame(source_face, dummy.copy())\n',
    '    print("[✓] GPU warm-up complete")\n',
    'except Exception as e:\n',
    '    print(f"[!] Warm-up issue (non-fatal): {e}")\n',
    '\n',
    '# ── Performance Metrics ──────────────────────────────────────\n',
    '\n',
    'class PerfMetrics:\n',
    '    """Thread-safe performance metrics."""\n',
    '    def __init__(self):\n',
    '        self.lock = threading.Lock()\n',
    '        self.input_fps = 0.0\n',
    '        self.output_fps = 0.0\n',
    '        self.inference_ms = 0.0\n',
    '        self.queue_length = 0\n',
    '        self.connected_clients = 0\n',
    '        self.frames_processed = 0\n',
    '        self.frames_dropped = 0\n',
    '        self._input_t = deque(maxlen=60)\n',
    '        self._output_t = deque(maxlen=60)\n',
    '        self._infer_t = deque(maxlen=30)\n',
    '\n',
    '    def record_input(self):\n',
    '        now = time.time()\n',
    '        with self.lock:\n',
    '            self._input_t.append(now)\n',
    '            if len(self._input_t) >= 2:\n',
    '                dt = self._input_t[-1] - self._input_t[0]\n',
    '                if dt > 0:\n',
    '                    self.input_fps = (len(self._input_t) - 1) / dt\n',
    '\n',
    '    def record_output(self):\n',
    '        now = time.time()\n',
    '        with self.lock:\n',
    '            self._output_t.append(now)\n',
    '            self.frames_processed += 1\n',
    '            if len(self._output_t) >= 2:\n',
    '                dt = self._output_t[-1] - self._output_t[0]\n',
    '                if dt > 0:\n',
    '                    self.output_fps = (len(self._output_t) - 1) / dt\n',
    '\n',
    '    def record_inference(self, ms):\n',
    '        with self.lock:\n',
    '            self._infer_t.append(ms)\n',
    '            self.inference_ms = sum(self._infer_t) / len(self._infer_t)\n',
    '\n',
    '    def record_drop(self):\n',
    '        with self.lock:\n',
    '            self.frames_dropped += 1\n',
    '\n',
    '    def snapshot(self):\n',
    '        with self.lock:\n',
    '            return dict(\n',
    '                input_fps=round(self.input_fps, 1),\n',
    '                output_fps=round(self.output_fps, 1),\n',
    '                inference_ms=round(self.inference_ms, 1),\n',
    '                queue_length=self.queue_length,\n',
    '                connected_clients=self.connected_clients,\n',
    '                frames_processed=self.frames_processed,\n',
    '                frames_dropped=self.frames_dropped,\n',
    '            )\n',
    '\n',
    'metrics = PerfMetrics()\n',
    '\n',
    '# ── WebRTC Video Track ──────────────────────────────────────\n',
    'import av\n',
    'from av import VideoFrame\n',
    'from aiohttp import web\n',
    'from aiortc import (\n',
    '    RTCPeerConnection, RTCSessionDescription,\n',
    '    MediaStreamTrack, RTCConfiguration, RTCIceServer,\n',
    ')\n',
    'from aiortc.contrib.media import MediaRelay\n',
    '\n',
    'PROC_RES = CONFIG.get("processing_resolution", 640)\n',
    'FPS_LIMIT = CONFIG.get("fps_limit", 30)\n',
    'MIN_INTERVAL = 1.0 / FPS_LIMIT\n',
    'DET_CONF = CONFIG.get("face_detection_confidence", 0.5)\n',
    '\n',
    'class FaceSwapTrack(MediaStreamTrack):\n',
    '    """Receives frames, runs DLC face swap, returns processed frames."""\n',
    '    kind = "video"\n',
    '\n',
    '    def __init__(self, track):\n',
    '        super().__init__()\n',
    '        self.track = track\n',
    '        self._last_t = 0\n',
    '        self._last_result = None\n',
    '\n',
    '    def _process_sync(self, bgr):\n',
    '        """Run face swap in thread pool (never blocks event loop)."""\n',
    '        try:\n',
    '            h, w = bgr.shape[:2]\n',
    '            scale = 1.0\n',
    '            if max(h, w) > PROC_RES:\n',
    '                scale = PROC_RES / max(h, w)\n',
    '                nw, nh = int(w * scale), int(h * scale)\n',
    '                proc = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)\n',
    '            else:\n',
    '                proc = bgr\n',
    '\n',
    '            t0 = time.time()\n',
    '            # Fast detection path (skips landmarks / recognition)\n',
    '            target = detect_one_face_fast(proc) if not modules.globals.many_faces else None\n',
    '            if target is not None and target.det_score < DET_CONF:\n',
    '                target = None\n',
    '\n',
    '            if target is not None or modules.globals.many_faces:\n',
    '                for pm in processor_modules:\n',
    '                    try:\n',
    '                        proc = pm.process_frame(source_face, proc, target_face=target)\n',
    '                    except TypeError:\n',
    '                        proc = pm.process_frame(source_face, proc)\n',
    '\n',
    '            metrics.record_inference((time.time() - t0) * 1000)\n',
    '            if scale != 1.0:\n',
    '                proc = cv2.resize(proc, (w, h), interpolation=cv2.INTER_LINEAR)\n',
    '            metrics.record_output()\n',
    '            return proc\n',
    '        except Exception as e:\n',
    '            logging.error(f"Processing error: {e}")\n',
    '            return bgr\n',
    '\n',
    '    async def recv(self):\n',
    '        frame = await self.track.recv()\n',
    '        metrics.record_input()\n',
    '\n',
    '        # FPS limiting — reuse last result if too fast\n',
    '        now = time.time()\n',
    '        if (now - self._last_t) < MIN_INTERVAL and self._last_result is not None:\n',
    '            metrics.record_drop()\n',
    '            out = VideoFrame.from_ndarray(self._last_result, format="bgr24")\n',
    '            out.pts = frame.pts\n',
    '            out.time_base = frame.time_base\n',
    '            return out\n',
    '        self._last_t = now\n',
    '\n',
    '        try:\n',
    '            img = frame.to_ndarray(format="bgr24")\n',
    '        except Exception:\n',
    '            if self._last_result is not None:\n',
    '                out = VideoFrame.from_ndarray(self._last_result, format="bgr24")\n',
    '                out.pts = frame.pts\n',
    '                out.time_base = frame.time_base\n',
    '                return out\n',
    '            return frame\n',
    '\n',
    '        loop = asyncio.get_event_loop()\n',
    '        try:\n',
    '            result = await loop.run_in_executor(None, self._process_sync, img)\n',
    '            self._last_result = result\n',
    '        except Exception:\n',
    '            result = img\n',
    '\n',
    '        out = VideoFrame.from_ndarray(result, format="bgr24")\n',
    '        out.pts = frame.pts\n',
    '        out.time_base = frame.time_base\n',
    '        return out\n',
    '\n',
    '# ── Signaling Server ────────────────────────────────────────\n',
    '\n',
    'pcs = set()\n',
    'relay = MediaRelay()\n',
    'RTC_CFG = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])\n',
    '\n',
    'async def handle_offer(request):\n',
    '    try:\n',
    '        params = await request.json()\n',
    '        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])\n',
    '    except Exception as e:\n',
    '        return web.json_response({"error": str(e)}, status=400)\n',
    '\n',
    '    pc = RTCPeerConnection(configuration=RTC_CFG)\n',
    '    pcs.add(pc)\n',
    '    metrics.connected_clients = len(pcs)\n',
    '    print(f"[WebRTC] New peer (total: {len(pcs)})")\n',
    '\n',
    '    @pc.on("connectionstatechange")\n',
    '    async def on_state():\n',
    '        st = pc.connectionState\n',
    '        print(f"[WebRTC] State: {st}")\n',
    '        if st in ("failed", "closed"):\n',
    '            await pc.close()\n',
    '            pcs.discard(pc)\n',
    '            metrics.connected_clients = len(pcs)\n',
    '            try:\n',
    '                import torch\n',
    '                if torch.cuda.is_available():\n',
    '                    torch.cuda.empty_cache()\n',
    '            except ImportError:\n',
    '                pass\n',
    '\n',
    '    @pc.on("track")\n',
    '    def on_track(track):\n',
    '        print(f"[WebRTC] Track: {track.kind}")\n',
    '        if track.kind == "video":\n',
    '            pc.addTrack(FaceSwapTrack(relay.subscribe(track)))\n',
    '        @track.on("ended")\n',
    '        async def on_ended():\n',
    '            print(f"[WebRTC] Track ended: {track.kind}")\n',
    '\n',
    '    await pc.setRemoteDescription(offer)\n',
    '    answer = await pc.createAnswer()\n',
    '    await pc.setLocalDescription(answer)\n',
    '    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})\n',
    '\n',
    'async def handle_health(request):\n',
    '    snap = metrics.snapshot()\n',
    '    gpu = {}\n',
    '    try:\n',
    '        import torch\n',
    '        if torch.cuda.is_available():\n',
    '            gpu = dict(\n',
    '                gpu_name=torch.cuda.get_device_name(0),\n',
    '                vram_alloc_mb=round(torch.cuda.memory_allocated(0)/1e6, 1),\n',
    '                vram_total_mb=round(torch.cuda.get_device_properties(0).total_mem/1e6, 1),\n',
    '            )\n',
    '    except ImportError:\n',
    '        pass\n',
    '    return web.json_response({"status": "ok", "metrics": snap, "gpu": gpu})\n',
    '\n',
    'async def handle_index(request):\n',
    '    s = metrics.snapshot()\n',
    '    html = (\n',
    '        "<html><head><title>DLC Server</title>"\n',
    '        "<style>body{font-family:monospace;background:#1a1a2e;color:#e0e0e0;padding:40px}"\n',
    '        "h1{color:#00d4ff}td,th{padding:6px 14px;border:1px solid #333}"\n',
    '        "th{background:#16213e;color:#00d4ff}.ok{color:#0f8}</style></head><body>"\n',
    '        "<h1>🎭 Deep-Live-Cam WebRTC Server</h1><p class=ok>● Running</p>"\n',
    '        f"<table><tr><th>Metric</th><th>Value</th></tr>"\n',
    '        f"<tr><td>Clients</td><td>{s[\'connected_clients\']}</td></tr>"\n',
    '        f"<tr><td>In FPS</td><td>{s[\'input_fps\']}</td></tr>"\n',
    '        f"<tr><td>Out FPS</td><td>{s[\'output_fps\']}</td></tr>"\n',
    '        f"<tr><td>Inference</td><td>{s[\'inference_ms\']} ms</td></tr>"\n',
    '        f"<tr><td>Processed</td><td>{s[\'frames_processed\']}</td></tr>"\n',
    '        f"<tr><td>Dropped</td><td>{s[\'frames_dropped\']}</td></tr></table></body></html>"\n',
    '    )\n',
    '    return web.Response(text=html, content_type="text/html")\n',
    '\n',
    'async def on_shutdown(app):\n',
    '    await asyncio.gather(*(pc.close() for pc in pcs))\n',
    '    pcs.clear()\n',
    '\n',
    '# CORS middleware\n',
    '@web.middleware\n',
    'async def cors_mw(request, handler):\n',
    '    if request.method == "OPTIONS":\n',
    '        resp = web.Response()\n',
    '    else:\n',
    '        try:\n',
    '            resp = await handler(request)\n',
    '        except web.HTTPException as ex:\n',
    '            resp = ex\n',
    '    resp.headers["Access-Control-Allow-Origin"] = "*"\n',
    '    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"\n',
    '    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"\n',
    '    return resp\n',
    '\n',
    'app = web.Application(middlewares=[cors_mw])\n',
    'app.router.add_get("/", handle_index)\n',
    'app.router.add_post("/offer", handle_offer)\n',
    'app.router.add_get("/health", handle_health)\n',
    'app.on_shutdown.append(on_shutdown)\n',
    '\n',
    'def _run_server():\n',
    '    loop = asyncio.new_event_loop()\n',
    '    asyncio.set_event_loop(loop)\n',
    '    runner = web.AppRunner(app)\n',
    '    loop.run_until_complete(runner.setup())\n',
    '    site = web.TCPSite(runner, "0.0.0.0", SERVER_PORT)\n',
    '    loop.run_until_complete(site.start())\n',
    '    print(f"\\n[✓] Server on port {SERVER_PORT}")\n',
    '    print(f"[✓] Ngrok: {_ngrok_url}")\n',
    '    sep = "=" * 60\n',
    '    print(f"\\n{sep}")\n',
    '    print(f"  🎭 READY — run client.py with:")\n',
    '    print(f"  python client.py --url {_ngrok_url}")\n',
    '    print(f"{sep}\\n")\n',
    '    loop.run_forever()\n',
    '\n',
    'server_thread = threading.Thread(target=_run_server, daemon=True)\n',
    'server_thread.start()\n',
    'time.sleep(2)\n',
    'print("[✓] Server running. Proceed to Cell 7 to get the client.")'
]))

# ============================================================
# CELL 7: Generate Client
# ============================================================
cells.append(md([
    "## 7️⃣ Generate Local Client\n",
    "Downloads `client.py` for your local PC. Run it to start streaming."
]))

# We'll put the client code inline
CLIENT_CODE = r'''#!/usr/bin/env python3
"""Deep-Live-Cam WebRTC Client

Streams your local webcam to a remote Deep-Live-Cam Colab server
via WebRTC and displays the face-swapped result in real time.

Usage:
    pip install aiortc aiohttp opencv-python av pyvirtualcam
    python client.py --url https://xxxx.ngrok-free.app
    python client.py --url https://xxxx.ngrok-free.app --camera 1 --vcam
"""

import argparse
import asyncio
import logging
import sys
import time
import threading
from fractions import Fraction

import aiohttp
import cv2
import numpy as np
from av import VideoFrame
from aiortc import (
    RTCPeerConnection, RTCSessionDescription,
    MediaStreamTrack, RTCConfiguration, RTCIceServer,
)

logger = logging.getLogger("dlc-client")


class WebcamTrack(MediaStreamTrack):
    """Captures frames from local webcam and sends via WebRTC."""
    kind = "video"

    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        af = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened: {aw}x{ah} @ {af}fps")
        self._time_base = Fraction(1, fps)
        self._pts = 0

    async def recv(self):
        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, self.cap.read)
        if not ret:
            raise RuntimeError("Failed to read from camera")
        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts = self._pts
        vf.time_base = self._time_base
        self._pts += 1
        return vf

    def stop(self):
        super().stop()
        if self.cap:
            self.cap.release()


class FrameReceiver:
    """Collects frames from remote processed video track."""
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.count = 0
        self._ts = []
        self.fps = 0.0

    def update(self, bgr):
        with self.lock:
            self.frame = bgr
            self.count += 1
            now = time.time()
            self._ts.append(now)
            if len(self._ts) > 30:
                self._ts = self._ts[-30:]
            if len(self._ts) >= 2:
                dt = self._ts[-1] - self._ts[0]
                if dt > 0:
                    self.fps = (len(self._ts) - 1) / dt

    def get(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


async def run_client(url, camera, w, h, fps, vcam, vw, vh, retries, delay):
    """Main loop with auto-reconnect."""
    for attempt in range(1, retries + 1):
        if attempt > 1:
            logger.info(f"Retry {attempt}/{retries} in {delay}s...")
            await asyncio.sleep(delay)
        try:
            await _stream(url, camera, w, h, fps, vcam, vw, vh)
            return
        except KeyboardInterrupt:
            return
        except Exception as e:
            logger.error(f"Connection error: {e}")
    logger.error(f"Max retries ({retries}) exceeded.")


async def _stream(url, camera, w, h, fps, use_vcam, vw, vh):
    """Connect, stream, display."""
    rx = FrameReceiver()
    cfg = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])
    pc = RTCPeerConnection(configuration=cfg)
    cam = WebcamTrack(camera, w, h, fps)
    pc.addTrack(cam)

    @pc.on("track")
    def on_track(track):
        logger.info(f"Remote track: {track.kind}")
        if track.kind == "video":
            asyncio.ensure_future(_consume(track, rx))

    @pc.on("connectionstatechange")
    async def on_state():
        logger.info(f"State: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            raise ConnectionError("WebRTC lost")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    offer_url = url.rstrip("/") + "/offer"
    logger.info(f"Sending offer to {offer_url}")

    async with aiohttp.ClientSession() as sess:
        hdrs = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
        payload = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        async with sess.post(offer_url, json=payload, headers=hdrs, ssl=False) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Server {resp.status}: {await resp.text()}")
            ans = await resp.json()

    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
    logger.info("Connected!")

    vcam_dev = None
    if use_vcam:
        try:
            import pyvirtualcam
            vcam_dev = pyvirtualcam.Camera(width=vw, height=vh, fps=fps,
                                           fmt=pyvirtualcam.PixelFormat.BGR)
            logger.info(f"Virtual cam: {vcam_dev.device}")
        except Exception as e:
            logger.warning(f"VCam unavailable: {e}")

    print("\n" + "=" * 50)
    print("  Deep-Live-Cam Client Running")
    print("  'q' = quit  |  'm' = mirror")
    print("=" * 50 + "\n")

    mirror = False
    try:
        while True:
            f = rx.get()
            if f is not None:
                disp = cv2.flip(f, 1) if mirror else f.copy()
                cv2.putText(disp, f"FPS: {rx.fps:.1f} | #{rx.count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Deep-Live-Cam", disp)
                if vcam_dev:
                    try:
                        vcam_dev.send(cv2.resize(f, (vw, vh)))
                        vcam_dev.sleep_until_next_frame()
                    except Exception:
                        pass
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                mirror = not mirror
                print(f"Mirror: {'ON' if mirror else 'OFF'}")
            await asyncio.sleep(0.001)
    finally:
        cv2.destroyAllWindows()
        cam.stop()
        if vcam_dev:
            vcam_dev.close()
        await pc.close()


async def _consume(track, rx):
    while True:
        try:
            frame = await track.recv()
            rx.update(frame.to_ndarray(format="bgr24"))
        except Exception:
            break


def main():
    ap = argparse.ArgumentParser(description="Deep-Live-Cam WebRTC Client")
    ap.add_argument("--url", required=True, help="Colab ngrok URL")
    ap.add_argument("--camera", type=int, default=0, help="Camera index")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--vcam", action="store_true", help="Virtual camera output")
    ap.add_argument("--vcam-width", type=int, default=1280)
    ap.add_argument("--vcam-height", type=int, default=720)
    ap.add_argument("--retries", type=int, default=10)
    ap.add_argument("--retry-delay", type=float, default=3.0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    print("\n" + "=" * 50)
    print("  🎭 Deep-Live-Cam WebRTC Client")
    print("=" * 50)
    print(f"  Server:  {a.url}")
    print(f"  Camera:  {a.camera} ({a.width}x{a.height} @ {a.fps}fps)")
    print(f"  VCam:    {'ON' if a.vcam else 'OFF'}")
    print("=" * 50 + "\n")

    try:
        asyncio.run(run_client(a.url, a.camera, a.width, a.height, a.fps,
                               a.vcam, a.vcam_width, a.vcam_height,
                               a.retries, a.retry_delay))
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
'''

# Escape the client code for embedding in a notebook cell
client_escaped = CLIENT_CODE.replace('\\', '\\\\').replace("'", "\\'")

cells.append(code([
    '#@title Generate client.py\n',
    '#@markdown Downloads the local Python client for your PC.\n',
    '\n',
    'CLIENT_CODE = """\n',
] + [line + '\n' for line in CLIENT_CODE.split('\n')] + [
    '"""\n',
    '\n',
    '# Write client.py\n',
    'for p in ["/content/client.py", "/content/Deep-Live-Cam/client.py"]:\n',
    '    with open(p, "w") as f:\n',
    '        f.write(CLIENT_CODE)\n',
    'print("[✓] client.py generated")\n',
    '\n',
    '# Download\n',
    'from google.colab import files\n',
    'files.download("/content/client.py")\n',
    '\n',
    'print()\n',
    'print("=" * 60)\n',
    'print("  📥 CLIENT INSTRUCTIONS")\n',
    'print("=" * 60)\n',
    'print()\n',
    'print("  1. Install deps on LOCAL PC:")\n',
    'print("     pip install aiortc aiohttp opencv-python av pyvirtualcam")\n',
    'print()\n',
    'print(f"  2. Run:  python client.py --url {_ngrok_url}")\n',
    'print(f"  3. VCam: python client.py --url {_ngrok_url} --vcam")\n',
    'print()\n',
    'print("  Keys: q=quit, m=mirror")\n',
    'print("  Flags: --verbose, --camera 1, --width 320 --height 240")\n',
    'print("=" * 60)'
]))

# ============================================================
# CELL 8: Monitoring Dashboard
# ============================================================
cells.append(md([
    "## 8️⃣ Monitoring Dashboard\n",
    "Run this cell to see live GPU / performance metrics. Re-run to refresh."
]))

cells.append(code([
    '#@title Monitoring Dashboard (auto-refreshes every 3s)\n',
    '#@markdown Displays GPU utilization, FPS, latency, and connection info.\n',
    '\n',
    'import subprocess, time\n',
    'from IPython.display import display, HTML, clear_output\n',
    '\n',
    'def gpu_stats():\n',
    '    try:\n',
    '        r = subprocess.run(\n',
    '            ["nvidia-smi",\n',
    '             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",\n',
    '             "--format=csv,noheader,nounits"],\n',
    '            capture_output=True, text=True, timeout=5)\n',
    '        if r.returncode == 0:\n',
    '            p = r.stdout.strip().split(", ")\n',
    '            return dict(util=f"{p[0]}%", used=f"{p[1]} MB",\n',
    '                        total=f"{p[2]} MB", temp=f"{p[3]}°C")\n',
    '    except Exception:\n',
    '        pass\n',
    '    return dict(util="N/A", used="N/A", total="N/A", temp="N/A")\n',
    '\n',
    'def render():\n',
    '    s = metrics.snapshot()\n',
    '    g = gpu_stats()\n',
    '    return f"""\n',
    '    <div style="font-family:monospace;background:#0d1117;color:#c9d1d9;\n',
    '                padding:20px;border-radius:10px;border:1px solid #30363d">\n',
    '      <h2 style="color:#58a6ff;margin-top:0">🎭 Deep-Live-Cam Monitor</h2>\n',
    '      <p style="color:#8b949e">Updated: {time.strftime("%H:%M:%S")}</p>\n',
    '      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">\n',
    '        <div style="background:#161b22;padding:12px;border-radius:8px;border:1px solid #30363d">\n',
    '          <h3 style="color:#7ee787;margin-top:0">🖥️ GPU</h3>\n',
    '          <table style="width:100%;color:#c9d1d9">\n',
    '            <tr><td>Utilization</td><td style="text-align:right;color:#58a6ff">{g["util"]}</td></tr>\n',
    '            <tr><td>VRAM</td><td style="text-align:right;color:#58a6ff">{g["used"]} / {g["total"]}</td></tr>\n',
    '            <tr><td>Temperature</td><td style="text-align:right;color:#58a6ff">{g["temp"]}</td></tr>\n',
    '          </table>\n',
    '        </div>\n',
    '        <div style="background:#161b22;padding:12px;border-radius:8px;border:1px solid #30363d">\n',
    '          <h3 style="color:#7ee787;margin-top:0">📊 Performance</h3>\n',
    '          <table style="width:100%;color:#c9d1d9">\n',
    '            <tr><td>Input FPS</td><td style="text-align:right;color:#58a6ff">{s["input_fps"]}</td></tr>\n',
    '            <tr><td>Output FPS</td><td style="text-align:right;color:#58a6ff">{s["output_fps"]}</td></tr>\n',
    '            <tr><td>Inference</td><td style="text-align:right;color:#58a6ff">{s["inference_ms"]} ms</td></tr>\n',
    '          </table>\n',
    '        </div>\n',
    '        <div style="background:#161b22;padding:12px;border-radius:8px;border:1px solid #30363d">\n',
    '          <h3 style="color:#7ee787;margin-top:0">🔗 Connections</h3>\n',
    '          <table style="width:100%;color:#c9d1d9">\n',
    '            <tr><td>Clients</td><td style="text-align:right;color:#58a6ff">{s["connected_clients"]}</td></tr>\n',
    '            <tr><td>Processed</td><td style="text-align:right;color:#58a6ff">{s["frames_processed"]}</td></tr>\n',
    '            <tr><td>Dropped</td><td style="text-align:right;color:#f85149">{s["frames_dropped"]}</td></tr>\n',
    '          </table>\n',
    '        </div>\n',
    '        <div style="background:#161b22;padding:12px;border-radius:8px;border:1px solid #30363d">\n',
    '          <h3 style="color:#7ee787;margin-top:0">⚙️ Config</h3>\n',
    '          <table style="width:100%;color:#c9d1d9">\n',
    '            <tr><td>Processors</td><td style="text-align:right;color:#58a6ff">{", ".join(modules.globals.frame_processors)}</td></tr>\n',
    '            <tr><td>Resolution</td><td style="text-align:right;color:#58a6ff">{PROC_RES}px</td></tr>\n',
    '            <tr><td>FPS Limit</td><td style="text-align:right;color:#58a6ff">{FPS_LIMIT}</td></tr>\n',
    '          </table>\n',
    '        </div>\n',
    '      </div>\n',
    '      <p style="color:#8b949e;margin-bottom:0;margin-top:12px">Ngrok: <code style="color:#58a6ff">{_ngrok_url}</code></p>\n',
    '    </div>\n',
    '    """\n',
    '\n',
    'REFRESH = 30  # iterations (30 × 3s = 90s, re-run to continue)\n',
    'for _ in range(REFRESH):\n',
    '    clear_output(wait=True)\n',
    '    display(HTML(render()))\n',
    '    time.sleep(3)\n',
    'print("Dashboard paused — re-run cell to continue.")'
]))

# ============================================================
# CELL 9: Troubleshooting
# ============================================================
cells.append(md([
    "## 🔧 Troubleshooting\n",
    "\n",
    "### Common Issues\n",
    "\n",
    "| Issue | Solution |\n",
    "|-------|----------|\n",
    '| "No CUDA GPU found" | Runtime → Change runtime type → GPU |\n',
    '| "Failed to read from camera" | `--camera 1` or `--camera 2` |\n',
    '| "Connection refused" | Ensure Cell 5 is running and ngrok URL is correct |\n',
    '| "No face detected" in source | Upload a clearer front-facing photo |\n',
    "| High latency | Reduce `processing_resolution` to 320 or 480 |\n",
    "| Choppy output | Reduce `fps_limit` to 15-20 |\n",
    "| OOM error | Reduce `max_memory_gb`; use only `face_swapper` |\n",
    "| Ngrok drops | Auto-reconnect monitor handles this |\n",
    "| Black/frozen video | Restart Cell 5, reconnect client |\n",
    "\n",
    "### Performance Tips\n",
    "- **Lowest latency**: Use only `face_swapper` (~30-50ms/frame on T4)\n",
    "- **Faster inference**: Set `processing_resolution` to 480\n",
    "- **Less bandwidth**: Use `--width 320 --height 240` on client\n",
    "- **Disable extras**: Turn off `mouth_mask`, `poisson_blend`, `color_correction`\n",
    "\n",
    "### Client Installation\n",
    "```bash\n",
    "pip install aiortc aiohttp opencv-python av pyvirtualcam\n",
    "python client.py --url https://xxxx.ngrok-free.app\n",
    "```"
]))

# ============================================================
# Write the notebook
# ============================================================
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep_live_cam_webrtc.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Notebook written to: {out_path}")
print(f"Total cells: {len(cells)}")
