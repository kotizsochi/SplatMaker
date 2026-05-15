#!/usr/bin/env python3
"""SplatMaker - Gaussian Splatting Pipeline (Local Only) v5
Cloud module preserved in cloud_module.py for future use.
"""
import json, os, queue, sqlite3, subprocess, threading, time, uuid, shutil, glob, psutil, logging
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('splat')
from flask import Flask, request, jsonify, Response, send_from_directory, render_template, send_file
from pathlib import Path

app = Flask(__name__, template_folder="templates", static_folder="static")
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(BASE, "projects")
HISTORY_FILE = os.path.join(BASE, "history.json")
os.makedirs(PROJECTS_DIR, exist_ok=True)

VIDEO_EXT = {'.mov', '.mp4', '.avi', '.mkv', '.mts', '.m4v'}
PHOTO_EXT = {'.jpg', '.jpeg', '.png', '.dng', '.tif', '.tiff', '.bmp', '.webp'}

jobs = {}
job_queue = queue.Queue()
cpu_history = []
pending_sources = {}  # session sources before job start

QUALITY_PRESETS = {
    "fast":   {"max_features": 4096,  "steps": 7000,  "overlap": 10},
    "medium": {"max_features": 8192,  "steps": 15000, "overlap": 15},
    "high":   {"max_features": 8192,  "steps": 30000, "overlap": 20},
    "ultra":  {"max_features": 16384, "steps": 50000, "overlap": 25},
}

STEPS = [
    {"id": "upload",   "name": "Подготовка источников"},
    {"id": "frames",   "name": "Извлечение кадров"},
    {"id": "blur",     "name": "Фильтр размытия"},
    {"id": "features", "name": "Feature Extraction"},
    {"id": "matching", "name": "Feature Matching"},
    {"id": "mapper",   "name": "3D Реконструкция"},
    {"id": "bundle",   "name": "Bundle Adjustment"},
    {"id": "undistort","name": "Image Undistorter"},
    {"id": "training", "name": "Gaussian Splat Training"},
    {"id": "done",     "name": "Готово!"},
]

def calc_blur_score(img_path):
    """Calculate blur score using Laplacian variance. Higher = sharper."""
    try:
        img = Image.open(img_path).convert('L')  # grayscale
        # Resize for speed (blur detection doesn't need full res)
        w, h = img.size
        if w > 800:
            ratio = 800 / w
            img = img.resize((800, int(h * ratio)), Image.LANCZOS)
        arr = np.array(img, dtype=np.float64)
        # Laplacian kernel
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        from scipy.signal import convolve2d
        filtered = convolve2d(arr, laplacian, mode='same', boundary='symm')
        return float(np.var(filtered))
    except:
        # Fallback without scipy: simple gradient variance
        try:
            img = Image.open(img_path).convert('L')
            w, h = img.size
            if w > 800:
                ratio = 800 / w
                img = img.resize((800, int(h * ratio)), Image.LANCZOS)
            arr = np.array(img, dtype=np.float64)
            gx = np.diff(arr, axis=1)
            gy = np.diff(arr, axis=0)
            return float(np.var(gx) + np.var(gy))
        except:
            return 999999  # keep on error

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f: return json.load(f)
        except: pass
    return []

def save_history():
    history = []
    for j in jobs.values():
        history.append({
            "id": j["id"], "name": j["name"], "quality": j["quality"],
            "status": j["status"], "total_images": j.get("total_images", 0),
            "created": j.get("created"), "finished": j.get("finished"),
            "project_dir": j.get("project_dir"),
        })
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def update_job(job_id, **kwargs):
    if job_id in jobs:
        job = jobs[job_id]
        if "current_step" in kwargs and kwargs["current_step"] != job.get("current_step"):
            old_step = job.get("current_step")
            if old_step and old_step in job.get("step_times", {}):
                dur = time.time() - job["step_times"][old_step]["start"]
                job["step_times"][old_step]["end"] = time.time()
                job["step_times"][old_step]["duration"] = dur
                log.info(f'[{job_id}] step "{old_step}" done ({dur:.1f}s)')
            if "step_times" not in job: job["step_times"] = {}
            job["step_times"][kwargs["current_step"]] = {"start": time.time(), "end": None, "duration": None}
            log.info(f'[{job_id}] -> {kwargs["current_step"]}')
        if "error" in kwargs and kwargs["error"]:
            log.error(f'[{job_id}] {kwargs["error"][:200]}')
        if kwargs.get("status") == "done":
            log.info(f'[{job_id}] DONE ({(time.time()-job.get("created",time.time())):.0f}s)')
        job.update(kwargs)
        job["updated"] = time.time()
        if kwargs.get("status") in ("done", "error"):
            save_history()

def run_cmd(cmd, job_id, step_id, cwd=None, total_items=0):
    """Run command with real-time progress parsing"""
    log.info(f'[{job_id}] RUN: {" ".join(cmd[:5])}')
    update_job(job_id, current_step=step_id, step_status="running")
    try:
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, cwd=cwd)
        stderr_lines = []
        processed = 0

        # Read stderr line by line (COLMAP outputs progress there)
        for line in proc.stderr:
            stderr_lines.append(line)
            # Parse COLMAP progress: "Processed file [123/972]" or "Matching image [45/972]"
            import re
            m = re.search(r'\[(\d+)/(\d+)\]', line)
            if m:
                current, total = int(m.group(1)), int(m.group(2))
                pct = min(99, int(current / max(total, 1) * 100))
                if pct != processed:
                    processed = pct
                    update_job(job_id, step_progress=pct)
            # Also parse "Processing image" lines
            elif 'Processing image' in line or 'Processed image' in line:
                if total_items > 0:
                    processed += 1
                    pct = min(99, int(processed / total_items * 100))
                    update_job(job_id, step_progress=pct)

        proc.wait(timeout=7200)
        dt = time.time() - t0
        stderr_text = "".join(stderr_lines)

        if proc.returncode != 0:
            log.error(f'[{job_id}] FAIL {step_id} ({dt:.1f}s): {stderr_text[-200:]}')
            update_job(job_id, step_status="error", error=stderr_text[-500:] if stderr_text else "Unknown error")
            return False
        log.info(f'[{job_id}] OK {step_id} ({dt:.1f}s)')
        return True
    except subprocess.TimeoutExpired:
        proc.kill()
        update_job(job_id, step_status="error", error="Timeout (2h)")
        return False

def resolve_source_files(src):
    """Get list of actual file paths from a source entry. Returns [(path, type)]"""
    path = src["path"]
    stype = src["type"]
    if stype == "video":
        return [(path, "video")] if os.path.isfile(path) else []
    elif stype == "photo":
        return [(path, "photo")] if os.path.isfile(path) else []
    elif stype == "folder":
        results = []
        if os.path.isdir(path):
            for f in sorted(os.listdir(path)):
                ext = os.path.splitext(f)[1].lower()
                fp = os.path.join(path, f)
                if not os.path.isfile(fp): continue
                if ext in VIDEO_EXT: results.append((fp, "video"))
                elif ext in PHOTO_EXT: results.append((fp, "photo"))
        return results
    return []

def detect_progress(proj):
    """Auto-detect which steps are completed by checking output files"""
    images_dir = os.path.join(proj, "images")
    db_path = os.path.join(proj, "colmap", "database.db")
    sparse_dir = os.path.join(proj, "colmap", "sparse", "0")
    output_dir = os.path.join(proj, "colmap_output")
    ply_path = os.path.join(proj, "output.ply")

    completed = []
    total_images = 0

    # frames done?
    if os.path.isdir(images_dir):
        files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
        total_images = len(files)
        if total_images > 0:
            completed.append("frames")

    # features done?
    if os.path.isfile(db_path):
        try:
            conn = sqlite3.connect(db_path)
            n = conn.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0]
            conn.close()
            if n > 0: completed.append("features")
        except: pass

    # matching done?
    if "features" in completed:
        try:
            conn = sqlite3.connect(db_path)
            n = conn.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0").fetchone()[0]
            conn.close()
            if n > 0: completed.append("matching")
        except: pass

    # mapper done?
    if os.path.isdir(sparse_dir) and os.path.isfile(os.path.join(sparse_dir, "cameras.bin")):
        completed.append("mapper")
        completed.append("bundle")  # bundle writes to same dir

    # undistort done?
    if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
        completed.append("undistort")

    # training done?
    if os.path.isfile(ply_path):
        completed.append("training")

    # Determine next step
    step_order = ["frames", "blur", "features", "matching", "mapper", "bundle", "undistort", "training", "done"]
    resume_from = "frames"
    for s in step_order:
        if s in completed:
            idx = step_order.index(s)
            if idx + 1 < len(step_order):
                resume_from = step_order[idx + 1]
        else:
            break

    return {"completed": completed, "resume_from": resume_from, "total_images": total_images}

def process_job(job_id):
    """Local pipeline (CPU) with resume support"""
    job = jobs[job_id]
    proj = job["project_dir"]
    quality = QUALITY_PRESETS[job["quality"]]
    sources = job.get("sources", [])
    start_from = job.get("start_from", "frames")  # resume point

    images_dir = os.path.join(proj, "images")
    colmap_dir = os.path.join(proj, "colmap")
    db_path = os.path.join(colmap_dir, "database.db")
    sparse_dir = os.path.join(colmap_dir, "sparse")
    output_dir = os.path.join(proj, "colmap_output")

    for d in [images_dir, colmap_dir, sparse_dir]:
        os.makedirs(d, exist_ok=True)

    step_order = ["frames", "blur", "features", "matching", "mapper", "bundle", "undistort", "training"]
    start_idx = step_order.index(start_from) if start_from in step_order else 0

    update_job(job_id, status="processing", current_step="upload", step_progress=100)
    if start_idx > 0:
        log.info(f'[{job_id}] RESUME from step "{start_from}" (skipping {start_idx} steps)')

    # Step 1: Extract frames / symlink photos
    if start_idx <= 0:
        update_job(job_id, current_step="frames", step_progress=0)
        img_counter = 0
        total_sources = len(sources)
        for si, src in enumerate(sources):
            src_files = resolve_source_files(src)
            frame_every = src.get("frame_every", 2)
            for fpath, ftype in src_files:
                if ftype == "video":
                    prefix = f"s{si}_"
                    vf = f"select=not(mod(n\\,{frame_every}))" if frame_every > 1 else ""
                    cmd = ["ffmpeg", "-y", "-i", fpath]
                    if vf: cmd += ["-vf", vf, "-vsync", "vfr"]
                    cmd += ["-qscale:v", "1", "-qmin", "1",
                            os.path.join(images_dir, f"{prefix}frame_%05d.jpg")]
                    subprocess.run(cmd, capture_output=True, timeout=1200)
                    log.info(f'[{job_id}] Extracted frames from {os.path.basename(fpath)} (every {frame_every})')
                elif ftype == "photo":
                    ext = os.path.splitext(fpath)[1]
                    link_name = f"photo_{img_counter:05d}{ext}"
                    link_path = os.path.join(images_dir, link_name)
                    try: os.symlink(os.path.abspath(fpath), link_path)
                    except FileExistsError: pass
                    img_counter += 1
            update_job(job_id, step_progress=int((si + 1) / max(total_sources, 1) * 100))

    total_images = len([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
    update_job(job_id, total_images=total_images, step_progress=100)
    if total_images == 0:
        update_job(job_id, status="error", error="Не найдено изображений")
        return

    # Step 1.5: Blur detection
    if start_idx <= 0:
        update_job(job_id, current_step="blur", step_progress=0)
        blur_dir = os.path.join(proj, "blurry")
        os.makedirs(blur_dir, exist_ok=True)
        all_imgs = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
        scores = []
        for i, fname in enumerate(all_imgs):
            fpath = os.path.join(images_dir, fname)
            score = calc_blur_score(fpath)
            scores.append((fname, score))
            if i % 50 == 0:
                update_job(job_id, step_progress=int(i / max(len(all_imgs), 1) * 80))

        if scores:
            avg_score = np.mean([s for _, s in scores])
            threshold = avg_score * 0.3  # 30% of average = blurry
            removed = 0
            for fname, score in scores:
                if score < threshold:
                    src = os.path.join(images_dir, fname)
                    dst = os.path.join(blur_dir, fname)
                    shutil.move(src, dst)
                    removed += 1

            remaining = total_images - removed
            log.info(f'[{job_id}] Blur filter: {removed} removed, {remaining} kept (threshold={threshold:.0f}, avg={avg_score:.0f})')
            update_job(job_id, total_images=remaining, step_progress=100,
                       blur_removed=removed, blur_threshold=round(threshold))
            if remaining == 0:
                update_job(job_id, status="error", error="Все кадры размыты")
                return
        else:
            update_job(job_id, step_progress=100)

    # Step 2: Feature Extraction
    if start_idx <= 1:
        update_job(job_id, current_step="features", step_progress=0)
        cmd = ["colmap", "feature_extractor", "--database_path", db_path, "--image_path", images_dir,
               "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1",
               "--SiftExtraction.max_num_features", str(quality["max_features"])]
        if not run_cmd(cmd, job_id, "features"): return
        update_job(job_id, step_progress=100)

    # Step 3: Sequential Matching
    if start_idx <= 2:
        update_job(job_id, current_step="matching", step_progress=0)
        cmd = ["colmap", "sequential_matcher", "--database_path", db_path,
               "--SequentialMatching.overlap", str(quality["overlap"]), "--SequentialMatching.quadratic_overlap", "1"]
        if not run_cmd(cmd, job_id, "matching"): return
        update_job(job_id, step_progress=100)

    # Step 4: Mapper
    if start_idx <= 3:
        update_job(job_id, current_step="mapper", step_progress=0)
        cmd = ["colmap", "mapper", "--database_path", db_path, "--image_path", images_dir,
               "--output_path", sparse_dir, "--Mapper.ba_refine_focal_length", "1",
               "--Mapper.ba_refine_extra_params", "1", "--Mapper.multiple_models", "0"]
        if not run_cmd(cmd, job_id, "mapper"): return
        if not os.path.isdir(os.path.join(sparse_dir, "0")):
            update_job(job_id, status="error", error="Mapper не создал модель")
            return
        update_job(job_id, step_progress=100)

    # Step 5: Bundle Adjustment
    if start_idx <= 4:
        update_job(job_id, current_step="bundle", step_progress=0)
        cmd = ["colmap", "bundle_adjuster", "--input_path", os.path.join(sparse_dir, "0"),
               "--output_path", os.path.join(sparse_dir, "0")]
        if not run_cmd(cmd, job_id, "bundle"): return
        update_job(job_id, step_progress=100)

    # Step 6: Undistort
    if start_idx <= 5:
        update_job(job_id, current_step="undistort", step_progress=0)
        os.makedirs(output_dir, exist_ok=True)
        cmd = ["colmap", "image_undistorter", "--image_path", images_dir,
               "--input_path", os.path.join(sparse_dir, "0"), "--output_path", output_dir, "--output_type", "COLMAP"]
        if not run_cmd(cmd, job_id, "undistort"): return
        update_job(job_id, step_progress=100)

    # Step 7: Brush Training
    if start_idx <= 6:
        update_job(job_id, current_step="training", step_progress=0)
        brush_path = shutil.which("brush_app")
        if brush_path:
            ply_output = os.path.join(proj, "output.ply")
            current_steps = jobs[job_id].get("training_steps", quality["steps"])
            cmd = [brush_path, "--total-steps", str(current_steps), "--export-path", ply_output, output_dir]
            if not run_cmd(cmd, job_id, "training"):
                update_job(job_id, step_progress=100, warning="Brush не завершён, COLMAP данные готовы")
        else:
            update_job(job_id, warning="Brush не найден. COLMAP данные готовы.")
        update_job(job_id, step_progress=100)

    update_job(job_id, current_step="done", step_progress=100, status="done", finished=time.time())

    # Save project metadata
    step_times = job.get("step_times", {})
    total_duration = time.time() - job.get("created", time.time())
    meta = {
        "name": job.get("name"),
        "id": job_id,
        "quality": job.get("quality"),
        "total_images": job.get("total_images", 0),
        "training_steps": job.get("training_steps"),
        "sources": [{
            "name": s.get("name"), "type": s.get("type"),
            "path": s.get("path"), "frame_every": s.get("frame_every")
        } for s in sources],
        "step_durations": {k: round(v.get("duration", 0), 1) for k, v in step_times.items() if v.get("duration")},
        "total_duration_sec": round(total_duration, 1),
        "total_duration_human": f"{int(total_duration//60)}m {int(total_duration%60)}s",
        "created": job.get("created"),
        "finished": job.get("finished"),
        "output_ply": os.path.exists(os.path.join(proj, "output.ply")),
        "colmap_output": os.path.isdir(output_dir),
    }
    with open(os.path.join(proj, "project.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info(f'[{job_id}] Saved project.json')

    # Cleanup: remove uploaded copies (originals stay untouched)
    uploads_dir = os.path.join(proj, "uploads")
    if os.path.isdir(uploads_dir):
        shutil.rmtree(uploads_dir, ignore_errors=True)
        log.info(f'[{job_id}] Cleaned uploads/ ({uploads_dir})')

def worker():
    while True:
        job_id = job_queue.get()
        j = jobs.get(job_id, {})
        log.info(f'===== JOB [{job_id}] "{j.get("name")}" q={j.get("quality")} sources={len(j.get("sources",[]))} =====')
        try:
            process_job(job_id)
        except Exception as e:
            log.exception(f'[{job_id}] EXCEPTION')
            update_job(job_id, status="error", error=str(e))
        log.info(f'===== JOB [{job_id}] END status={jobs.get(job_id,{}).get("status")} =====')
        job_queue.task_done()

def cpu_monitor():
    while True:
        cpu_history.append({"t": time.time(), "cpu": psutil.cpu_percent(interval=1),
                           "mem": psutil.virtual_memory().percent})
        if len(cpu_history) > 300: cpu_history.pop(0)

threading.Thread(target=worker, daemon=True).start()
threading.Thread(target=cpu_monitor, daemon=True).start()

# ========== Routes ==========

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/probe", methods=["POST"])
def probe_video():
    """Probe video by path or uploaded file"""
    data = request.get_json() or {}
    path = data.get("path", "").strip()

    if path and os.path.isfile(path):
        target = path
        cleanup = False
    elif request.files.get("file"):
        f = request.files["file"]
        target = os.path.join("/tmp", f"probe_{uuid.uuid4().hex[:8]}_{f.filename}")
        f.save(target)
        cleanup = True
    else:
        return jsonify({"error": "No file"}), 400

    try:
        r = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", target
        ], capture_output=True, text=True, timeout=15)
        d = json.loads(r.stdout)
        vs = next((s for s in d.get("streams", []) if s["codec_type"] == "video"), {})
        fps_str = vs.get("r_frame_rate", "30/1")
        num, den = fps_str.split("/")
        fps = round(int(num) / max(int(den), 1), 2)
        duration = float(d.get("format", {}).get("duration", 0))
        total_frames = int(float(vs.get("nb_frames", fps * duration)))
        return jsonify({
            "fps": fps, "duration": round(duration, 1),
            "total_frames": total_frames,
            "width": int(vs.get("width", 0)), "height": int(vs.get("height", 0)),
            "size_mb": round(os.path.getsize(target) / 1048576, 1),
            "codec": vs.get("codec_name", "unknown"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if cleanup: os.unlink(target)

@app.route("/api/scan-folder", methods=["POST"])
def scan_folder():
    """Scan a folder for media files"""
    data = request.get_json()
    path = data.get("path", "").strip()
    if not path or not os.path.isdir(path):
        return jsonify({"error": "Folder not found"}), 400

    videos, photos = [], []
    for f in sorted(os.listdir(path)):
        ext = os.path.splitext(f)[1].lower()
        fp = os.path.join(path, f)
        if not os.path.isfile(fp): continue
        size = round(os.path.getsize(fp) / 1048576, 1)
        if ext in VIDEO_EXT: videos.append({"name": f, "size_mb": size})
        elif ext in PHOTO_EXT: photos.append({"name": f, "size_mb": size})

    return jsonify({"path": path, "videos": videos, "photos": photos,
                    "total": len(videos) + len(photos)})

@app.route("/api/start", methods=["POST"])
def start_job():
    """Start processing with configured sources"""
    data = request.get_json()
    sources = data.get("sources", [])
    quality = data.get("quality", "medium")
    name = data.get("name", f"project_{int(time.time())}")

    if not sources:
        return jsonify({"error": "No sources"}), 400

    job_id = str(uuid.uuid4())[:8]
    proj = os.path.join(PROJECTS_DIR, job_id)
    os.makedirs(proj, exist_ok=True)

    # Validate sources exist
    valid_sources = []
    for src in sources:
        p = src.get("path", "")
        if os.path.exists(p):
            valid_sources.append(src)
        else:
            log.warning(f'[{job_id}] Source not found: {p}')

    if not valid_sources:
        return jsonify({"error": "No valid sources found"}), 400

    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["medium"])
    jobs[job_id] = {
        "id": job_id, "name": name, "quality": quality,
        "project_dir": proj, "status": "queued",
        "current_step": "upload", "step_progress": 0,
        "total_images": 0, "error": None, "warning": None,
        "created": time.time(), "updated": time.time(), "finished": None,
        "sources": valid_sources,
        "training_steps": preset["steps"],
        "step_times": {},
        "steps_def": STEPS,
    }
    job_queue.put(job_id)
    log.info(f'[{job_id}] Started with {len(valid_sources)} sources')
    return jsonify({"job_id": job_id})

# Legacy upload endpoint for browser file uploads
@app.route("/api/upload", methods=["POST"])
def upload():
    quality = request.form.get("quality", "medium")
    name = request.form.get("name", f"project_{int(time.time())}")
    frame_every = int(request.form.get("frame_every", 2))
    job_id = str(uuid.uuid4())[:8]
    proj = os.path.join(PROJECTS_DIR, job_id)
    uploads = os.path.join(proj, "uploads")
    os.makedirs(uploads, exist_ok=True)

    sources = []
    files = request.files.getlist("files")
    for f in files:
        if not f.filename: continue
        safe = os.path.basename(f.filename)
        dest = os.path.join(uploads, safe)
        f.save(dest)
        ext = os.path.splitext(safe)[1].lower()
        stype = "video" if ext in VIDEO_EXT else "photo"
        sources.append({"path": dest, "type": stype, "name": safe, "frame_every": frame_every})

    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["medium"])
    jobs[job_id] = {
        "id": job_id, "name": name, "quality": quality,
        "project_dir": proj, "status": "queued",
        "current_step": "upload", "step_progress": 0,
        "total_images": 0, "error": None, "warning": None,
        "created": time.time(), "updated": time.time(), "finished": None,
        "sources": sources,
        "training_steps": preset["steps"],
        "step_times": {},
        "steps_def": STEPS,
    }
    job_queue.put(job_id)
    return jsonify({"job_id": job_id})

@app.route("/api/job/<job_id>/quality", methods=["POST"])
def update_quality(job_id):
    if job_id not in jobs: return jsonify({"error": "Not found"}), 404
    data = request.get_json()
    if "training_steps" in data: jobs[job_id]["training_steps"] = int(data["training_steps"])
    if "quality" in data:
        preset = QUALITY_PRESETS.get(data["quality"])
        if preset:
            jobs[job_id]["quality"] = data["quality"]
            jobs[job_id]["training_steps"] = preset["steps"]
    return jsonify({"ok": True})

@app.route("/api/job/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    if job_id not in jobs: return jsonify({"error": "Not found"}), 404
    job = jobs[job_id]
    if job.get("status") == "processing":
        job["cancelled"] = True
        update_job(job_id, status="error", error="Отменено пользователем")
    proj = job.get("project_dir", "")
    if os.path.isdir(proj):
        shutil.rmtree(proj, ignore_errors=True)
    if job_id in jobs: del jobs[job_id]
    return jsonify({"ok": True})

@app.route("/api/detect-progress", methods=["POST"])
def api_detect_progress():
    """Detect progress of an existing project folder"""
    data = request.get_json()
    path = data.get("path", "").strip()
    if not path or not os.path.isdir(path):
        return jsonify({"error": "Folder not found"}), 400
    result = detect_progress(path)
    result["path"] = path
    result["name"] = os.path.basename(path)
    return jsonify(result)

@app.route("/api/resume", methods=["POST"])
def resume_job():
    """Resume a project from detected step"""
    data = request.get_json()
    path = data.get("path", "").strip()
    start_from = data.get("start_from", "")
    quality = data.get("quality", "medium")
    name = data.get("name", os.path.basename(path))

    if not path or not os.path.isdir(path):
        return jsonify({"error": "Folder not found"}), 400

    progress = detect_progress(path)
    if not start_from:
        start_from = progress["resume_from"]

    job_id = str(uuid.uuid4())[:8]
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["medium"])
    jobs[job_id] = {
        "id": job_id, "name": name, "quality": quality,
        "project_dir": path, "status": "queued",
        "current_step": "upload", "step_progress": 0,
        "total_images": progress.get("total_images", 0),
        "error": None, "warning": None,
        "created": time.time(), "updated": time.time(), "finished": None,
        "sources": [], "start_from": start_from,
        "training_steps": preset["steps"],
        "step_times": {}, "steps_def": STEPS,
    }
    job_queue.put(job_id)
    log.info(f'[{job_id}] RESUME "{name}" from {start_from}')
    return jsonify({"job_id": job_id, "start_from": start_from})

@app.route("/api/job/<job_id>/open-folder", methods=["POST"])
def open_folder(job_id):
    if job_id not in jobs: return jsonify({"error": "Not found"}), 404
    subprocess.Popen(["open", jobs[job_id]["project_dir"]])
    return jsonify({"ok": True})

@app.route("/api/job/<job_id>/ply")
def serve_ply(job_id):
    """Serve PLY file inline for 3D viewer"""
    if job_id not in jobs: return jsonify({"error": "Not found"}), 404
    proj = jobs[job_id]["project_dir"]
    ply = os.path.join(proj, "output.ply")
    if not os.path.exists(ply):
        return jsonify({"error": "PLY not found"}), 404
    return send_file(ply, mimetype="application/octet-stream")

@app.route("/viewer/<job_id>")
def viewer_page(job_id):
    if job_id not in jobs: return "Not found", 404
    return render_template("viewer.html", job_id=job_id, job_name=jobs[job_id].get("name", ""))

@app.route("/api/job/<job_id>/download")
def download_result(job_id):
    if job_id not in jobs: return jsonify({"error": "Not found"}), 404
    proj = jobs[job_id]["project_dir"]
    ply = os.path.join(proj, "output.ply")
    if os.path.exists(ply):
        return send_file(ply, as_attachment=True, download_name=f"{jobs[job_id]['name']}.ply")
    output_dir = os.path.join(proj, "colmap_output")
    if os.path.isdir(output_dir):
        tar_path = os.path.join(proj, "download.tar.gz")
        if not os.path.exists(tar_path):
            subprocess.run(["tar", "czf", tar_path, "-C", proj, "colmap_output"], capture_output=True)
        return send_file(tar_path, as_attachment=True, download_name=f"{jobs[job_id]['name']}_colmap.tar.gz")
    return jsonify({"error": "Результат не найден"}), 404

@app.route("/api/job/<job_id>/thumbnail")
def get_thumbnail(job_id):
    if job_id not in jobs: return jsonify({"error": "Not found"}), 404
    proj = jobs[job_id]["project_dir"]
    for d in ["images", "frames", "uploads"]:
        path = os.path.join(proj, d)
        if os.path.isdir(path):
            imgs = sorted(glob.glob(os.path.join(path, "*.jpg")) + glob.glob(os.path.join(path, "*.JPG")))
            if imgs: return send_file(imgs[0])
    return "", 404

@app.route("/api/jobs")
def get_jobs():
    return jsonify(list(jobs.values()))

@app.route("/api/history")
def get_history():
    return jsonify(load_history())

@app.route("/api/cache-stats")
def cache_stats():
    total, count = 0, 0
    if os.path.exists(PROJECTS_DIR):
        for d in os.listdir(PROJECTS_DIR):
            dp = os.path.join(PROJECTS_DIR, d)
            if os.path.isdir(dp):
                count += 1
                for root, dirs, files in os.walk(dp):
                    for f in files:
                        fp = os.path.join(root, f)
                        if not os.path.islink(fp): total += os.path.getsize(fp)
    return jsonify({"size_mb": round(total / 1048576), "projects": count})

@app.route("/api/clear-cache", methods=["POST"])
def clear_cache():
    removed, kept = 0, 0
    active_dirs = set()
    for jid, j in list(jobs.items()):
        if j.get("status") == "processing":
            active_dirs.add(os.path.basename(j.get("project_dir", "")))
            kept += 1
        else:
            del jobs[jid]
    if os.path.exists(PROJECTS_DIR):
        for d in os.listdir(PROJECTS_DIR):
            dp = os.path.join(PROJECTS_DIR, d)
            if os.path.isdir(dp) and d not in active_dirs:
                shutil.rmtree(dp, ignore_errors=True)
                removed += 1
    return jsonify({"ok": True, "removed": removed, "kept": kept})

@app.route("/api/cpu")
def get_cpu():
    return jsonify(cpu_history[-60:])

@app.route("/api/stream")
def stream():
    def gen():
        while True:
            data = {"jobs": list(jobs.values()), "cpu": cpu_history[-1] if cpu_history else {}}
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(2)
    return Response(gen(), mimetype="text/event-stream")

if __name__ == "__main__":
    print("\n  SplatMaker v5 (Local) - http://localhost:8800\n")
    subprocess.Popen(["open", "http://localhost:8800"])
    app.run(host="0.0.0.0", port=8800, debug=False, threaded=True)
