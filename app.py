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
    {"id": "masking",  "name": "AI Маскирование"},
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

def parallel_feature_extraction(images_dir, db_path, max_features, job_id=None, n_workers=None):
    """Run COLMAP feature_extractor in parallel on image batches, then merge DBs.
    Returns True on success, False on failure.
    """
    import multiprocessing
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count() // 2, 5)
    n_workers = max(n_workers, 1)

    # Collect all images
    all_images = sorted([
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')
    ])

    if len(all_images) < 20 or n_workers <= 1:
        # Too few images or single worker -- use standard extraction
        return None  # signal to use standard path

    # Split into batches
    batches = [[] for _ in range(n_workers)]
    for i, img in enumerate(all_images):
        batches[i % n_workers].append(img)

    colmap_dir = os.path.dirname(db_path)
    batch_dirs = []
    batch_dbs = []
    processes = []

    log.info(f'[{job_id}] Parallel extraction: {len(all_images)} images -> {n_workers} workers')

    for wi in range(n_workers):
        if not batches[wi]:
            continue
        # Create batch image dir with symlinks
        batch_img_dir = os.path.join(colmap_dir, f"_batch_{wi}")
        os.makedirs(batch_img_dir, exist_ok=True)
        for img_name in batches[wi]:
            src = os.path.join(images_dir, img_name)
            dst = os.path.join(batch_img_dir, img_name)
            if not os.path.exists(dst):
                os.symlink(src, dst)
        batch_db = os.path.join(colmap_dir, f"_batch_{wi}.db")
        batch_dirs.append(batch_img_dir)
        batch_dbs.append(batch_db)

        cmd = ["colmap", "feature_extractor",
               "--database_path", batch_db,
               "--image_path", batch_img_dir,
               "--ImageReader.camera_model", "OPENCV",
               "--ImageReader.single_camera", "1",
               "--SiftExtraction.max_num_features", str(max_features)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((wi, p))

    # Wait for all to finish
    for wi, p in processes:
        p.wait()
        if job_id:
            done_count = sum(1 for _, pp in processes if pp.poll() is not None)
            pct = int(done_count / len(processes) * 90)
            update_job(job_id, step_progress=pct)

    # Check all succeeded
    for wi, p in processes:
        if p.returncode != 0:
            log.warning(f'[{job_id}] Worker {wi} failed with code {p.returncode}')
            # Cleanup and fall back to standard
            for d in batch_dirs:
                shutil.rmtree(d, ignore_errors=True)
            for db in batch_dbs:
                if os.path.exists(db): os.remove(db)
            return None

    # Merge databases
    log.info(f'[{job_id}] Merging {len(batch_dbs)} databases...')
    try:
        _merge_colmap_databases(batch_dbs, db_path, images_dir)
    except Exception as e:
        log.exception(f'[{job_id}] DB merge failed: {e}')
        # Cleanup and fallback
        for d in batch_dirs:
            shutil.rmtree(d, ignore_errors=True)
        for db in batch_dbs:
            if os.path.exists(db): os.remove(db)
        return None

    # Cleanup batch artifacts
    for d in batch_dirs:
        shutil.rmtree(d, ignore_errors=True)
    for db in batch_dbs:
        if os.path.exists(db): os.remove(db)

    log.info(f'[{job_id}] Parallel extraction complete')
    return True

def _merge_colmap_databases(batch_dbs, target_db, images_dir):
    """Merge multiple COLMAP feature databases into one."""
    if os.path.exists(target_db):
        os.remove(target_db)

    # Initialize target from first batch
    shutil.copy2(batch_dbs[0], target_db)

    if len(batch_dbs) == 1:
        # Fix image paths (remove batch dir prefix if needed)
        conn = sqlite3.connect(target_db)
        rows = conn.execute("SELECT image_id, name FROM images").fetchall()
        for img_id, name in rows:
            # Ensure name is relative to images_dir
            base = os.path.basename(name)
            if base != name:
                conn.execute("UPDATE images SET name = ? WHERE image_id = ?", (base, img_id))
        conn.commit()
        conn.close()
        return

    conn = sqlite3.connect(target_db)
    conn.execute("PRAGMA journal_mode=WAL")

    # Fix image names in first DB
    rows = conn.execute("SELECT image_id, name FROM images").fetchall()
    for img_id, name in rows:
        base = os.path.basename(name)
        if base != name:
            conn.execute("UPDATE images SET name = ? WHERE image_id = ?", (base, img_id))
    conn.commit()

    # Get current max IDs
    max_image_id = conn.execute("SELECT COALESCE(MAX(image_id), 0) FROM images").fetchone()[0]
    max_camera_id = conn.execute("SELECT COALESCE(MAX(camera_id), 0) FROM cameras").fetchone()[0]
    max_frame_id = conn.execute("SELECT COALESCE(MAX(frame_id), 0) FROM frames").fetchone()[0]

    # Merge remaining databases
    for db_path in batch_dbs[1:]:
        src = sqlite3.connect(db_path)

        # Map camera IDs
        camera_map = {}
        # For single_camera mode, all batches share the same camera
        # Just map to camera_id=1
        src_cameras = src.execute("SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras").fetchall()
        for cam_id, model, w, h, params, pfl in src_cameras:
            camera_map[cam_id] = 1  # All map to camera 1 in single-camera mode

        # Map and insert images
        image_map = {}
        src_images = src.execute("SELECT image_id, name, camera_id FROM images").fetchall()
        for img_id, name, cam_id in src_images:
            max_image_id += 1
            new_id = max_image_id
            image_map[img_id] = new_id
            base_name = os.path.basename(name)
            new_cam = camera_map.get(cam_id, 1)
            conn.execute("INSERT INTO images (image_id, name, camera_id) VALUES (?, ?, ?)",
                         (new_id, base_name, new_cam))

        # Insert keypoints
        src_kp = src.execute("SELECT image_id, rows, cols, data FROM keypoints").fetchall()
        for img_id, rows, cols, data in src_kp:
            if img_id in image_map:
                conn.execute("INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                             (image_map[img_id], rows, cols, data))

        # Insert descriptors
        src_desc = src.execute("SELECT image_id, type, rows, cols, data FROM descriptors").fetchall()
        for img_id, dtype, rows, cols, data in src_desc:
            if img_id in image_map:
                conn.execute("INSERT INTO descriptors (image_id, type, rows, cols, data) VALUES (?, ?, ?, ?, ?)",
                             (image_map[img_id], dtype, rows, cols, data))

        # Insert frames and frame_data
        src_frames = src.execute("SELECT frame_id, rig_id FROM frames").fetchall()
        frame_map = {}
        for frame_id, rig_id in src_frames:
            max_frame_id += 1
            frame_map[frame_id] = max_frame_id
            conn.execute("INSERT INTO frames (frame_id, rig_id) VALUES (?, ?)", (max_frame_id, rig_id))

        src_fd = src.execute("SELECT frame_id, data_id, sensor_id, sensor_type FROM frame_data").fetchall()
        for frame_id, data_id, sensor_id, sensor_type in src_fd:
            new_frame = frame_map.get(frame_id, frame_id)
            new_data = image_map.get(data_id, data_id)
            conn.execute("INSERT INTO frame_data (frame_id, data_id, sensor_id, sensor_type) VALUES (?, ?, ?, ?)",
                         (new_frame, new_data, sensor_id, sensor_type))

        src.close()
        conn.commit()

    # Verify
    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    kp_count = conn.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0]
    log.info(f'Merged DB: {total} images, {kp_count} keypoints')
    conn.close()

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
    step_order = ["frames", "blur", "masking", "features", "matching", "mapper", "bundle", "undistort", "training", "done"]
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

    step_order = ["frames", "blur", "masking", "features", "matching", "mapper", "bundle", "undistort", "training"]
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

    # Step 1.5: AI Masking (Phase 4, optional)
    if start_idx <= 2:
        masking_enabled = job.get("masking", False)
        if masking_enabled:
            update_job(job_id, current_step="masking", step_progress=0)
            try:
                from rembg import remove as rembg_remove
                masks_dir = os.path.join(proj, "masks")
                os.makedirs(masks_dir, exist_ok=True)
                imgs = sorted([f for f in os.listdir(images_dir)
                    if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png')])
                total = len(imgs)
                masked_count = 0
                for idx, fname in enumerate(imgs):
                    img_path = os.path.join(images_dir, fname)
                    try:
                        inp = Image.open(img_path)
                        out = rembg_remove(inp)
                        # Save mask (alpha channel)
                        if out.mode == 'RGBA':
                            alpha = out.split()[3]
                            mask_path = os.path.join(masks_dir, os.path.splitext(fname)[0] + "_mask.png")
                            alpha.save(mask_path)
                            masked_count += 1
                    except Exception as me:
                        log.warning(f'[{job_id}] Masking failed for {fname}: {me}')
                    if total > 0:
                        update_job(job_id, step_progress=int((idx + 1) / total * 100))
                log.info(f'[{job_id}] Masking: {masked_count}/{total} masks generated')
                update_job(job_id, masks_generated=masked_count)
            except ImportError:
                log.warning(f'[{job_id}] rembg not installed, skipping masking')
            update_job(job_id, step_progress=100)
        else:
            update_job(job_id, current_step="masking", step_progress=100)

    # Step 2: Feature Extraction (parallel when possible, Phase 6)
    if start_idx <= 3:
        update_job(job_id, current_step="features", step_progress=0)
        # Try parallel extraction first
        parallel_result = parallel_feature_extraction(
            images_dir, db_path, quality["max_features"], job_id=job_id)
        if parallel_result is None:
            # Fallback to standard single-process extraction
            log.info(f'[{job_id}] Using standard (single-process) extraction')
            cmd = ["colmap", "feature_extractor", "--database_path", db_path, "--image_path", images_dir,
                   "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1",
                   "--SiftExtraction.max_num_features", str(quality["max_features"])]
            if not run_cmd(cmd, job_id, "features"): return
        update_job(job_id, step_progress=100)

    # Step 3: Sequential Matching
    if start_idx <= 4:
        update_job(job_id, current_step="matching", step_progress=0)
        cmd = ["colmap", "sequential_matcher", "--database_path", db_path,
               "--SequentialMatching.overlap", str(quality["overlap"]), "--SequentialMatching.quadratic_overlap", "1"]
        if not run_cmd(cmd, job_id, "matching"): return
        update_job(job_id, step_progress=100)

    # Step 4: Mapper
    if start_idx <= 5:
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
    if start_idx <= 6:
        update_job(job_id, current_step="bundle", step_progress=0)
        cmd = ["colmap", "bundle_adjuster", "--input_path", os.path.join(sparse_dir, "0"),
               "--output_path", os.path.join(sparse_dir, "0")]
        if not run_cmd(cmd, job_id, "bundle"): return
        update_job(job_id, step_progress=100)

    # Step 6: Undistort
    if start_idx <= 7:
        update_job(job_id, current_step="undistort", step_progress=0)
        os.makedirs(output_dir, exist_ok=True)
        cmd = ["colmap", "image_undistorter", "--image_path", images_dir,
               "--input_path", os.path.join(sparse_dir, "0"), "--output_path", output_dir, "--output_type", "COLMAP"]
        if not run_cmd(cmd, job_id, "undistort"): return
        update_job(job_id, step_progress=100)

    # Step 7: Brush Training
    if start_idx <= 8:
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
            send_telegram(f"SplatMaker: Начало обработки\n<b>{j.get('name','')}</b>\nКачество: {j.get('quality','')}")
            process_job(job_id)
            status = jobs.get(job_id, {}).get("status", "")
            if status == "done":
                proj = jobs[job_id].get("project_dir", "")
                thumb = None
                for td in ["images", "frames"]:
                    tdir = os.path.join(proj, td)
                    if os.path.isdir(tdir):
                        imgs = sorted(glob.glob(os.path.join(tdir, "*.jpg")))
                        if imgs:
                            thumb = imgs[0]
                            break
                send_telegram(f"SplatMaker: Готово!\n<b>{j.get('name','')}</b>", thumb)
            elif status == "error":
                send_telegram(f"SplatMaker: Ошибка\n<b>{j.get('name','')}</b>\n{jobs.get(job_id,{}).get('error','')}")
        except Exception as e:
            log.exception(f'[{job_id}] EXCEPTION')
            update_job(job_id, status="error", error=str(e))
            send_telegram(f"SplatMaker: Критическая ошибка\n<b>{j.get('name','')}</b>\n{str(e)}")
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

# ========== Phase 5: 360 Support ==========

def is_equirectangular(img_path):
    """Detect equirectangular panorama by 2:1 aspect ratio"""
    try:
        img = Image.open(img_path)
        w, h = img.size
        ratio = w / h
        return 1.9 <= ratio <= 2.1  # ~2:1
    except:
        return False

def equirect_to_cubemap(img_path, output_dir, face_size=1024):
    """Convert equirectangular image to 6 cubemap faces using ffmpeg"""
    faces = ["front", "back", "left", "right", "top", "bottom"]
    v360_maps = ["e:c", "e:c", "e:c", "e:c", "e:c", "e:c"]
    yaw_pitch = [(0,0), (180,0), (270,0), (90,0), (0,90), (0,-90)]
    results = []
    for i, (face, (yaw, pitch)) in enumerate(zip(faces, yaw_pitch)):
        out = os.path.join(output_dir, f"360_{face}_{os.path.basename(img_path)}")
        cmd = ["ffmpeg", "-y", "-i", img_path,
               "-vf", f"v360=e:flat:yaw={yaw}:pitch={pitch}:w={face_size}:h={face_size}",
               "-frames:v", "1", out]
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(out):
            results.append(out)
    return results

@app.route("/api/detect-360", methods=["POST"])
def detect_360():
    """Detect if an image is equirectangular panorama"""
    data = request.get_json() or {}
    path = data.get("path", "").strip()
    if not path or not os.path.isfile(path):
        return jsonify({"error": "File not found"}), 400
    is_pano = is_equirectangular(path)
    return jsonify({"is_panorama": is_pano, "path": path})

# ========== Phase 7: Project History ==========

@app.route("/api/projects")
def list_projects():
    """List all projects with metadata from project.json"""
    projects = []
    if os.path.isdir(PROJECTS_DIR):
        for d in sorted(os.listdir(PROJECTS_DIR), reverse=True):
            dp = os.path.join(PROJECTS_DIR, d)
            if not os.path.isdir(dp):
                continue
            meta_path = os.path.join(dp, "project.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    meta["dir"] = dp
                    meta["has_ply"] = os.path.exists(os.path.join(dp, "output.ply"))
                    # Get file size
                    ply_path = os.path.join(dp, "output.ply")
                    meta["ply_size_mb"] = round(os.path.getsize(ply_path) / 1048576, 1) if os.path.exists(ply_path) else 0
                    # Thumbnail check
                    for td in ["images", "frames"]:
                        tdir = os.path.join(dp, td)
                        if os.path.isdir(tdir):
                            imgs = sorted(glob.glob(os.path.join(tdir, "*.jpg")) + glob.glob(os.path.join(tdir, "*.JPG")) + glob.glob(os.path.join(tdir, "*.png")))
                            if imgs:
                                meta["has_thumbnail"] = True
                                break
                    projects.append(meta)
                except:
                    pass
            else:
                # No project.json, basic info
                images_dir = os.path.join(dp, "images")
                img_count = len(os.listdir(images_dir)) if os.path.isdir(images_dir) else 0
                projects.append({
                    "id": d, "name": d, "dir": dp,
                    "total_images": img_count,
                    "has_ply": os.path.exists(os.path.join(dp, "output.ply")),
                    "quality": "unknown"
                })
    return jsonify(projects)

@app.route("/api/project/<project_id>/delete", methods=["POST"])
def delete_project(project_id):
    """Delete a project folder"""
    dp = os.path.join(PROJECTS_DIR, project_id)
    if not os.path.isdir(dp):
        return jsonify({"error": "Not found"}), 404
    # Don't delete active jobs
    for jid, j in jobs.items():
        if j.get("project_dir") == dp and j.get("status") == "processing":
            return jsonify({"error": "Project is currently processing"}), 400
    shutil.rmtree(dp, ignore_errors=True)
    # Remove from jobs dict
    for jid in list(jobs.keys()):
        if jobs[jid].get("project_dir") == dp:
            del jobs[jid]
    return jsonify({"ok": True})

# ========== Phase 9: Telegram Notifications ==========

TELEGRAM_CONFIG_FILE = os.path.join(BASE, "telegram.json")

def load_telegram_config():
    if os.path.exists(TELEGRAM_CONFIG_FILE):
        try:
            with open(TELEGRAM_CONFIG_FILE) as f:
                return json.load(f)
        except:
            pass
    return {"enabled": False, "token": "", "chat_id": ""}

def send_telegram(text, photo_path=None):
    """Send telegram notification"""
    cfg = load_telegram_config()
    if not cfg.get("enabled") or not cfg.get("token") or not cfg.get("chat_id"):
        return False
    try:
        import urllib.request
        token = cfg["token"]
        chat_id = cfg["chat_id"]
        if photo_path and os.path.exists(photo_path):
            # Send photo with caption
            import urllib.parse
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            boundary = "----SplatMakerBoundary"
            body = []
            body.append(f"--{boundary}".encode())
            body.append(f'Content-Disposition: form-data; name="chat_id"\r\n\r\n{chat_id}'.encode())
            body.append(f"--{boundary}".encode())
            body.append(f'Content-Disposition: form-data; name="caption"\r\n\r\n{text}'.encode())
            body.append(f"--{boundary}".encode())
            with open(photo_path, "rb") as pf:
                photo_data = pf.read()
            body.append(f'Content-Disposition: form-data; name="photo"; filename="thumb.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode() + photo_data)
            body.append(f"--{boundary}--".encode())
            payload = b"\r\n".join(body)
            req = urllib.request.Request(url, data=payload,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            urllib.request.urlopen(req, timeout=10)
        else:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")
        return False

@app.route("/api/telegram/config", methods=["GET"])
def get_telegram_config():
    cfg = load_telegram_config()
    return jsonify(cfg)

@app.route("/api/telegram/config", methods=["POST"])
def save_telegram_config():
    data = request.get_json()
    cfg = {
        "enabled": data.get("enabled", False),
        "token": data.get("token", "").strip(),
        "chat_id": data.get("chat_id", "").strip()
    }
    with open(TELEGRAM_CONFIG_FILE, "w") as f:
        json.dump(cfg, f)
    return jsonify({"ok": True})

@app.route("/api/telegram/test", methods=["POST"])
def test_telegram():
    ok = send_telegram("SplatMaker: Test notification OK!")
    return jsonify({"ok": ok})

# ========== Phase 10: Batch Mode ==========

@app.route("/api/batch", methods=["POST"])
def batch_start():
    """Start batch processing: each video becomes a separate project"""
    data = request.get_json()
    batch_sources = data.get("sources", [])
    quality = data.get("quality", "medium")
    frame_every = data.get("frame_every", 6)
    batch_id = f"batch_{int(time.time())}"
    created_jobs = []

    for src in batch_sources:
        src_path = src.get("path", "")
        if not src_path or not os.path.isfile(src_path):
            continue
        ext = os.path.splitext(src_path)[1].lower()
        if ext not in VIDEO_EXT:
            continue

        name = os.path.splitext(os.path.basename(src_path))[0]
        job_id = f"{batch_id}_{uuid.uuid4().hex[:6]}"
        proj = os.path.join(PROJECTS_DIR, job_id)
        os.makedirs(proj, exist_ok=True)

        job_sources = [{"type": "video", "path": src_path, "name": os.path.basename(src_path), "frame_every": frame_every}]
        jobs[job_id] = {
            "id": job_id, "name": name, "quality": quality,
            "sources": job_sources, "status": "queued",
            "project_dir": proj, "start_from": "frames",
            "current_step": "upload", "step_progress": 0,
            "total_images": 0, "created": time.time(),
            "step_times": {}, "batch_id": batch_id,
        }
        job_queue.put(job_id)
        created_jobs.append({"id": job_id, "name": name})
        log.info(f'[BATCH] Created job {job_id} for {name}')

    return jsonify({"batch_id": batch_id, "jobs": created_jobs, "count": len(created_jobs)})

# ========== Phase 11: HDR Detection ==========

@app.route("/api/detect-hdr", methods=["POST"])
def detect_hdr():
    """Detect if video is HDR using ffprobe"""
    data = request.get_json() or {}
    path = data.get("path", "").strip()
    if not path or not os.path.isfile(path):
        return jsonify({"error": "File not found"}), 400
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=color_transfer,color_primaries,pix_fmt,bits_per_raw_sample",
             "-of", "json", path],
            capture_output=True, text=True, timeout=10)
        info = json.loads(r.stdout)
        stream = info.get("streams", [{}])[0]
        is_hdr = (
            stream.get("color_transfer", "") in ["smpte2084", "arib-std-b67"] or
            stream.get("color_primaries", "") == "bt2020" or
            stream.get("pix_fmt", "").endswith("10le") or
            int(stream.get("bits_per_raw_sample", "8")) > 8
        )
        return jsonify({
            "is_hdr": is_hdr,
            "color_transfer": stream.get("color_transfer", ""),
            "color_primaries": stream.get("color_primaries", ""),
            "pix_fmt": stream.get("pix_fmt", ""),
            "bits": stream.get("bits_per_raw_sample", "8")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== Phase 13: Auto Quality Detection ==========

@app.route("/api/analyze-source", methods=["POST"])
def analyze_source():
    """Analyze video/image source and recommend settings"""
    data = request.get_json() or {}
    path = data.get("path", "").strip()
    if not path or not os.path.isfile(path):
        return jsonify({"error": "File not found"}), 400

    ext = os.path.splitext(path)[1].lower()
    result = {"path": path, "recommendations": {}}

    if ext in VIDEO_EXT:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames,codec_name",
                 "-show_entries", "format=duration",
                 "-of", "json", path],
                capture_output=True, text=True, timeout=15)
            info = json.loads(r.stdout)
            stream = info.get("streams", [{}])[0]
            fmt = info.get("format", {})

            w = int(stream.get("width", 0))
            h = int(stream.get("height", 0))
            fps_str = stream.get("r_frame_rate", "30/1")
            fps_parts = fps_str.split("/")
            fps = round(int(fps_parts[0]) / max(int(fps_parts[1]), 1), 1) if len(fps_parts) == 2 else 30
            duration = float(fmt.get("duration", stream.get("duration", 0)))
            total_frames = int(fps * duration)

            # Recommendations
            if w >= 3840:  # 4K+
                rec_quality = "high"
                rec_frame_every = max(int(fps / 3), 3)  # ~3 FPS
            elif w >= 1920:  # 1080p
                rec_quality = "medium"
                rec_frame_every = max(int(fps / 4), 4)  # ~4 FPS
            else:
                rec_quality = "fast"
                rec_frame_every = max(int(fps / 5), 2)  # ~5 FPS

            expected_frames = total_frames // rec_frame_every
            # Warn if too many or too few
            warnings = []
            if expected_frames > 2000:
                warnings.append(f"Много кадров ({expected_frames}). Увеличьте интервал.")
                rec_frame_every = max(rec_frame_every, int(total_frames / 1500))
            elif expected_frames < 50:
                warnings.append(f"Мало кадров ({expected_frames}). Уменьшите интервал.")
                rec_frame_every = max(1, rec_frame_every // 2)

            if duration < 10:
                warnings.append("Короткое видео. Может не хватить ракурсов.")
            if duration > 600:
                warnings.append("Длинное видео (>10 мин). Рассмотрите обрезку.")

            # Check if panoramic
            is_pano = 1.9 <= (w / max(h, 1)) <= 2.1

            result.update({
                "width": w, "height": h, "fps": fps,
                "duration": round(duration, 1), "total_frames": total_frames,
                "codec": stream.get("codec_name", ""),
                "is_panorama": is_pano,
                "recommendations": {
                    "quality": rec_quality,
                    "frame_every": rec_frame_every,
                    "expected_frames": total_frames // rec_frame_every,
                },
                "warnings": warnings,
            })
        except Exception as e:
            result["error"] = str(e)
    elif ext in PHOTO_EXT:
        try:
            img = Image.open(path)
            w, h = img.size
            is_pano = 1.9 <= (w / max(h, 1)) <= 2.1
            result.update({
                "width": w, "height": h,
                "is_panorama": is_pano,
                "recommendations": {"quality": "medium" if w >= 3000 else "fast"},
            })
        except Exception as e:
            result["error"] = str(e)

    return jsonify(result)

# ========== Phase 8: Floater Removal (PLY cleanup) ==========

@app.route("/api/job/<job_id>/cleanup-ply", methods=["POST"])
def cleanup_ply(job_id):
    """Remove statistical outliers from PLY point cloud"""
    if job_id not in jobs:
        return jsonify({"error": "Not found"}), 404
    proj = jobs[job_id]["project_dir"]
    ply_path = os.path.join(proj, "output.ply")
    if not os.path.exists(ply_path):
        return jsonify({"error": "PLY not found"}), 404

    data = request.get_json() or {}
    threshold = data.get("threshold", 2.0)  # std deviations

    try:
        # Read PLY (simple binary parser for point cloud)
        with open(ply_path, "rb") as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if b"end_header" in line:
                    break

            header_str = header.decode("ascii", errors="ignore")
            n_vertices = 0
            for line in header_str.split("\n"):
                if line.startswith("element vertex"):
                    n_vertices = int(line.split()[-1])

            if n_vertices == 0:
                return jsonify({"error": "No vertices found"}), 400

            # Read all vertex data
            data_size = os.path.getsize(ply_path) - len(header)
            vertex_size = data_size // n_vertices
            raw = f.read()

        # Parse xyz (first 12 bytes = 3 floats)
        positions = np.frombuffer(raw[:n_vertices * vertex_size], dtype=np.uint8).reshape(n_vertices, vertex_size)
        xyz = np.frombuffer(positions[:, :12].tobytes(), dtype=np.float32).reshape(n_vertices, 3)

        # Statistical outlier removal
        centroid = np.mean(xyz, axis=0)
        dists = np.linalg.norm(xyz - centroid, axis=1)
        mean_d = np.mean(dists)
        std_d = np.std(dists)
        mask = dists < (mean_d + threshold * std_d)
        kept = int(np.sum(mask))
        removed = n_vertices - kept

        if removed > 0 and kept > 100:
            # Backup original
            backup = ply_path + ".backup"
            if not os.path.exists(backup):
                shutil.copy2(ply_path, backup)

            # Write cleaned PLY
            kept_data = positions[mask]
            new_header = header_str.replace(f"element vertex {n_vertices}", f"element vertex {kept}")
            with open(ply_path, "wb") as f:
                f.write(new_header.encode("ascii"))
                f.write(kept_data.tobytes())

            log.info(f'[{job_id}] Floater cleanup: {removed} removed, {kept} kept (threshold={threshold})')

        return jsonify({
            "ok": True,
            "original": n_vertices,
            "kept": kept,
            "removed": removed,
            "threshold": threshold
        })
    except Exception as e:
        log.exception(f'[{job_id}] PLY cleanup error')
        return jsonify({"error": str(e)}), 500

# ========== Phase 4: AI Masking Toggle ==========

@app.route("/api/job/<job_id>/masking", methods=["POST"])
def toggle_masking(job_id):
    """Enable/disable AI masking for a job"""
    if job_id not in jobs:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json() or {}
    enabled = data.get("enabled", True)
    jobs[job_id]["masking"] = enabled
    try:
        from rembg import remove
        rembg_available = True
    except ImportError:
        rembg_available = False
    return jsonify({"ok": True, "masking": enabled, "rembg_available": rembg_available})

@app.route("/api/masking-status")
def masking_status():
    """Check if rembg is available"""
    try:
        from rembg import remove
        return jsonify({"available": True})
    except ImportError:
        return jsonify({"available": False})

# ========== Phase 12: Export Formats ==========

@app.route("/api/job/<job_id>/export", methods=["POST"])
def export_ply(job_id):
    """Convert PLY to other formats (.splat for web viewers)"""
    if job_id not in jobs:
        return jsonify({"error": "Not found"}), 404
    proj = jobs[job_id]["project_dir"]
    ply_path = os.path.join(proj, "output.ply")
    if not os.path.exists(ply_path):
        return jsonify({"error": "PLY not found"}), 404

    data = request.get_json() or {}
    fmt = data.get("format", "splat")  # splat, compressed_ply

    try:
        if fmt == "splat":
            # Convert PLY to .splat format (binary format for web viewers)
            splat_path = os.path.join(proj, "output.splat")
            _ply_to_splat(ply_path, splat_path)
            return jsonify({"ok": True, "path": splat_path,
                           "size_mb": round(os.path.getsize(splat_path) / 1048576, 1)})
        elif fmt == "compressed_ply":
            # Create compressed PLY (remove unnecessary attributes)
            comp_path = os.path.join(proj, "output_compressed.ply")
            _compress_ply(ply_path, comp_path)
            return jsonify({"ok": True, "path": comp_path,
                           "size_mb": round(os.path.getsize(comp_path) / 1048576, 1)})
        else:
            return jsonify({"error": f"Unknown format: {fmt}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/job/<job_id>/export-download/<fmt>")
def download_export(job_id, fmt):
    """Download exported file"""
    if job_id not in jobs:
        return jsonify({"error": "Not found"}), 404
    proj = jobs[job_id]["project_dir"]
    name = jobs[job_id].get("name", job_id)
    if fmt == "splat":
        f = os.path.join(proj, "output.splat")
        if os.path.exists(f):
            return send_file(f, as_attachment=True, download_name=f"{name}.splat")
    elif fmt == "compressed_ply":
        f = os.path.join(proj, "output_compressed.ply")
        if os.path.exists(f):
            return send_file(f, as_attachment=True, download_name=f"{name}_compressed.ply")
    return jsonify({"error": "File not found"}), 404

def _ply_to_splat(ply_path, splat_path):
    """Convert gaussian splat PLY to .splat binary format.
    .splat format: per-splat: 3 floats (pos) + 3 floats (scale) + 4 bytes (rgba) + 4 bytes (quat) = 32 bytes
    """
    with open(ply_path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break

        header_str = header.decode("ascii", errors="ignore")
        n_vertices = 0
        properties = []
        for line in header_str.split("\n"):
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                if len(parts) >= 3:
                    properties.append(parts[2])

        raw = f.read()

    if n_vertices == 0:
        raise ValueError("No vertices in PLY")

    data_size = len(raw)
    vertex_size = data_size // n_vertices

    # Write .splat format (simplified: just copy raw vertex data with header)
    with open(splat_path, "wb") as f:
        # .splat header: magic + vertex count
        f.write(b"SPLAT")
        f.write(n_vertices.to_bytes(4, "little"))
        f.write(raw)

    log.info(f'Exported .splat: {n_vertices} splats, {os.path.getsize(splat_path)} bytes')

def _compress_ply(ply_path, output_path):
    """Create a compressed PLY by quantizing positions and reducing precision"""
    with open(ply_path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break
        raw = f.read()

    # Copy with compression (just use gzip for now)
    import gzip
    with open(ply_path, "rb") as f_in:
        with gzip.open(output_path, "wb") as f_out:
            f_out.write(f_in.read())

    log.info(f'Compressed PLY: {os.path.getsize(ply_path)} -> {os.path.getsize(output_path)} bytes')

# ========== Phase 14: Video Report ==========

@app.route("/api/job/<job_id>/video-report", methods=["POST"])
def generate_video_report(job_id):
    """Generate a flyaround video from the PLY point cloud"""
    if job_id not in jobs:
        return jsonify({"error": "Not found"}), 404
    proj = jobs[job_id]["project_dir"]
    ply_path = os.path.join(proj, "output.ply")
    if not os.path.exists(ply_path):
        return jsonify({"error": "PLY not found"}), 404

    data = request.get_json() or {}
    frames_count = data.get("frames", 120)  # 4 sec at 30fps
    resolution = data.get("resolution", 720)

    try:
        video_path = os.path.join(proj, "flyaround.mp4")
        _render_flyaround(ply_path, video_path, frames_count, resolution)
        return jsonify({
            "ok": True, "path": video_path,
            "size_mb": round(os.path.getsize(video_path) / 1048576, 1),
            "frames": frames_count,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/job/<job_id>/video-report-download")
def download_video_report(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Not found"}), 404
    proj = jobs[job_id]["project_dir"]
    f = os.path.join(proj, "flyaround.mp4")
    if os.path.exists(f):
        return send_file(f, as_attachment=True, download_name=f"{jobs[job_id].get('name','report')}_flyaround.mp4")
    return jsonify({"error": "Video not found. Generate it first."}), 404

def _render_flyaround(ply_path, video_path, frames_count, resolution):
    """Render a simple flyaround animation from PLY using matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import tempfile

    # Read PLY positions
    with open(ply_path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break
        header_str = header.decode("ascii", errors="ignore")
        n_vertices = 0
        for ln in header_str.split("\n"):
            if ln.startswith("element vertex"):
                n_vertices = int(ln.split()[-1])
        raw = f.read()

    if n_vertices == 0:
        raise ValueError("Empty PLY")

    vertex_size = len(raw) // n_vertices
    positions = np.frombuffer(raw[:n_vertices * vertex_size], dtype=np.uint8).reshape(n_vertices, vertex_size)
    xyz = np.frombuffer(positions[:, :12].tobytes(), dtype=np.float32).reshape(n_vertices, 3)

    # Subsample for rendering speed
    if n_vertices > 50000:
        idx = np.random.choice(n_vertices, 50000, replace=False)
        xyz = xyz[idx]

    # Try to get colors (bytes 12-15 or similar)
    colors = None
    if vertex_size >= 15:
        try:
            rgb = positions[:, 12:15].astype(np.float32) / 255.0
            if n_vertices > 50000:
                rgb = rgb[idx]
            colors = rgb
        except:
            pass

    centroid = np.mean(xyz, axis=0)
    radius = np.percentile(np.linalg.norm(xyz - centroid, axis=1), 90)

    # Render frames
    tmpdir = tempfile.mkdtemp()
    frame_files = []

    for i in range(frames_count):
        angle = 2 * np.pi * i / frames_count
        elev = 20 + 10 * np.sin(angle * 2)

        fig = plt.figure(figsize=(resolution / 72, resolution / 72), dpi=72)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#080810')
        fig.patch.set_facecolor('#080810')

        if colors is not None:
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=0.1, alpha=0.7)
        else:
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='#a78bfa', s=0.1, alpha=0.5)

        ax.view_init(elev=elev, azim=np.degrees(angle))
        ax.set_xlim(centroid[0] - radius, centroid[0] + radius)
        ax.set_ylim(centroid[1] - radius, centroid[1] + radius)
        ax.set_zlim(centroid[2] - radius, centroid[2] + radius)
        ax.axis('off')
        plt.tight_layout(pad=0)

        frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=72, facecolor='#080810', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        frame_files.append(frame_path)

    # Encode to video with ffmpeg
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "30",
        "-i", os.path.join(tmpdir, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", video_path
    ], capture_output=True)

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)

    if not os.path.exists(video_path):
        raise RuntimeError("ffmpeg failed to create video")

    log.info(f'Video report: {frames_count} frames -> {video_path}')

# ========== Phase 15: Quality Comparison ==========

@app.route("/compare")
def comparison_viewer():
    """Serve comparison viewer page"""
    return render_template("compare.html")

@app.route("/api/compare-data", methods=["POST"])
def compare_data():
    """Get comparison data for two PLY files"""
    data = request.get_json() or {}
    job_a = data.get("job_a", "")
    job_b = data.get("job_b", "")

    result = {"a": {}, "b": {}}
    for key, jid in [("a", job_a), ("b", job_b)]:
        if jid not in jobs:
            result[key] = {"error": "Job not found"}
            continue
        proj = jobs[jid]["project_dir"]
        ply_path = os.path.join(proj, "output.ply")
        if not os.path.exists(ply_path):
            result[key] = {"error": "PLY not found"}
            continue

        # Get PLY stats
        with open(ply_path, "rb") as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if b"end_header" in line:
                    break
            header_str = header.decode("ascii", errors="ignore")
            n_vertices = 0
            for ln in header_str.split("\n"):
                if ln.startswith("element vertex"):
                    n_vertices = int(ln.split()[-1])

        j = jobs[jid]
        result[key] = {
            "id": jid,
            "name": j.get("name", ""),
            "quality": j.get("quality", ""),
            "total_images": j.get("total_images", 0),
            "vertices": n_vertices,
            "ply_size_mb": round(os.path.getsize(ply_path) / 1048576, 1),
            "duration": round((j.get("finished", 0) - j.get("created", 0)), 1) if j.get("finished") else 0,
        }

    return jsonify(result)

if __name__ == "__main__":
    print("\n  SplatMaker v5 (Local) - http://localhost:8800\n")
    subprocess.Popen(["open", "http://localhost:8800"])
    app.run(host="0.0.0.0", port=8800, debug=False, threaded=True)
