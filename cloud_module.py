#!/usr/bin/env python3
"""SplatMaker Cloud Module v6 - Upload video, extract + process on GPU"""
import json, os, glob, time, subprocess, logging
from vastai import VastAI

log = logging.getLogger("splat.cloud")

# GPU pipeline: extract frames FROM VIDEO + COLMAP (all on server)
GPU_PIPELINE = r'''#!/bin/bash
set -e
cd /workspace/splat
echo "PROGRESS:init:0"

# Install dependencies
apt-get update -qq && apt-get install -y -qq colmap sqlite3 ffmpeg > /dev/null 2>&1
echo "PROGRESS:init:100"

# Extract frames from video
echo "PROGRESS:frames:0"
mkdir -p images
FRAME_EVERY=${FRAME_EVERY:-2}
for VID in uploads/*; do
    ffmpeg -i "$VID" -vf "select=not(mod(n\,$FRAME_EVERY))" -vsync vfr -q:v 1 images/frame_%05d.jpg 2>&1 | \
        grep -oP 'frame=\s*\K\d+' | while read F; do echo "PROGRESS:frames:$F"; done
done
TOTAL=$(ls images/*.jpg 2>/dev/null | wc -l)
echo "PROGRESS:frames_done:$TOTAL"

# Database
mkdir -p colmap/sparse
DB=colmap/database.db

# Feature extraction (GPU CUDA)
echo "PROGRESS:features:0"
colmap feature_extractor \
    --database_path $DB \
    --image_path images/ \
    --ImageReader.camera_model OPENCV \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.max_image_size 3200 \
    --SiftExtraction.max_num_features 16384
echo "PROGRESS:features:100"

# Feature matching (GPU CUDA)
echo "PROGRESS:matching:0"
colmap exhaustive_matcher \
    --database_path $DB \
    --SiftMatching.use_gpu 1
echo "PROGRESS:matching:100"

# Sparse reconstruction
echo "PROGRESS:mapper:0"
colmap mapper \
    --database_path $DB \
    --image_path images/ \
    --output_path colmap/sparse/
echo "PROGRESS:mapper:100"

# Bundle adjustment
echo "PROGRESS:bundle:0"
colmap bundle_adjuster \
    --input_path colmap/sparse/0/ \
    --output_path colmap/sparse/0/
echo "PROGRESS:bundle:100"

# Undistortion
echo "PROGRESS:undistort:0"
colmap image_undistorter \
    --image_path images/ \
    --input_path colmap/sparse/0/ \
    --output_path colmap_output/
echo "PROGRESS:undistort:100"

# Pack result
echo "PROGRESS:pack:0"
tar czf /workspace/result.tar.gz colmap_output/ colmap/sparse/
echo "PROGRESS:pack:100"

echo "PIPELINE_DONE"
'''

STEP_NAMES = {
    "init": "Установка зависимостей",
    "frames": "Извлечение кадров",
    "frames_done": "Кадры извлечены",
    "features": "Извлечение фич (GPU)",
    "matching": "Сопоставление (GPU)",
    "mapper": "3D реконструкция",
    "bundle": "Оптимизация",
    "undistort": "Коррекция",
    "pack": "Упаковка результата",
}

STEP_WEIGHTS = {
    "init": (0, 5),
    "frames": (5, 15),
    "features": (15, 40),
    "matching": (40, 60),
    "mapper": (60, 80),
    "bundle": (80, 88),
    "undistort": (88, 95),
    "pack": (95, 100),
}

ERRORS = {
    "no_video": "Нет видео/фото для обработки. Загрузите файлы.",
    "no_offers": "Нет доступных GPU RTX 4090 на Vast.ai. Попробуйте позже.",
    "rent_fail": "Не удалось арендовать GPU. Попробуйте другой сервер.",
    "boot_timeout": "Сервер не запустился за 10 минут. Возможно, хост перегружен.",
    "boot_crash": "Сервер упал при загрузке (статус: {status}). Выбран плохой хост.",
    "upload_fail": "Ошибка загрузки данных на сервер: {err}",
    "pipeline_fail": "Ошибка обработки на GPU: {err}",
    "pipeline_timeout": "Обработка на GPU заняла слишком долго (>2ч). Слишком много кадров?",
    "download_fail": "Ошибка скачивания результата: {err}",
}


def cloud_process(job, update_cb):
    """Cloud pipeline v6: upload VIDEO (not frames) -> extract + COLMAP on GPU"""

    vast = VastAI(raw=True)
    instance_id = None
    uploads_dir = os.path.join(job["project_dir"], "uploads")
    ssh_key = os.path.expanduser("~/.ssh/id_ed25519")
    frame_every = job.get("frame_every", 2)

    try:
        # -- Find source files --
        video_files = []
        for ext in ("*.mov", "*.mp4", "*.MOV", "*.MP4", "*.avi", "*.mkv"):
            video_files.extend(glob.glob(os.path.join(uploads_dir, ext)))

        if not video_files:
            update_cb("cloud_prep", 0, ERRORS["no_video"])
            return None

        # Calculate total upload size
        total_bytes = sum(os.path.getsize(f) if not os.path.islink(f) 
                         else os.path.getsize(os.path.realpath(f)) for f in video_files)
        size_mb = total_bytes / 1048576
        update_cb("cloud_prep", 100, f"Видео: {size_mb:.0f} MB (загрузим на GPU)")
        log.info(f"Video files: {len(video_files)}, {size_mb:.0f} MB total")

        # -- Step 2: Search & Rent GPU --
        update_cb("cloud_rent", 0, "Поиск дешёвого RTX 4090...")

        try:
            offers = vast.search_offers(
                query="gpu_name=RTX_4090 num_gpus=1 verified=true direct_port_count>=1 rentable=true inet_down>200 disk_space>50",
                order="dph",
                limit=5,
            )
        except Exception as e:
            log.error(f"Search offers failed: {e}")
            offers = []

        if not offers:
            update_cb("cloud_rent", 0, ERRORS["no_offers"])
            return None

        offer = offers[0]
        offer_id = offer["id"]
        dph = offer.get("dph_total", 0)
        update_cb("cloud_rent", 20, f"RTX 4090: ${dph:.2f}/час")
        log.info(f"Best offer: #{offer_id} at ${dph:.2f}/hr")

        # Disk: video + extracted frames (~5x) + results
        disk_gb = max(int(size_mb / 1024 * 8) + 20, 50)

        # Try up to 3 different servers
        for offer_idx, offer in enumerate(offers):
            offer_id = offer["id"]
            dph = offer.get("dph_total", 0)
            update_cb("cloud_rent", 20, f"Сервер {offer_idx+1}/{len(offers)}: ${dph:.2f}/час")
            log.info(f"Trying offer #{offer_id} at ${dph:.2f}/hr")

            try:
                result = vast.create_instance(
                    id=offer_id,
                    image="nvidia/cuda:12.2.0-devel-ubuntu22.04",
                    disk=disk_gb,
                    onstart_cmd="echo 'INSTANCE_READY'",
                )
            except Exception as e:
                log.error(f"Create instance failed: {e}")
                continue

            if not result.get("success"):
                log.error(f"Offer #{offer_id} rejected: {result}")
                continue

            instance_id = result.get("new_contract")
            update_cb("cloud_rent", 40, f"Инстанс #{instance_id}, загрузка...")
            log.info(f"Instance created: #{instance_id}")

            # Wait for running (max 3 min per server)
            booted = False
            for attempt in range(36):  # 36 * 5s = 3 min
                try:
                    info = vast.show_instance(id=instance_id)
                    status = info.get("actual_status", "unknown")
                except Exception:
                    status = "loading"

                pct = min(40 + attempt * 2, 99)
                status_ru = {"loading": "Загрузка образа", "created": "Подготовка GPU", "running": "Готов"}.get(status, status)
                elapsed_s = (attempt + 1) * 5
                update_cb("cloud_rent", pct, f"{status_ru}... ({elapsed_s}с)")

                if status == "running":
                    booted = True
                    break
                if status in ("exited", "offline", "error"):
                    log.error(f"Instance #{instance_id} crashed: {status}")
                    break
                time.sleep(5)

            if booted:
                break
            
            # This server failed - destroy and try next
            log.warning(f"Server #{offer_id} too slow, trying next...")
            update_cb("cloud_rent", 20, f"Сервер {offer_idx+1} слишком медленный, пробую другой...")
            vast.destroy_instance(id=instance_id)
            instance_id = None
            time.sleep(2)
        else:
            update_cb("cloud_rent", 0, ERRORS["boot_timeout"])
            return None

        update_cb("cloud_rent", 100, "GPU готов!")
        log.info(f"Instance #{instance_id} is running")

        # -- Step 3: Upload VIDEO via SCP (1.5 GB instead of 6.4 GB!) --
        update_cb("cloud_upload", 0, f"Загрузка видео {size_mb:.0f} MB на GPU...")

        ssh_url = vast.ssh_url(id=instance_id).strip().replace("ssh://", "")
        log.info(f"SSH URL: {ssh_url}")

        user_host, port = ssh_url.rsplit(":", 1)
        user, host = user_host.split("@")
        ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
                    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
                    "-i", ssh_key]
        ssh_base = ["ssh"] + ssh_opts + ["-p", port, f"{user}@{host}"]

        # Wait for SSH to be ready (up to 60 sec)
        update_cb("cloud_upload", 0, "Ожидание SSH...")
        for ssh_try in range(12):
            r = subprocess.run(ssh_base + ["echo OK"], capture_output=True, timeout=15)
            if r.returncode == 0:
                log.info(f"SSH ready after {(ssh_try+1)*5}s")
                break
            log.info(f"SSH not ready yet (try {ssh_try+1}/12)...")
            update_cb("cloud_upload", 0, f"SSH: попытка {ssh_try+1}/12...")
            time.sleep(5)
        else:
            update_cb("cloud_upload", 0, "SSH не доступен после 60 сек")
            log.error("SSH never became ready")
            vast.destroy_instance(id=instance_id)
            return None

        # Create workspace
        subprocess.run(ssh_base + ["mkdir -p /workspace/splat/uploads"], capture_output=True, timeout=30)

        # Upload each video file via SCP (with retry)
        for i, vf in enumerate(video_files):
            real_path = os.path.realpath(vf) if os.path.islink(vf) else vf
            fname = os.path.basename(vf)
            vf_mb = os.path.getsize(real_path) / 1048576
            update_cb("cloud_upload", int(i / len(video_files) * 80), f"SCP: {fname} ({vf_mb:.0f} MB)...")
            log.info(f"Uploading {fname}: {vf_mb:.0f} MB")

            scp_cmd = [
                "scp", "-o", "StrictHostKeyChecking=no", "-i", ssh_key, "-P", port,
                real_path, f"{user}@{host}:/workspace/splat/uploads/{fname}"
            ]
            
            # Retry SCP up to 3 times
            scp_ok = False
            for attempt in range(3):
                proc = subprocess.run(scp_cmd, capture_output=True, timeout=3600)
                if proc.returncode == 0:
                    scp_ok = True
                    break
                err = proc.stderr.decode("utf-8", errors="replace")[:200] if proc.stderr else "SCP error"
                log.warning(f"SCP attempt {attempt+1}/3 failed: {err}")
                update_cb("cloud_upload", int(i / len(video_files) * 80), f"SCP retry {attempt+2}/3...")
                time.sleep(5)
            
            if not scp_ok:
                err = proc.stderr.decode("utf-8", errors="replace")[:200] if proc.stderr else "SCP error"
                update_cb("cloud_upload", 0, ERRORS["upload_fail"].format(err=err))
                log.error(f"SCP failed after 3 attempts: {err}")
                vast.destroy_instance(id=instance_id)
                return None

        update_cb("cloud_upload", 85, "Видео загружено, отправка скрипта...")
        log.info("Video uploaded OK")

        # Upload pipeline script
        script_path = os.path.join(job["project_dir"], "gpu_pipeline.sh")
        with open(script_path, "w") as f:
            f.write(GPU_PIPELINE)

        scp_script = [
            "scp", "-o", "StrictHostKeyChecking=no", "-i", ssh_key, "-P", port,
            script_path, f"{user}@{host}:/workspace/splat/gpu_pipeline.sh"
        ]
        subprocess.run(scp_script, capture_output=True, timeout=60)
        subprocess.run(ssh_base + ["chmod +x /workspace/splat/gpu_pipeline.sh"], capture_output=True, timeout=10)

        update_cb("cloud_upload", 100, "Данные загружены!")

        # -- Step 4: Run pipeline via nohup (SSH-independent) --
        update_cb("cloud_gpu", 0, "Запуск обработки на RTX 4090...")
        log.info("Starting GPU pipeline via nohup...")

        # Launch script in background with nohup, output to log file
        launch_cmd = (
            f"nohup bash -c 'FRAME_EVERY={frame_every} "
            f"bash /workspace/splat/gpu_pipeline.sh > /workspace/splat/pipeline.log 2>&1 "
            f"&& echo PIPELINE_DONE >> /workspace/splat/pipeline.log' &"
        )
        subprocess.run(ssh_base + [launch_cmd], capture_output=True, timeout=15)
        time.sleep(3)  # Give it a moment to start

        # Poll log file for progress
        pipeline_done = False
        last_line_count = 0
        timeout_start = time.time()

        while time.time() - timeout_start < 7200:  # 2 hour max
            # Read new lines from remote log
            try:
                r = subprocess.run(
                    ssh_base + [f"tail -n +{last_line_count + 1} /workspace/splat/pipeline.log 2>/dev/null"],
                    capture_output=True, timeout=30
                )
                if r.returncode != 0:
                    time.sleep(5)
                    continue
                output = r.stdout.decode("utf-8", errors="replace")
            except Exception:
                time.sleep(5)
                continue

            new_lines = output.strip().split("\n") if output.strip() else []
            last_line_count += len(new_lines)

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("PROGRESS:"):
                    parts = line.split(":")
                    if len(parts) >= 3:
                        step = parts[1]
                        try:
                            pct = int(parts[2].split()[0])
                        except ValueError:
                            pct = 0

                        if step == "frames":
                            update_cb("cloud_gpu", 5 + min(pct // 10, 9), f"Извлечение кадров: {pct}")
                        elif step == "frames_done":
                            update_cb("cloud_gpu", 15, f"Извлечено {pct} кадров")
                            log.info(f"Extracted {pct} frames on GPU")
                        else:
                            step_name = STEP_NAMES.get(step, step)
                            w = STEP_WEIGHTS.get(step, (0, 100))
                            overall = w[0] + (w[1] - w[0]) * pct // 100
                            update_cb("cloud_gpu", overall, f"{step_name}: {pct}%")

                elif "PIPELINE_DONE" in line:
                    pipeline_done = True
                    update_cb("cloud_gpu", 100, "COLMAP завершён!")
                    log.info("Pipeline completed successfully")
                    break

            if pipeline_done:
                break

            time.sleep(10)  # Poll every 10 sec

        if not pipeline_done:
            update_cb("cloud_gpu", 0, ERRORS["pipeline_fail"].format(err="Pipeline не завершился"))
            log.error("Pipeline did not complete")
            vast.destroy_instance(id=instance_id)
            return None

        # -- Step 5: Download result --
        update_cb("cloud_download", 0, "Скачивание результата...")

        result_local = os.path.join(job["project_dir"], "result.tar.gz")
        scp_down = [
            "scp", "-o", "StrictHostKeyChecking=no", "-i", ssh_key, "-P", port,
            f"{user}@{host}:/workspace/result.tar.gz", result_local
        ]

        proc = subprocess.run(scp_down, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            err = proc.stderr[:200] if proc.stderr else "SCP download error"
            update_cb("cloud_download", 0, ERRORS["download_fail"].format(err=err))
            log.error(f"Download failed: {err}")
            vast.destroy_instance(id=instance_id)
            return None

        update_cb("cloud_download", 50, "Распаковка...")

        import tarfile
        with tarfile.open(result_local, "r:gz") as tar:
            tar.extractall(path=job["project_dir"])

        update_cb("cloud_download", 100, "Результат получен!")
        log.info("Result downloaded and unpacked")

        job["cloud_instance_id"] = instance_id
        job["cloud_dph"] = dph

        return True

    except Exception as e:
        log.exception(f"Cloud process error: {e}")
        update_cb("cloud_gpu", 0, f"Ошибка: {str(e)[:200]}")
        if instance_id:
            try:
                vast.destroy_instance(id=instance_id)
                log.info(f"Destroyed instance #{instance_id} after error")
            except Exception:
                pass
        return None


def destroy_instance(instance_id):
    """Safely destroy a Vast.ai instance"""
    try:
        vast = VastAI(raw=True)
        vast.destroy_instance(id=instance_id)
        log.info(f"Instance #{instance_id} destroyed")
        return True
    except Exception as e:
        log.error(f"Failed to destroy #{instance_id}: {e}")
        return False
