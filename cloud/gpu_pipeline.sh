#!/bin/bash
# SplatMaker Cloud Pipeline - выполняется на GPU-сервере
# Автоматически: распаковка -> COLMAP (GPU) -> результат
set -e

WORK="/workspace/splat"
IMAGES="$WORK/images"
DB="$WORK/colmap/database.db"
SPARSE="$WORK/colmap/sparse"
OUTPUT="$WORK/output"
RESULT="$WORK/result.tar.gz"

mkdir -p "$IMAGES" "$WORK/colmap" "$SPARSE" "$OUTPUT"

log() { echo "[$(date +%H:%M:%S)] $1"; }
timer() { date +%s; }

START=$(timer)
log "=== SplatMaker Cloud Pipeline ==="
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 1. Распаковка
log "[1/6] Распаковка кадров..."
T1=$(timer)
if [ -f "$WORK/frames.tar.gz" ]; then
    tar xzf "$WORK/frames.tar.gz" -C "$IMAGES" --strip-components=1 2>/dev/null || tar xzf "$WORK/frames.tar.gz" -C "$IMAGES"
fi
TOTAL=$(ls "$IMAGES"/*.jpg 2>/dev/null | wc -l)
log "  Кадров: $TOTAL ($(( $(timer)-T1 )) сек)"

# 2. Feature Extraction (GPU)
log "[2/6] Feature Extraction (CUDA GPU)..."
T2=$(timer)
colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$IMAGES" \
    --ImageReader.camera_model OPENCV \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.max_num_features 8192
log "  Готово ($(( $(timer)-T2 )) сек)"

# 3. Sequential Matching (GPU)
log "[3/6] Sequential Matching (CUDA GPU)..."
T3=$(timer)
colmap sequential_matcher \
    --database_path "$DB" \
    --SiftMatching.use_gpu 1 \
    --SequentialMatching.overlap 15 \
    --SequentialMatching.quadratic_overlap 1
log "  Готово ($(( $(timer)-T3 )) сек)"

# 4. Mapper
log "[4/6] Mapper (3D реконструкция)..."
T4=$(timer)
colmap mapper \
    --database_path "$DB" \
    --image_path "$IMAGES" \
    --output_path "$SPARSE" \
    --Mapper.ba_refine_focal_length 1 \
    --Mapper.ba_refine_extra_params 1 \
    --Mapper.multiple_models 0
log "  Готово ($(( $(timer)-T4 )) сек)"

# 5. Bundle Adjustment
log "[5/6] Bundle Adjustment..."
T5=$(timer)
colmap bundle_adjuster \
    --input_path "$SPARSE/0" \
    --output_path "$SPARSE/0"
log "  Готово ($(( $(timer)-T5 )) сек)"

# 6. Image Undistorter
log "[6/6] Image Undistorter..."
T6=$(timer)
colmap image_undistorter \
    --image_path "$IMAGES" \
    --input_path "$SPARSE/0" \
    --output_path "$OUTPUT" \
    --output_type COLMAP
log "  Готово ($(( $(timer)-T6 )) сек)"

# Пакуем результат
log "Упаковка результата..."
cd "$OUTPUT" && tar czf "$RESULT" sparse/ images/ 2>/dev/null || tar czf "$RESULT" .
RESULT_SIZE=$(du -sh "$RESULT" | cut -f1)

TOTAL_TIME=$(( $(timer)-START ))
log "==================================="
log "ГОТОВО за ${TOTAL_TIME} сек ($(( TOTAL_TIME/60 )) мин)"
log "Кадров: $TOTAL"
log "Результат: $RESULT ($RESULT_SIZE)"
log "==================================="
echo "PIPELINE_DONE"
