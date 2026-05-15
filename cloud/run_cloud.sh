#!/bin/bash
# SplatMaker Cloud - Одна команда: видео -> Gaussian Splat на GPU
# Использование: ./run_cloud.sh /путь/к/видео.MOV [quality]
# quality: fast|medium|high|ultra (default: medium)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT="$1"
QUALITY="${2:-medium}"
SSH_KEY="$HOME/.ssh/id_ed25519"
PROJECT_NAME="splat_$(date +%Y%m%d_%H%M%S)"
WORK_DIR="/tmp/splatmaker_$PROJECT_NAME"

# Настройки качества (FPS для извлечения кадров)
case "$QUALITY" in
    fast)   FPS=5  ;;
    medium) FPS=10 ;;
    high)   FPS=20 ;;
    ultra)  FPS=0  ;; # 0 = все кадры
esac

log()  { echo -e "\033[1;35m[SplatMaker]\033[0m $1"; }
ok()   { echo -e "\033[1;32m[OK]\033[0m $1"; }
err()  { echo -e "\033[1;31m[ERROR]\033[0m $1"; exit 1; }
pbar() {
    local pct=$1 label=$2
    local filled=$((pct/5)) empty=$((20-filled))
    printf "\r  [%s%s] %3d%% %s" "$(printf '#%.0s' $(seq 1 $filled 2>/dev/null))" "$(printf '.%.0s' $(seq 1 $empty 2>/dev/null))" "$pct" "$label"
}

# Проверки
[ -z "$INPUT" ] && err "Использование: $0 /путь/к/видео.MOV [fast|medium|high|ultra]"
[ ! -f "$INPUT" ] && err "Файл не найден: $INPUT"
command -v vastai >/dev/null || err "vastai CLI не установлен (pip install vastai)"
command -v ffmpeg >/dev/null || err "ffmpeg не установлен"

log "=== SplatMaker Cloud ==="
log "Файл: $(basename "$INPUT")"
log "Качество: $QUALITY (FPS: ${FPS:-all})"
echo ""

# ============================================
# ШАГ 1: Извлечение кадров локально
# ============================================
log "[1/6] Извлечение кадров из видео..."
mkdir -p "$WORK_DIR/frames"

if [ "$FPS" -eq 0 ] 2>/dev/null; then
    ffmpeg -y -i "$INPUT" -qscale:v 2 "$WORK_DIR/frames/frame_%05d.jpg" 2>/dev/null
else
    ffmpeg -y -i "$INPUT" -vf "fps=$FPS" -qscale:v 1 -qmin 1 "$WORK_DIR/frames/frame_%05d.jpg" 2>/dev/null
fi

FRAME_COUNT=$(ls "$WORK_DIR/frames/"*.jpg 2>/dev/null | wc -l | tr -d ' ')
ok "$FRAME_COUNT кадров извлечено"

# ============================================
# ШАГ 2: Упаковка
# ============================================
log "[2/6] Упаковка..."
cp "$SCRIPT_DIR/gpu_pipeline.sh" "$WORK_DIR/"
cd "$WORK_DIR"
tar czf frames.tar.gz frames/
ARCHIVE_SIZE=$(du -sh frames.tar.gz | cut -f1)
ok "Архив: $ARCHIVE_SIZE"

# ============================================
# ШАГ 3: Аренда GPU
# ============================================
log "[3/6] Поиск дешёвого RTX 4090..."
OFFER_ID=$(vastai search offers 'gpu_name=RTX_4090 num_gpus=1 rentable=true inet_down>100 disk_space>30' -o 'dph' --limit 1 --raw 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['id'])")
PRICE=$(vastai search offers 'gpu_name=RTX_4090 num_gpus=1 rentable=true inet_down>100 disk_space>30' -o 'dph' --limit 1 --raw 2>/dev/null | python3 -c "import json,sys; print(f'{json.load(sys.stdin)[0][\"dph_total\"]:.2f}')")
ok "Найден: Offer #$OFFER_ID (\$$PRICE/час)"

log "Создание инстанса..."
RESULT=$(vastai create instance "$OFFER_ID" --image nvidia/cuda:12.2.0-devel-ubuntu22.04 --disk 50 --ssh --direct --onstart-cmd 'apt-get update -qq && apt-get install -y -qq colmap' 2>&1)
INSTANCE_ID=$(echo "$RESULT" | python3 -c "import json,sys,re; m=re.search(r'new_contract.: (\d+)', sys.stdin.read()); print(m.group(1))" 2>/dev/null)
[ -z "$INSTANCE_ID" ] && err "Не удалось создать инстанс: $RESULT"
ok "Инстанс #$INSTANCE_ID создан"

# Ждём запуска
log "Ожидание запуска..."
for i in $(seq 1 60); do
    STATUS=$(vastai show instances --raw 2>/dev/null | python3 -c "import json,sys; data=json.load(sys.stdin); print(next((i['actual_status'] for i in data if i['id']==$INSTANCE_ID), 'unknown'))" 2>/dev/null)
    pbar $((i*100/60)) "Статус: $STATUS"
    [ "$STATUS" = "running" ] && break
    sleep 5
done
echo ""

# Получение SSH
SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null | tr -d '\n')
SSH_HOST=$(echo "$SSH_URL" | sed 's|ssh://||' | cut -d: -f1 | cut -d@ -f2)
SSH_PORT=$(echo "$SSH_URL" | sed 's|ssh://||' | cut -d: -f2)
SSH_USER=$(echo "$SSH_URL" | sed 's|ssh://||' | cut -d@ -f1)
SSH_CMD="ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $SSH_PORT $SSH_USER@$SSH_HOST"
ok "SSH: $SSH_USER@$SSH_HOST:$SSH_PORT"

# Ждём SSH
sleep 10
for i in $(seq 1 12); do
    $SSH_CMD 'echo ready' 2>/dev/null && break
    sleep 5
done

# ============================================
# ШАГ 4: Загрузка данных
# ============================================
log "[4/6] Загрузка на GPU-сервер..."
$SSH_CMD 'mkdir -p /workspace/splat'
scp -i "$SSH_KEY" -P "$SSH_PORT" "$WORK_DIR/frames.tar.gz" "$WORK_DIR/gpu_pipeline.sh" "$SSH_USER@$SSH_HOST:/workspace/splat/"
ok "Данные загружены"

# ============================================
# ШАГ 5: Запуск pipeline на GPU
# ============================================
log "[5/6] COLMAP pipeline на RTX 4090..."
$SSH_CMD 'chmod +x /workspace/splat/gpu_pipeline.sh && bash /workspace/splat/gpu_pipeline.sh' 2>&1 | while read line; do
    echo "  [GPU] $line"
done
ok "Pipeline завершён!"

# ============================================
# ШАГ 6: Скачивание результата
# ============================================
log "[6/6] Скачивание результата..."
RESULT_DIR="$(dirname "$INPUT")/splat_result_$PROJECT_NAME"
mkdir -p "$RESULT_DIR"
scp -i "$SSH_KEY" -P "$SSH_PORT" "$SSH_USER@$SSH_HOST:/workspace/splat/result.tar.gz" "$RESULT_DIR/"
cd "$RESULT_DIR" && tar xzf result.tar.gz && rm result.tar.gz
ok "Результат: $RESULT_DIR"

# Удаление инстанса
log "Удаление инстанса..."
vastai destroy instance "$INSTANCE_ID" 2>/dev/null
ok "Инстанс удалён, деньги не тикают"

# Очистка
rm -rf "$WORK_DIR"

echo ""
log "========================================="
log "  ГОТОВО!"
log "  Результат: $RESULT_DIR"
log "  Откройте в SuperSplat: https://superspl.at"
log "========================================="
