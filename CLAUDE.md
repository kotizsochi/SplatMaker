# SplatMaker - CLAUDE.md

## Проект
Локальный пайплайн для создания 3D Gaussian Splats из видео/фото.
Video/Photo -> ffmpeg -> COLMAP -> brush_app -> .ply

## GitHub
https://github.com/kotizsochi/SplatMaker

## Стек
- **Backend**: Python 3 + Flask (app.py, 1760 строк)
- **Frontend**: Vanilla HTML/JS/CSS (3 шаблона: index, viewer, compare)
- **CLI Tools**: ffmpeg, COLMAP, brush_app (WGPU/Metal)
- **AI**: rembg[cpu] (AI masking), numpy, PIL
- **DB**: SQLite (COLMAP internal)
- **Hardware**: M1 Max, 10 cores, 32GB RAM

## Запуск
```bash
cd /Users/dmac/Desktop/ПРОЕКТЫ/SplatMaker
python3 app.py
# -> http://localhost:8800
```

## Структура файлов
```
app.py              - Backend (1760 строк, 38 API endpoints)
templates/
  index.html        - Главный UI (вкладки: Обработка/Проекты/Настройки)
  viewer.html       - 3D PLY viewer (three.js + OrbitControls)
  compare.html      - Side-by-side comparison viewer
projects/           - Данные проектов (images, colmap, output.ply)
telegram.json       - Telegram bot config
history.json        - История задач
```

## Пайплайн (step_order)
1. **frames** - Извлечение кадров (ffmpeg, каждый N-й кадр)
2. **blur** - Фильтр размытия (Laplacian variance, авто-порог)
3. **masking** - AI маскирование (rembg, опциональный)
4. **features** - Feature Extraction (COLMAP, параллельный на 5 workers)
5. **matching** - Feature Matching (COLMAP sequential_matcher)
6. **mapper** - 3D Реконструкция (COLMAP mapper)
7. **bundle** - Bundle Adjustment
8. **undistort** - Image Undistorter
9. **training** - Gaussian Splat Training (brush_app, Metal GPU)

## API Endpoints (38 шт)

### Основные
- `POST /api/start` - Запуск обработки (sources, quality, name)
- `POST /api/upload` - Загрузка файлов (FormData)
- `POST /api/resume` - Возобновление проекта
- `GET /api/jobs` - Список задач
- `GET /api/stream` - SSE real-time обновления

### Анализ источников
- `POST /api/probe` - Probe видео (ffprobe)
- `POST /api/scan-folder` - Сканирование папки
- `POST /api/detect-progress` - Detect existing project
- `POST /api/analyze-source` - Автоопределение качества (Phase 13)
- `POST /api/detect-360` - Detect equirectangular (Phase 5)
- `POST /api/detect-hdr` - Detect HDR video (Phase 11)

### Job actions
- `POST /api/job/<id>/cancel` - Отмена
- `POST /api/job/<id>/open-folder` - Открыть в Finder
- `GET /api/job/<id>/download` - Скачать результат
- `GET /api/job/<id>/ply` - Скачать PLY
- `GET /api/job/<id>/thumbnail` - Thumbnail
- `POST /api/job/<id>/quality` - Изменить качество
- `POST /api/job/<id>/masking` - Toggle AI masking (Phase 4)
- `POST /api/job/<id>/cleanup-ply` - Floater removal (Phase 8)
- `POST /api/job/<id>/export` - Export .splat (Phase 12)
- `GET /api/job/<id>/export-download/<fmt>` - Download export
- `POST /api/job/<id>/video-report` - Video flyaround (Phase 14)
- `GET /api/job/<id>/video-report-download` - Download video

### Проекты и история
- `GET /api/projects` - Список проектов (Phase 7)
- `POST /api/project/<id>/delete` - Удалить проект
- `GET /api/history` - История
- `GET /api/cache-stats` - Статистика хранилища
- `POST /api/clear-cache` - Очистка кэша

### Telegram (Phase 9)
- `GET /api/telegram/config` - Получить конфиг
- `POST /api/telegram/config` - Сохранить конфиг
- `POST /api/telegram/test` - Тест-сообщение

### Batch и сравнение
- `POST /api/batch` - Пакетная обработка (Phase 10)
- `GET /api/masking-status` - Проверка rembg
- `POST /api/compare-data` - Данные для сравнения (Phase 15)
- `GET /api/cpu` - CPU/RAM мониторинг

## Качество (QUALITY_PRESETS)
| Preset | max_features | steps  | overlap |
|--------|-------------|--------|---------|
| fast   | 4096        | 7000   | 10      |
| medium | 8192        | 15000  | 15      |
| high   | 16384       | 30000  | 20      |
| ultra  | 16384       | 50000  | 25      |

## Phase 6: Parallel Feature Extraction
- Разбивает images на N батчей (N = CPU cores / 2 = 5)
- Каждый батч = symlinks + отдельная SQLite DB
- N процессов COLMAP feature_extractor параллельно
- SQL merge: cameras, images, keypoints, descriptors, frames
- Fallback на standard при <20 images или ошибке
- Ускорение ~2-3x

## Phase 4: AI Masking (rembg)
- Опциональный шаг (toggle в UI)
- Генерирует alpha-маски для каждого кадра
- Маски сохраняются в projects/<id>/masks/
- Требует: `pip install "rembg[cpu]"`

## Phase 14: Video Report
- matplotlib 3D scatter (subsample до 50k точек)
- 120 frames, orbit camera + elevation
- ffmpeg encode -> flyaround.mp4

## Telegram интеграция
- Уведомления: старт, готово (с thumbnail), ошибка
- Config: telegram.json (token, chat_id, enabled)

## Важные заметки
- COLMAP на Mac: без --use_gpu (нет CUDA)
- brush_app: WGPU/Metal для GPU-тренировки
- Symlinks вместо копирования файлов (экономия места)
- project.json: метаданные каждого проекта
- Автоудаление uploads/ после успешной обработки
- Resume с любого шага через detect_progress

## Зависимости (pip)
flask, numpy, Pillow, psutil, rembg[cpu], matplotlib, scipy

## UI структура (index.html)
- **Вкладка "Обработка"**: dropzone, sources, settings, jobs, CPU chart
- **Вкладка "Проекты"**: история с карточками, удаление
- **Вкладка "Настройки"**: Telegram config, хранилище
- **Ссылка "Сравнение"**: -> /compare (side-by-side viewer)
- **Job actions**: 3D просмотр, скачать, очистить PLY, экспорт .splat, видео-отчёт, папка
