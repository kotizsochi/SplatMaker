# SplatMaker

Video/Photo to 3D Gaussian Splat -- local processing pipeline.

## Features
- **Video to Splat**: Drag & drop video files, extract frames, run COLMAP, train Gaussian Splats
- **Per-source settings**: Each video gets individual frame extraction rate
- **Multiple sources**: Combine videos, photos, and folders in one project
- **Symlinks**: References originals instead of copying (saves disk space)
- **Resume**: Auto-detect progress and continue from any step
- **3D Viewer**: Built-in PLY point cloud viewer (three.js)
- **Real-time progress**: Live COLMAP output parsing with progress bars
- **Project metadata**: Saves settings, timings, and source info as project.json
- **CPU/RAM monitoring**: Live system resource chart

## Requirements
- Python 3.9+
- ffmpeg
- COLMAP
- brush_app (optional, for Gaussian Splat training)

## Install
```bash
pip install flask psutil
```

## Usage
```bash
python3 app.py
# Opens http://localhost:8800
```

Or use the launcher:
```bash
chmod +x Start.command
./Start.command
```

## Pipeline
1. Frame extraction (ffmpeg)
2. Feature extraction (COLMAP)
3. Feature matching (COLMAP)
4. 3D Reconstruction / Mapper (COLMAP)
5. Bundle Adjustment (COLMAP)
6. Image Undistortion (COLMAP)
7. Gaussian Splat Training (brush_app)

## Roadmap
See [ROADMAP.md](ROADMAP.md) for planned features including:
- Blur detection
- AI masking
- 360 video/photo support
- Parallel processing
- Telegram notifications
- And more...

## License
MIT
