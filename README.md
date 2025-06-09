# Emotion recognition Demo

简单的表情识别示例脚本，使用 [DeepFace](https://github.com/serengil/deepface)。

## 运行

应用使用 [uv](https://docs.astral.sh/uv/) 管理。

```bash
uv run app_image.py
uv run app_video.py
```

## `data` 目录说明

- `data/frames/`：存储视频分析时的帧图片。
- `data/images/`：存储需要分析的图片。
- `data/results/`：存储图片分析结果。
- `data/videos/`：存储需要分析的视频。
