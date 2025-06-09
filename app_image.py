import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace

# ================== 配置与常量 ==================
EMOTION_QUEUE_SIZE = 7
ANALYZE_INTERVAL = 5  # 每 N 帧分析一次
FONT_PATH = r"c:\WINDOWS\Fonts\MSYH.TTC"
FONT_SIZE = 24

EMOTION_MAP = {
    "angry": "愤怒",
    "disgust": "厌恶",
    "fear": "恐惧",
    "happy": "高兴",
    "sad": "伤心",
    "surprise": "惊讶",
    "neutral": "中性",
    "Unknown": "未知",
}

PROJECT_DIR = Path(__file__).parent.resolve()
VIDEO_PATH = PROJECT_DIR / "data" / "videos" / "expressions.mp4"
os.environ["DEEPFACE_HOME"] = str(PROJECT_DIR)
os.environ["HTTP_PROXY"] = r"http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = r"http://127.0.0.1:7890"
# ===============================================


def analyze_emotion(frame: np.ndarray) -> Tuple[str, float, str]:
    """分析表情，返回中文表情名、置信度和错误信息"""
    try:
        results = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        eng_emotion = results[0]["dominant_emotion"]
        emotion_score = results[0]["emotion"][eng_emotion]
        return EMOTION_MAP.get(eng_emotion, eng_emotion), float(emotion_score), ""
    except Exception as e:
        return "未知", 0.0, str(e)


def draw_overlay(
    frame: np.ndarray, emotion: str, confidence: float, font: ImageFont.ImageFont
) -> np.ndarray:
    """在画面上绘制表情和置信度信息"""
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    draw.text((10, 50), f"表情: {emotion}", font=font, fill=(0, 255, 0))
    draw.text((10, 90), f"置信度: {confidence:.1f}%", font=font, fill=(0, 128, 255))
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)


def save_frame(frame: np.ndarray, img_save_num: int) -> None:
    """保存当前帧到本地"""
    data_dir = PROJECT_DIR / "data" / "frames"
    data_dir.mkdir(exist_ok=True)
    filename = data_dir / f"frames_{img_save_num}.jpg"
    cv2.imwrite(str(filename), frame)
    print(f"已保存图片: {filename}")


def load_font() -> ImageFont.ImageFont:
    """加载字体"""
    try:
        return ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception as e:
        print(f"字体加载失败: {e}")
        raise


def analyze_images_in_folder(src_dir: Path, dst_dir: Path):
    """批量分析指定目录下的所有图片，并保存带分析结果的图片到目标目录"""
    font = load_font()
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    img_files = [
        f
        for f in src_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ]
    print(f"共检测到 {len(img_files)} 张图片，开始分析...")

    for idx, img_path in enumerate(img_files, 1):
        # 用支持中文路径的方式读取图片
        try:
            img_array = np.fromfile(str(img_path), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"无法读取图片: {img_path}，错误: {e}")
            continue
        if frame is None:
            print(f"无法读取图片: {img_path}")
            continue
        emotion, confidence, error = analyze_emotion(frame)
        if error:
            print(f"[{img_path.name}] 分析失败: {error}")
            continue
        frame_disp = draw_overlay(frame, emotion, confidence, font)
        save_path = dst_dir / img_path.name
        # 用 imencode + tofile 支持中文路径
        ext = save_path.suffix
        result, encoded_img = cv2.imencode(ext, frame_disp)
        if result:
            encoded_img.tofile(str(save_path))
            print(f"[{idx}/{len(img_files)}] 已分析并保存: {save_path}")
        else:
            print(f"[{idx}/{len(img_files)}] 保存失败: {save_path}")


if __name__ == "__main__":
    src_folder = PROJECT_DIR / "data" / "images"  # 源图片目录
    dst_folder = PROJECT_DIR / "data" / "results"  # 结果保存目录
    analyze_images_in_folder(src_folder, dst_folder)
