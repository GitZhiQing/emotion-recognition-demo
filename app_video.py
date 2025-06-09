import os
from pathlib import Path
from collections import deque, Counter
from typing import Tuple, Deque

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


def get_major_emotion(emotion_queue: Deque[str]) -> str:
    """取队列中出现次数最多的情绪"""
    if not emotion_queue:
        return "未知"
    counter = Counter(emotion_queue)
    return counter.most_common(1)[0][0]


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


def main():
    """主程序入口"""
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"无法打开视频文件: {VIDEO_PATH}")
        return

    font = load_font()
    emotion_queue = deque(maxlen=EMOTION_QUEUE_SIZE)
    confidence_queue = deque(maxlen=EMOTION_QUEUE_SIZE)
    frame_count = 0
    img_save_num = 0
    last_error = ""

    # 当前帧的表情和置信度，保证每帧都刷新
    current_emotion = "未知"
    current_confidence = 0.0

    cv2.namedWindow("camera", 1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取画面，可能已到视频结尾")
                break

            frame_count += 1

            # 每 ANALYZE_INTERVAL 帧分析一次，分析结果立即用于当前帧
            if frame_count % ANALYZE_INTERVAL == 0:
                emotion, confidence, error = analyze_emotion(frame)
                if error and error != last_error:
                    print(f"Error: {error}")
                    last_error = error
                if not error:
                    current_emotion = emotion
                    current_confidence = confidence
                    emotion_queue.append(emotion)
                    confidence_queue.append(confidence)
            # 其余帧使用上一次分析结果
            major_emotion = current_emotion
            avg_confidence = current_confidence

            frame_disp = draw_overlay(frame, major_emotion, avg_confidence, font)
            cv2.imshow("camera", frame_disp)

            key = cv2.waitKey(10) & 0xFF
            if cv2.getWindowProperty("camera", cv2.WND_PROP_VISIBLE) < 1:
                print("窗口已关闭，退出...")
                break
            if key == 27 or key == ord("q"):
                print("退出...")
                break
            if key == ord(" "):
                img_save_num += 1
                save_frame(frame, img_save_num)
    except KeyboardInterrupt:
        print("键盘中断，退出程序...")
    except Exception as e:
        print(f"发生异常: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
