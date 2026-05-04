from typing import List
import yt_dlp
import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from langchain_core.tools import tool


# Get video stream URL
def get_video_stream_url(youtube_url: str) -> str:
    ydl_opts = {"format": "best[height<=720]", "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]

# Encode frame to base64
def encode_frame(frame) -> str:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# Scene-based frame extraction
def extract_scene_frames(youtube_url: str, max_frames: int = 20) -> List[str]:
    stream_url = get_video_stream_url(youtube_url)
    cap = cv2.VideoCapture(stream_url)

    frames = []
    prev_gray = None

    threshold = 30  #tune

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            continue

        diff = cv2.absdiff(prev_gray, gray)
        score = np.mean(diff)

        # Scene change detected
        if score > threshold:
            resized = cv2.resize(frame, (640, 360))
            frames.append(encode_frame(resized))
            prev_gray = gray

    cap.release()
    return frames



# Uniform fallback sampling
def extract_uniform_frames(youtube_url: str, num_frames: int = 5) -> List[str]:
    stream_url = get_video_stream_url(youtube_url)
    cap = cv2.VideoCapture(stream_url)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    interval = int(fps * 2)

    frames = []
    count = 0

    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            resized = cv2.resize(frame, (640, 360))
            frames.append(encode_frame(resized))

        count += 1

    cap.release()
    return frames



# Combined extractor
def extract_key_frames(youtube_url: str, max_frames: int = 20) -> List[str]:
    # Scene-based extraction
    frames = extract_scene_frames(youtube_url, max_frames)

    # Fallback if nothing found
    if len(frames) == 0:
        frames = extract_uniform_frames(youtube_url, num_frames=5)

    return frames


# Tool for LangGraph
@tool
def youtube_frame_extractor(url: str) -> dict:
    """
    Use this tool when the question requires visual understanding of a video

    Returns:
    - dict with base64 encoded frames
    """

    try:
        frames = extract_key_frames(url)

        return {
            "frames": frames
        }

    except Exception as e:
        return {
            "error": str(e)
        }