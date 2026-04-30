from langchain_core.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi
import re

def extract_video_id(url: str) -> str:
    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?]+)"
    ]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube URL")


@tool
def youtube_transcript_qa(url: str, question: str) -> dict:
    """
    Extract relevant context window from YouTube transcript for QA.
    """

    try:
        video_id = extract_video_id(url)

        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)

        transcript_list = [
            {
                "text": t.text,
                "start": t.start,
                "duration": t.duration
            }
            for t in transcript
        ]

        # STEP 1: build full text
        full_text = " ".join([t["text"] for t in transcript_list])

   
        # STEP 2: find best match index   
        keywords = question.lower().split()

        match_indices = []
        for i, t in enumerate(transcript_list):
            text = t["text"].lower()
            if any(k in text for k in keywords):
                match_indices.append(i)


        # STEP 3: expand context window
        window_size = 5  # ±5 lines

        context_chunks = []

        if match_indices:
            for idx in match_indices[:3]:  # limit multiple matches
                start = max(0, idx - window_size)
                end = min(len(transcript_list), idx + window_size)

                chunk = " ".join(
                    t["text"] for t in transcript_list[start:end]
                )
                context_chunks.append(chunk)

        # fallback
        context = " ".join(context_chunks) if context_chunks else full_text[:3000]

        return {
            "type": "transcript_qa",
            "context": context
        }

    except Exception as e:
        return {"error": str(e)}