from tools.web_tool import web_tool
from tools.wiki_tool import wiki_tool
from tools.code_tool import code_tool
from tools.youtube_video_tool import youtube_frame_extractor
from tools.download_file_tool import download_file_from_url
from tools.save_file_tool import save_and_read_file
from tools.arxiv_tool import arxiv_search
from tools.youtube_transcript_tool import youtube_transcript_qa


ALL_TOOLS = [
    web_tool,
    wiki_tool,
    code_tool,
    youtube_frame_extractor,
    download_file_from_url,
    save_and_read_file,
    arxiv_search,
    youtube_transcript_qa
]