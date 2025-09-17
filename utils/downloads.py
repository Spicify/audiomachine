import base64
from pathlib import Path


def get_audio_download_link(audio_data, filename="dialogue.mp3"):
    """Generate a download link for audio data"""
    b64 = base64.b64encode(audio_data).decode()
    href = f'''
    <a href="data:audio/mp3;base64,{b64}" download="{filename}" 
       style="background: linear-gradient(45deg, #ff6b6b, #4ecdc4); 
              color: white; 
              padding: 12px 24px; 
              text-decoration: none; 
              border-radius: 8px; 
              display: inline-block; 
              margin: 10px 0;
              font-weight: bold;
              box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
       📥 Download Audio File
    </a>'''
    return href


def create_output_folders():
    """Create organized output folder structure"""
    base_output = Path("./audio_output")
    base_output.mkdir(exist_ok=True)

    # Create subfolders
    folders = {
        "teasers": base_output / "teasers",
        "chapters": base_output / "chapters",
        "voice_tests": base_output / "voice_tests",
        "books": base_output / "books"
    }

    for folder in folders.values():
        folder.mkdir(exist_ok=True)

    return folders
