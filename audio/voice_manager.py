from typing import Dict, Optional
from config_loader import CHARACTER_VOICES
from audio.utils import get_flat_character_voices
import re
import datetime
import json
from pathlib import Path
import streamlit as st
from typing import List


def save_voice_mappings(mappings: Dict[str, str], project_name: str):
    """Save voice mappings to JSON file"""
    mappings_dir = Path("./voice_mappings")
    mappings_dir.mkdir(exist_ok=True)

    # Clean project name for filename
    clean_name = re.sub(r'[^\w\s-]', '', project_name).strip()
    clean_name = re.sub(r'[-\s]+', '_', clean_name)

    mappings_file = mappings_dir / f"{clean_name}_voices.json"

    with open(mappings_file, 'w') as f:
        json.dump({
            "project_name": project_name,
            "created_date": datetime.now().isoformat(),
            "voice_mappings": mappings
        }, f, indent=2)

    return mappings_file


def load_voice_mappings(project_name: str) -> Optional[Dict[str, str]]:
    """Load voice mappings from JSON file"""
    mappings_dir = Path("./voice_mappings")
    if not mappings_dir.exists():
        return None

    clean_name = re.sub(r'[^\w\s-]', '', project_name).strip()
    clean_name = re.sub(r'[-\s]+', '_', clean_name)

    mappings_file = mappings_dir / f"{clean_name}_voices.json"

    if mappings_file.exists():
        try:
            with open(mappings_file, 'r') as f:
                data = json.load(f)
                return data.get("voice_mappings", {})
        except Exception as e:
            st.warning(f"Error loading voice mappings: {e}")

    return None


def get_saved_voice_projects() -> List[str]:
    """Get list of saved voice mapping projects"""
    mappings_dir = Path("./voice_mappings")
    if not mappings_dir.exists():
        return []

    projects = []
    for file in mappings_dir.glob("*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                projects.append(data.get("project_name", file.stem))
        except:
            continue

    return sorted(projects)


class VoiceManager:
    """Manage character voice assignments"""

    def __init__(self):
        # Start with predefined voices
        self.voice_assignments = get_flat_character_voices().copy()

    def get_available_voices(self) -> Dict[str, Dict[str, str]]:
        """Get all available voices with descriptions"""
        return CHARACTER_VOICES

    def assign_voice(self, character: str, voice_id: str):
        """Assign a voice to a character"""
        if character in self.voice_assignments:
            self.voice_assignments[character]["voice_id"] = voice_id

    def get_voice_for_character(self, character: str) -> Optional[str]:
        """Get voice ID for character"""
        data = self.voice_assignments.get(character)
        if not data:
            return None
        voice_id = data["voice_id"] if isinstance(data, dict) else data
        return voice_id

    def get_character_description(self, character: str) -> str:
        """Get description of character's current voice"""
        data = self.voice_assignments.get(character)
        if not data:
            return "No voice assigned"
        voice_id = data["voice_id"] if isinstance(data, dict) else data
        if not voice_id:
            return "No voice assigned"

        for category in CHARACTER_VOICES.values():
            for char, char_data in category.items():
                if char_data["voice_id"] == voice_id:
                    return char_data.get("description", "No description")
        return "Custom voice assignment"
