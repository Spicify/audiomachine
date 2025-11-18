"""
Local parser package combining the exported ParserPipeline implementation
with app-specific helpers (dialogue/text parser utilities).
"""

from .parser_core.pipeline import ParserPipeline

__all__ = ["ParserPipeline"]
