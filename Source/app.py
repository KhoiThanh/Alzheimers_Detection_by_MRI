"""Streamlit entry point for the verified hold-out ensemble application."""
from pathlib import Path
_page = Path(__file__).with_name("app_holdout.py")
exec(compile(_page.read_text(encoding="utf-8"), str(_page), "exec"), globals())
