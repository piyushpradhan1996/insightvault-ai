from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(name: str) -> str:
    prompt_path = PROMPT_DIR / name
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {name}")
    return prompt_path.read_text(encoding="utf-8")

