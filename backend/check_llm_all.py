import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
sys.path.append(str(BACKEND_DIR))

from livekit.agents import llm

print("llm.__all__:", llm.__all__)
