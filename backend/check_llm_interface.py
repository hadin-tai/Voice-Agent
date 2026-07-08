import sys
from pathlib import Path
import inspect

BACKEND_DIR = Path(__file__).parent
sys.path.append(str(BACKEND_DIR))

from livekit.plugins import google

print("google.LLM source:")
print(inspect.getsource(google.LLM))
