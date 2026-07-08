import sys
from pathlib import Path
import inspect

BACKEND_DIR = Path(__file__).parent
sys.path.append(str(BACKEND_DIR))

from livekit.agents import llm

print("llm module contents:")
print(dir(llm))

print("\nllm.LLM source:")
print(inspect.getsource(llm.LLM))

print("\nLLMStream source:")
print(inspect.getsource(llm.LLMStream))

print("\nChatContext source:")
print(inspect.getsource(llm.ChatContext))
