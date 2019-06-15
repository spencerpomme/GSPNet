import sys
from pathlib import Path

parent = str(Path().resolve().parent)
src = parent + '\src'
print(src)
sys.path.append(src)
