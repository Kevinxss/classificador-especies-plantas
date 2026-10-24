from pathlib import Path
from sys import os 
RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PLANTVILLAGE_DIR", RAIZ / "data" / "plantvillage"))
TAMANO_IMG = (128, 128)