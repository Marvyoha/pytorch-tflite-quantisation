import os
import sys

# Colab quick defaults
os.environ.setdefault("COLAB_EPOCHS", "50")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Ensure repo root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.master_pipeline import main

if __name__ == "__main__":
    print(f"Running with EPOCHS={os.getenv('COLAB_EPOCHS')}")
    main()
