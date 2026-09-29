# Run master_pipeline.py in Google Colab from VS Code

## Workflow
VS Code = editor. Colab = GPU compute.

1. Push repo to GitHub from VS Code.
2. Open Colab notebook.
3. Mount Drive and clone repo.
4. Set `COLAB_EPOCHS` to reduce training time, then run `src/master_pipeline.py`.

## Colab cells to paste

### 1. Setup
```python
!pip install torch torchvision tensorflow scikit-learn onnxruntime

from google.colab import drive
drive.mount('/content/drive')
```

### 2. Get code
```python
!git clone https://github.com/Marvyoha/pytorch-tflite-quantisation.git
%cd /content/repo
```

### 3. Optional: copy from Drive if you prefer
```python
# !cp -r /content/drive/MyDrive/pytorch-tflite-quantisation /content/repo
```

### 4. Set quick epochs for Colab
```python
import os

os.environ["COLAB_EPOCHS"] = "5"  # change to 10, 20 etc. 50 = full spec
```

### 5. Run master pipeline
```python
!python src/master_pipeline.py
```

### 6. Save artifacts to Drive
```python
from pathlib import Path
import shutil

out = "/content/drive/MyDrive/pytorch-tflite-quantisation_run"
Path(out).mkdir(parents=True, exist_ok=True)
shutil.copytree("models", f"{out}/models", dirs_exist_ok=True)
shutil.copytree("graphs", f"{out}/graphs", dirs_exist_ok=True)
print("Saved to", out)
```

## Notes
- Colab runtime: Runtime → Change runtime type → GPU
- `COLAB_EPOCHS` env var overrides `src/master_config.py`. Default 50.
- VS Code changes are synced via git push/pull before each Colab run.
- Models and graphs persist in Drive after step 6.

## VS Code tips
- Use GitLens to push frequently.
- Keep `master_config.py` changes minimal; use env var for Colab-only tweaks.
