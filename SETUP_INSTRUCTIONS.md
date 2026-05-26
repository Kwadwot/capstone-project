# Setup Instructions

## Environment Setup

### Local Development

#### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

#### 2. Set up Comet ML API key (optional experiment tracking)

**Step 1:** Create a free account at [comet.ml](https://www.comet.ml/)

**Step 2:** Get your API key:
- Log in to Comet ML
- Go to Settings → API Keys
- Copy your API key

**Step 3:** Create a `.env` file in the project root:
```bash
# .env (DO NOT COMMIT THIS FILE)
COMET_API_KEY=your_api_key_here
```

**Note:** `.env` is already in `.gitignore` and will never be committed to Git.

#### 3. Install python-dotenv
```bash
pip install python-dotenv
```

The notebook will automatically load this file.

---

### Google Colab

#### 1. Open notebook in Colab
Click the "Open In Colab" badge in the README, or go to:
https://colab.research.google.com/github/YOUR_USERNAME/capstone_project/blob/main/Project_PT_Part2_Debiasing_Kaggle.ipynb

#### 2. Set up Comet ML (optional)
The notebook will prompt you for the API key. To avoid prompting:

1. Click the **Secrets** icon (🔑) in the left sidebar
2. Click **+ Add new secret**
3. Name: `COMET_API_KEY`
4. Value: Paste your Comet API key from [comet.ml](https://www.comet.ml/)
5. Click **Add secret**

The notebook will automatically read from Colab Secrets.

---

## Dataset Setup

The notebook handles dataset downloads automatically on first run.

### Manual download (optional)
```python
from src.data_download import download_utkface, download_tiny_imagenet

download_utkface()          # Downloads ~23,000 face images
download_tiny_imagenet()    # Downloads ~100,000 non-face images
```

**Storage requirement:** ~50 GB total disk space

---

## Kaggle API Setup (if needed for dataset access)

**Note:** The notebook can download without Kaggle, but if you hit rate limits:

1. Create account at [kaggle.com](https://www.kaggle.com)
2. Go to **Settings → API → Create New API Token**
3. Save `kaggle.json` to `~/.kaggle/kaggle.json`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json` (on macOS/Linux)

---

## Verifying Setup

### Test local installation
```python
# Test PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__} on {torch.cuda.get_device_name()}')"

# Test src imports
python -c "from src.dataset import prepare_datasets; print('OK')"
```

### Test Colab
1. Run the first cell in the notebook
2. Should complete without errors
3. Datasets will auto-download (first run takes ~5-10 minutes)

---

## Requirements

See `requirements.txt` for full list. Key packages:
- `torch` ≥ 1.9.0 (with CUDA support optional)
- `torchvision` ≥ 0.10.0
- `numpy`, `pandas`, `matplotlib`
- `scikit-learn`, `scipy`
- `comet_ml` (optional, for experiment tracking)
- `kagglehub` (for dataset downloads)

---

## Troubleshooting

### "Comet API key not provided"
- If you want to skip Comet: just press Enter when prompted
- If you want to use it: ensure your `.env` file has the correct key or Colab secret is set

### "CUDA not available"
- The notebook will fall back to CPU automatically
- Note: Training will be much slower on CPU
- To install CUDA support: follow [PyTorch installation guide](https://pytorch.org/get-started/locally/)

### "Dataset download fails"
- Check internet connection
- If Kaggle API rate limit: set up Kaggle credentials as described above
- Datasets are cached; second runs will use local cache

### "Out of memory"
- Reduce batch size in notebook (look for `batch_size` parameters)
- Or reduce `num_epochs` for testing

---

## API Security

⚠️ **IMPORTANT:** Never commit API keys to Git!

This project securely handles API keys:
- ✓ Notebook prompts or reads from environment
- ✓ `.env` file is in `.gitignore`
- ✓ Colab uses native Secrets system
- ✓ Original exposed key has been removed

If you have any old keys exposed publicly, regenerate them immediately.

---

## Next Steps

1. Follow one of the setup paths above (Local or Colab)
2. Run the notebook cell-by-cell
3. First run will download datasets (~20 GB, 10-15 min)
4. Training will begin automatically
5. Results will be saved to `results/` and visualized

For detailed methodology, see `report/main.pdf`.
