# Deployment Guide: LSTM Stock Predictor

Your Streamlit application is now ready to run locally and be deployed.

## 1. Local Execution
To run the application locally in the correct environment, use:
```bash
conda activate stock-predictor
streamlit run app.py
```

## 2. Fast Deployment Options

### Option A: Streamlit Community Cloud (Recommended)
This is the fastest and easiest way to deploy Streamlit apps.

1.  **Prepare your repository:**
    *   Push your code to a **GitHub** repository.
    *   Ensure `app.py` and `requirements.txt` are in the root directory.
2.  **Deploy:**
    *   Go to [share.streamlit.io](https://share.streamlit.io/).
    *   Connect your GitHub account.
    *   Select your repository, branch, and `app.py` as the main file.
    *   Click **Deploy**.

> [!IMPORTANT]
> **TensorFlow Versioning:**
> Your current `requirements.txt` specifies `tensorflow-macos` and `tensorflow-metal`. These are specific to Apple Silicon. For deployment on Streamlit Cloud (Linux), you should use a universal `requirements.txt`.

**Recommended `requirements.txt` for Deployment:**
```text
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
yfinance>=0.2.38
scikit-learn>=1.3
tensorflow>=2.13
streamlit>=1.30
```

### Option B: Hugging Face Spaces
1.  Create a new [Space](https://huggingface.co/spaces) on Hugging Face.
2.  Select **Streamlit** as the SDK.
3.  Upload your files (`app.py`, `requirements.txt`).
4.  It will automatically build and deploy.

## 3. Fixed Import Issues
The import errors were caused by:
1.  Running the application in the base environment instead of the `stock-predictor` environment.
2.  Missing packages (`yfinance`) in the base environment.

I have verified that all imports work correctly within the `stock-predictor` environment.
