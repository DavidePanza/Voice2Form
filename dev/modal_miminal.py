import os
import io
import json
import numpy as np
import librosa
import audeer
import audonnx
import audinterface
from modal import App, Image, web_endpoint, Volume
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Create the Modal app
app = App("voice-analysis-api")

# Create a persistent volume for model storage
model_volume = Volume.from_name("voice-analysis-models", create_if_missing=True)

# Define the container image with all dependencies
image = (
    Image.debian_slim(python_version="3.10")
    .pip_install([
        "audeer",
        "audonnx", 
        "audinterface",
        "librosa",
        "numpy",
        "pandas",
        "fastapi",
        "python-multipart",
        "soundfile"
    ])
    .apt_install(["ffmpeg"]) # ffmpeg for audio processing (nneeded by librosa)
    # Download model during image build
    .run_commands([
        "mkdir -p /model_cache",
        "python -c \"import audeer; audeer.download_url('https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip', '/model_cache/model.zip', verbose=True)\"",
        "python -c \"import audeer; audeer.extract_archive('/model_cache/model.zip', '/model_cache/model', verbose=True)\""
    ])
)

# Mount local files if needed (optional)
# mount = Mount.from_local_dir("./", remote_path="/app")

@app.function(
    image=image,
    volumes={"/persistent_model": model_volume},  # Mount persistent volume
    timeout=300,  # 5 minutes timeout
    memory=2048,  # 2GB memory
    cpu=2.0,
    gpu="L4", 
    # keep_warm=1 
    # Removed keep_warm for cost optimization
    # Cold start will be ~5-10 seconds but no continuous billing
    concurrency_limit=10,  # Allow multiple concurrent requests
    allow_concurrent_inputs=100,  # Queue up to 100 requests
)
@web_endpoint(method="POST", docs=True)
def analyze_voice_endpoint(
    audio_file: UploadFile = File(...),
    window_size: float = Form(default=1.0),
    hop_size: float = Form(default=0.25)
):
    """
    Analyze voice for VAD (Valence, Arousal, Dominance) + envelope extraction
    
    Parameters:
    - audio_file: Audio file (WAV, MP3, etc.)
    - window_size: Analysis window size in seconds (default: 1.0)
    - hop_size: Hop size between windows in seconds (default: 0.25)
    
    Returns:
    - JSON with frame-by-frame analysis results
    """
    try:
        # Load and process the uploaded audio file
        audio_bytes = audio_file.file.read()
        
        # Load audio using librosa
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        
        # Load the official model (with caching)
        model, interface = load_official_model()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")


def load_official_model():
    """Load the official ONNX model with multi-level caching"""
    # Check if model is already cached in memory
    if hasattr(load_official_model, '_cached_model'):
        print("Using in-memory cached model")
        return load_official_model._cached_model
    
    print("Loading official model...")
    
    # Try to use model from persistent volume first
    persistent_model_path = "/persistent_model/w2v2_model"
    if os.path.exists(persistent_model_path):
        print("Using model from persistent volume")
        model_root = persistent_model_path
    else:
        # Fall back to model baked into image, then copy to persistent volume
        print("Using model from image, copying to persistent volume...")
        image_model_path = "/model_cache/model"
        if os.path.exists(image_model_path):
            os.makedirs("/persistent_model", exist_ok=True)
            import shutil
            shutil.copytree(image_model_path, persistent_model_path)
            model_root = persistent_model_path
            # Commit the volume to persist the model
            model_volume.commit()
        else:
            # Critical error: model should have been baked into image during deployment
            error_msg = (
                "DEPLOYMENT ERROR: Model not found in container image!\n"
                "This means the image build failed. Check deployment logs.\n"
                "Expected model at: /model_cache/model\n"
                "Re-run: modal deploy modal_voice_analysis.py"
            )
            print(error_msg)
            raise FileNotFoundError(error_msg)
    
    model = audonnx.load(model_root)
    
    # Create interface for easier usage
    interface = audinterface.Feature(
        model.labels('logits'),
        process_func=model,
        process_func_args={'outputs': 'logits'},
        sampling_rate=16000,
        resample=True,
        verbose=False,
    )
    
    # Cache the loaded model in memory
    load_official_model._cached_model = (model, interface)
    print("Model loaded and cached successfully")
    
    return model, interface


# Add the local entrypoint here, at the end of the script
@app.local_entrypoint()
def test_local():
    """Test the voice analysis function locally"""
    print("Testing voice analysis...")
    print("Note: This is a web endpoint, so test with HTTP requests")
    print("\nDeployment commands:")
    print("- Deploy: modal deploy your_script.py")
    print("- Test locally: modal serve your_script.py")
    print("\nAfter deployment, test with:")
    print("curl -X POST [your-modal-url] -F 'audio_file=@sample.wav'")

