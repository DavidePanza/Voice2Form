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
        
        # Analyze the audio
        results = predict_vad_frames_with_envelope_official(
            audio, interface, window_size, hop_size
        )
        
        # Format results for JSON response
        response_data = {
            "status": "success",
            "metadata": {
                "total_frames": len(results['times']),
                "window_size": results['window_size'],
                "hop_size": results['hop_size'],
                "audio_duration": len(audio) / sr,
                "sample_rate": sr
            },
            "frames": []
        }
        
        # Add frame-by-frame results
        for i in range(len(results['times'])):
            frame_data = {
                "frame_index": i,
                "time": float(results['times'][i]),
                "valence": float(results['valence'][i]),
                "arousal": float(results['arousal'][i]),
                "dominance": float(results['dominance'][i]),
                "envelope": {
                    "mean_amplitude": float(results['envelope_segments'][i]['mean_amplitude']),
                    "max_amplitude": float(results['envelope_segments'][i]['max_amplitude']),
                    "amplitude_std": float(results['envelope_segments'][i]['amplitude_std'])
                }
            }
            response_data["frames"].append(frame_data)
        
        # Add full envelope data
        response_data["full_envelope"] = {
            "times": results['full_envelope_times'].tolist(),
            "values": results['full_envelope'].tolist()
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


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


# def download_official_model_fallback():
#     """Fallback model download function"""
#     model_root = '/persistent_model/w2v2_model'
#     cache_root = '/tmp/cache'
    
#     os.makedirs(cache_root, exist_ok=True)
#     os.makedirs('/persistent_model', exist_ok=True)
    
#     def cache_path(file):
#         return os.path.join(cache_root, file)
    
#     url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
#     dst_path = cache_path('model.zip')
    
#     if not os.path.exists(dst_path):
#         print("Downloading official model...")
#         audeer.download_url(url, dst_path, verbose=True)
    
#     if not os.path.exists(model_root):
#         print("Extracting model...")
#         audeer.extract_archive(dst_path, model_root, verbose=True)
#         # Commit to persistent volume
#         model_volume.commit()
    
#     return model_root


def predict_vad_official(audio, interface):
    """Predict VAD using official model"""
    # Process audio
    result = interface.process_signal(audio, 16000)
    
    # Extract values (result is a pandas DataFrame)
    arousal = float(result['arousal'].iloc[0])
    dominance = float(result['dominance'].iloc[0])
    valence = float(result['valence'].iloc[0])
    
    return {
        'arousal': arousal,
        'dominance': dominance,
        'valence': valence
    }


def extract_audio_envelope(audio, sr=16000, hop_length=512):
    """Extract amplitude envelope from audio"""
    rms_envelope = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    envelope_times = librosa.frames_to_time(np.arange(len(rms_envelope)),
                                           sr=sr, hop_length=hop_length)
    return envelope_times, rms_envelope


def predict_vad_frames_with_envelope_official(audio, interface, window_size=1.0, hop_size=0.25):
    """Predict VAD frame-by-frame using official model + extract envelope"""
    sr = 16000
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    
    vad_frames = []
    envelope_segments = []
    times = []
    
    # Extract full envelope
    full_envelope_times, full_envelope = extract_audio_envelope(audio, sr)
    
    # Sliding window for VAD
    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        end = start + window_samples
        window_audio = audio[start:end]
        
        # Predict VAD for this window
        vad = predict_vad_official(window_audio, interface)
        vad_frames.append(vad)
        
        # Extract envelope for this segment
        seg_times, seg_envelope = extract_audio_envelope(window_audio, sr)
        envelope_segments.append({
            'times': seg_times + (start / sr),
            'envelope': seg_envelope,
            'mean_amplitude': np.mean(seg_envelope),
            'max_amplitude': np.max(seg_envelope),
            'amplitude_std': np.std(seg_envelope)
        })
        
        times.append(start / sr)
    
    # Convert to arrays
    arousal = [frame['arousal'] for frame in vad_frames]
    dominance = [frame['dominance'] for frame in vad_frames] 
    valence = [frame['valence'] for frame in vad_frames]
    
    return {
        'times': np.array(times),
        'arousal': np.array(arousal),
        'dominance': np.array(dominance),
        'valence': np.array(valence),
        'envelope_segments': envelope_segments,
        'full_envelope_times': full_envelope_times,
        'full_envelope': full_envelope,
        'window_size': window_size,
        'hop_size': hop_size
    }


# Add CORS middleware for Netlify app integration
@app.function(image=image)
@web_endpoint(method="GET")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Voice Analysis API is running"}


if __name__ == "__main__":
    # For local testing
    import modal
    print("Deploy with: modal deploy modal_voice_analysis.py")
    print("\nCost Optimization Notes:")
    print("- No keep_warm: Only pay when processing requests")
    print("- Cold start: ~5-10 seconds (model loads from persistent volume)")
    print("- Warm requests: ~100ms (model cached in memory)")
    print("- Instance auto-shuts down after idle period")
    print("\nOptional: Add keep_warm=1 for production if you need <1s response times")