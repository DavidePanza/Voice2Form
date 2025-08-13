import os
import io
import json
import time
import numpy as np
import librosa
import audeer
import audonnx
import audinterface
from modal import App, Image, fastapi_endpoint, Volume, concurrent
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
    .apt_install(["ffmpeg"])
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
    volumes={"/persistent_model": model_volume},
    timeout=300,
    memory=2048,
    cpu=2.0,
    gpu="L4", 
    keep_warm=0,  # Keep warm for production use
    max_containers=10,  
)
@concurrent(max_inputs=100) 
@fastapi_endpoint(method="POST", docs=True) 
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
    # Start total timing
    start_total = time.time()
    timing_info = {}
    
    try:
        print(f"Starting voice analysis at {time.strftime('%H:%M:%S')}")
        
        # Step 1: Read audio file
        step_start = time.time()
        audio_bytes = audio_file.file.read()
        timing_info['file_read'] = time.time() - step_start
        print(f"File read: {timing_info['file_read']:.3f}s")
        
        # Step 2: Load and process audio with librosa
        step_start = time.time()
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        timing_info['audio_processing'] = time.time() - step_start
        print(f"🎵 Audio processing: {timing_info['audio_processing']:.3f}s")
        
        # Step 3: Load model (with caching)
        step_start = time.time()
        model, interface = load_official_model()
        timing_info['model_loading'] = time.time() - step_start
        print(f"Model loading: {timing_info['model_loading']:.3f}s")
        
        # Step 4: VAD analysis
        step_start = time.time()
        results = predict_vad_frames_with_envelope_official(
            audio, interface, window_size, hop_size
        )
        timing_info['vad_analysis'] = time.time() - step_start
        print(f"🔬 VAD analysis: {timing_info['vad_analysis']:.3f}s")
        
        # Step 5: Format results
        step_start = time.time()
        response_data = {
            "status": "success",
            "metadata": {
                "total_frames": len(results['times']),
                "window_size": results['window_size'],
                "hop_size": results['hop_size'],
                "audio_duration": len(audio) / sr,
                "sample_rate": sr
            },
            "timing": timing_info,  # Add timing info to response
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
        
        timing_info['response_formatting'] = time.time() - step_start
        timing_info['total_time'] = time.time() - start_total
        
        print(f"Response formatting: {timing_info['response_formatting']:.3f}s")
        print(f"Total processing time: {timing_info['total_time']:.3f}s")
        print(f"Analysis completed at {time.strftime('%H:%M:%S')}")
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        total_time = time.time() - start_total
        print(f"Error after {total_time:.3f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


def load_official_model():
    """Load the official ONNX model with multi-level caching"""
    load_start = time.time()
    
    # Check if model is already cached in memory
    if hasattr(load_official_model, '_cached_model'):
        print(f"Using in-memory cached model ({time.time() - load_start:.3f}s)")
        return load_official_model._cached_model
    
    print("Loading official model...")
    
    # Try to use model from persistent volume first
    persistent_model_path = "/persistent_model/w2v2_model"
    if os.path.exists(persistent_model_path):
        print(f"Using model from persistent volume")
        model_root = persistent_model_path
    else:
        # Fall back to model baked into image, then copy to persistent volume
        print("Using model from image, copying to persistent volume...")
        copy_start = time.time()
        image_model_path = "/model_cache/model"
        if os.path.exists(image_model_path):
            os.makedirs("/persistent_model", exist_ok=True)
            import shutil
            shutil.copytree(image_model_path, persistent_model_path)
            model_root = persistent_model_path
            # Commit the volume to persist the model
            model_volume.commit()
            print(f"📁 Model copied to persistent volume ({time.time() - copy_start:.3f}s)")
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
    
    # Load the actual model
    model_load_start = time.time()
    model = audonnx.load(model_root)
    print(f"Model loaded from disk ({time.time() - model_load_start:.3f}s)")
    
    # Create interface
    interface_start = time.time()
    interface = audinterface.Feature(
        model.labels('logits'),
        process_func=model,
        process_func_args={'outputs': 'logits'},
        sampling_rate=16000,
        resample=True,
        verbose=False,
    )
    print(f"🔧 Interface created ({time.time() - interface_start:.3f}s)")
    
    # Cache the loaded model in memory
    load_official_model._cached_model = (model, interface)
    total_load_time = time.time() - load_start
    print(f"Model loaded and cached successfully (total: {total_load_time:.3f}s)")
    
    return model, interface


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
    analysis_start = time.time()
    
    sr = 16000
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    
    vad_frames = []
    envelope_segments = []
    times = []
    
    # Extract full envelope
    envelope_start = time.time()
    full_envelope_times, full_envelope = extract_audio_envelope(audio, sr)
    envelope_time = time.time() - envelope_start
    print(f"Full envelope extraction: {envelope_time:.3f}s")
    
    # Sliding window for VAD
    frames_start = time.time()
    frame_count = 0
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
        frame_count += 1
    
    frames_time = time.time() - frames_start
    print(f"Frame-by-frame analysis: {frames_time:.3f}s ({frame_count} frames, {frames_time/frame_count:.3f}s per frame)")
    
    # Convert to arrays
    array_start = time.time()
    arousal = [frame['arousal'] for frame in vad_frames]
    dominance = [frame['dominance'] for frame in vad_frames] 
    valence = [frame['valence'] for frame in vad_frames]
    array_time = time.time() - array_start
    print(f"Array conversion: {array_time:.3f}s")
    
    total_analysis_time = time.time() - analysis_start
    print(f"Total VAD analysis: {total_analysis_time:.3f}s")
    
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
@fastapi_endpoint(method="GET")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Voice Analysis API is running"}

