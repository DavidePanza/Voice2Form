// Client-side resampling function
async function resampleAudio(audioBuffer, targetSampleRate = 16000) {
    const originalSampleRate = audioBuffer.sampleRate;
    
    if (originalSampleRate === targetSampleRate) {
        console.log(`✅ Already ${targetSampleRate}Hz`);
        return audioBuffer;
    }
    
    console.log(`🔄 Resampling ${originalSampleRate}Hz → ${targetSampleRate}Hz`);
    
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: targetSampleRate
    });
    
    // Create offline context for resampling
    const offlineContext = new OfflineAudioContext(
        1, // mono
        audioBuffer.duration * targetSampleRate,
        targetSampleRate
    );
    
    const source = offlineContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineContext.destination);
    source.start();
    
    const resampledBuffer = await offlineContext.startRendering();
    console.log(`✅ Resampled to ${resampledBuffer.length} samples`);
    
    return resampledBuffer;
}

// Convert to WAV for upload
function audioBufferToWav(audioBuffer) {
    const length = audioBuffer.length;
    const arrayBuffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(arrayBuffer);
    
    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, audioBuffer.sampleRate, true);
    view.setUint32(28, audioBuffer.sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, length * 2, true);
    
    // Convert float samples to 16-bit PCM
    const channelData = audioBuffer.getChannelData(0);
    let offset = 44;
    for (let i = 0; i < length; i++) {
        const sample = Math.max(-1, Math.min(1, channelData[i]));
        view.setInt16(offset, sample * 0x7FFF, true);
        offset += 2;
    }
    
    return arrayBuffer;
}

// Complete upload function
async function uploadAudioForAnalysis(file, windowSize = 1.0, hopSize = 0.25) {
    try {
        console.log(`📁 Processing file: ${file.name} (${file.size} bytes)`);
        
        // Step 1: Read file into AudioBuffer
        const arrayBuffer = await file.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        console.log(`🎵 Original: ${audioBuffer.duration.toFixed(2)}s at ${audioBuffer.sampleRate}Hz`);
        
        // Step 2: Client-side resample to 16kHz
        const resampledBuffer = await resampleAudio(audioBuffer, 16000);
        
        // Step 3: Convert to WAV for upload
        const wavArrayBuffer = audioBufferToWav(resampledBuffer);
        const wavBlob = new Blob([wavArrayBuffer], { type: 'audio/wav' });
        
        console.log(`📤 Uploading ${wavBlob.size} bytes (vs original ${file.size})`);
        
        // Step 4: Upload to your API
        const formData = new FormData();
        formData.append('audio_file', wavBlob, 'audio_16khz.wav');
        formData.append('window_size', windowSize.toString());
        formData.append('hop_size', hopSize.toString());
        
        const response = await fetch('/analyze_voice_endpoint', {
            method: 'POST',
            body: formData
        });
        
        return await response.json();
        
    } catch (error) {
        console.error('❌ Client-side processing failed:', error);
        throw error;
    }
}