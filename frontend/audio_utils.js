// Simple logging helper
function addLog(message) {
    const log = document.getElementById('log');
    if (log) {
        log.textContent += new Date().toLocaleTimeString() + ' - ' + message + '\n';
        log.scrollTop = log.scrollHeight;
    } else {
        console.log(message);
    }
}

// Upload audio file directly to endpoint
async function uploadAudio(file, endpoint, windowSize = 1.0, hopSize = 0.25) {
    try {
        addLog(`Uploading: ${file.name} (${(file.size/1024/1024).toFixed(2)} MB)`);
        
        const formData = new FormData();
        formData.append('audio_file', file);
        formData.append('window_size', windowSize.toString());
        formData.append('hop_size', hopSize.toString());
        formData.append('use_single_VAD', document.getElementById('useSingleVAD').checked);

        addLog('Sending request...');
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const results = await response.json();
        addLog('Analysis complete!');
        
        return results;
        
    } catch (error) {
        addLog(`Error: ${error.message}`);
        throw error;
    }
}

// Analyze audio
async function analyzeAudio() {
    const endpoint = document.getElementById('endpoint').value;
    const windowSize = parseFloat(document.getElementById('windowSize').value);
    const hopSize = parseFloat(document.getElementById('hopSize').value);
    const analyzeBtn = document.getElementById('analyzeBtn');
    const fileInput = document.getElementById('audioFile');
    
    if (fileInput.files.length === 0) {
        alert('Please select an audio file');
        return;
    }
    
    analyzeBtn.disabled = true;
    document.getElementById('results').style.display = 'none';
    
    try {
        const results = await uploadAudio(fileInput.files[0], endpoint, windowSize, hopSize);
        
        // Display results
        document.getElementById('results').style.display = 'block';
        document.getElementById('resultsContent').innerHTML = `
            <p><strong>Status:</strong> ${results.status}</p>
            <p><strong>Frames:</strong> ${results.metadata?.total_frames || 'N/A'}</p>
            <p><strong>Duration:</strong> ${results.metadata?.audio_duration?.toFixed(2) || 'N/A'}s</p>
            <p><strong>Processing Time:</strong> ${results.timing?.total_time?.toFixed(2) || 'N/A'}s</p>
            <details>
                <summary>View Data (click to expand)</summary>
                <pre>${JSON.stringify(results, null, 2)}</pre>
            </details>
        `;
        
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        analyzeBtn.disabled = false;
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // File upload handler
    document.getElementById('audioFile').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            document.getElementById('fileInfo').innerHTML = 
                `Selected: ${file.name} (${(file.size/1024/1024).toFixed(2)} MB)`;
        }
    });
    
    // Analyze button
    document.getElementById('analyzeBtn').addEventListener('click', analyzeAudio);
});