document.addEventListener('DOMContentLoaded', () => {
    let currentSessionId = null;
    let currentView = 'original';

    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const mainImage = document.getElementById('mainImage');
    const stressIntensity = document.getElementById('stressIntensity');
    const intensityVal = document.getElementById('intensityVal');
    const noiseBoost = document.getElementById('noiseBoost');
    const boostVal = document.getElementById('boostVal');
    const loader = document.getElementById('loader');
    const statusBar = document.querySelector('.status-bar');

    // UI elements to update
    const ui = {
        effDepth: document.getElementById('effDepth'),
        lsbPadding: document.getElementById('lsbPadding'),
        uniqueRatio: document.getElementById('uniqueRatio'),
        isUpscaled: document.getElementById('isUpscaled'),
        bandingScore: document.getElementById('bandingScore'),
        fftRatio: document.getElementById('fftRatio'),
        channelCorr: document.getElementById('channelCorr'),
        sessionPath: document.getElementById('sessionPath')
    };

    // --- Interaction ---

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleUpload(e.target.files[0]);
        }
    });

    stressIntensity.addEventListener('input', (e) => {
        intensityVal.textContent = e.target.value;
    });

    stressIntensity.addEventListener('change', async () => {
        if (currentSessionId) {
            await runStressAnalysis();
            if (currentView === 'stress') updateImageView();
        }
    });

    noiseBoost.addEventListener('input', (e) => {
        boostVal.textContent = e.target.value + 'x';
    });

    noiseBoost.addEventListener('change', () => {
        if (currentSessionId && ['noise', 'fft'].includes(currentView)) {
            updateImageView();
        }
    });

    document.querySelectorAll('.toolbar button').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelector('.toolbar button.active').classList.remove('active');
            btn.classList.add('active');
            currentView = btn.dataset.view;
            updateImageView();
        });
    });

    // --- API Calls ---

    async function handleUpload(file) {
        setLoading(true);
        updateStatus(`UPLOADING ${file.name.toUpperCase()}...`);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/upload', { method: 'POST', body: formData });

            // Handle non-JSON responses (like 413 or 500 HTML pages)
            const contentType = resp.headers.get("content-type");
            let data = {};
            if (contentType && contentType.includes("application/json")) {
                data = await resp.json();
            } else {
                const text = await resp.text();
                throw new Error(`Server returned non-JSON response (${resp.status}): ${text.slice(0, 100)}`);
            }

            if (resp.ok) {
                currentSessionId = data.session_id;
                ui.sessionPath.textContent = file.name;
                updateStatus(`IMAGE LOADED: ${data.resolution} [${data.dtype}]`);

                // Run all analyses
                await Promise.all([
                    runAuthenticityAnalysis(),
                    runNoiseAnalysis(),
                    runStressAnalysis()
                ]);

                updateImageView();
            } else {
                alert(`Upload failed: ${data.detail || 'Unknown error'}`);
            }
        } catch (err) {
            console.error(err);
            alert(`Error during upload: ${err.message}`);
        } finally {
            setLoading(false);
        }
    }

    async function runAuthenticityAnalysis() {
        const resp = await fetch(`/analyze/authenticity/${currentSessionId}`);
        const data = await resp.json();

        ui.effDepth.textContent = `${data.bit_depth.effective_depth}-bit`;
        ui.lsbPadding.textContent = data.bit_depth.is_padded ? `Padded (LSB ${data.bit_depth.lowest_active_bit})` : "None";
        ui.lsbPadding.className = data.bit_depth.is_padded ? 'value warn' : 'value pass';

        ui.uniqueRatio.textContent = (data.histogram.unique_values_ratio * 100).toFixed(2) + '%';

        ui.isUpscaled.textContent = data.histogram.likely_upscaled_8bit ? "YES (Combing)" : "NO";
        ui.isUpscaled.className = data.histogram.likely_upscaled_8bit ? 'value fail' : 'value pass';
    }

    async function runNoiseAnalysis() {
        const resp = await fetch(`/analyze/noise/${currentSessionId}`);
        const data = await resp.json();

        ui.fftRatio.textContent = data.fft_spike_ratio;
        ui.fftRatio.className = data.has_periodic_patterns ? 'value fail' : 'value pass';

        if (data.avg_channel_correlation) {
            ui.channelCorr.textContent = data.avg_channel_correlation.toFixed(4);
            ui.channelCorr.className = data.interpretation === "Natural" ? 'value pass' : 'value warn';
        } else {
            ui.channelCorr.textContent = "N/A";
        }
    }

    async function runStressAnalysis() {
        const intensity = stressIntensity.value;
        const resp = await fetch(`/analyze/stress/${currentSessionId}?intensity=${intensity}`);
        const data = await resp.json();

        ui.bandingScore.textContent = `${data.banding_metric.toFixed(2)}%`;
        document.getElementById('analyzed_area').textContent = (data.analyzed_area || 0).toFixed(1) + '%';
        document.getElementById('global_impact').textContent = (data.global_impact || 0).toFixed(2) + '%';
        ui.bandingScore.className = data.passed ? 'value pass' : 'value fail';
    }

    function updateImageView() {
        if (!currentSessionId) return;

        const intensity = stressIntensity.value;
        const boost = noiseBoost.value;
        let url = '';

        switch (currentView) {
            case 'original': url = `/visualize/original/${currentSessionId}`; break;
            case 'stress': url = `/visualize/stress/${currentSessionId}?intensity=${intensity}`; break;
            case 'banding': url = `/visualize/banding/${currentSessionId}?intensity=${intensity}`; break;
            case 'noise': url = `/visualize/noise_residual/${currentSessionId}?boost=${boost}`; break;
            case 'fft': url = `/visualize/fft_spectrum/${currentSessionId}?boost=${boost}`; break;
        }

        // Add cache buster for dynamic params, correctly handling existing query strings
        const needsCacheBuster = ['stress', 'banding', 'noise', 'fft'].includes(currentView);
        if (needsCacheBuster) {
            const separator = url.includes('?') ? '&' : '?';
            mainImage.src = `${url}${separator}t=${Date.now()}`;
        } else {
            mainImage.src = url;
        }
    }

    function setLoading(isLoading) {
        if (isLoading) loader.classList.remove('hidden');
        else loader.classList.add('hidden');
    }

    function updateStatus(msg) {
        statusBar.textContent = `SYSTEM // ${msg}`;
    }
});
