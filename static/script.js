async function analyzeNews() {
    const textarea = document.getElementById('newsText');
    const text = textarea.value.trim();
    
    // Validation
    if (!text) {
        showError('Please enter a news statement to analyze.');
        return;
    }
    
    if (text.length < 10) {
        showError('Please enter at least 10 characters.');
        return;
    }
    
    // Hide previous results/errors
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';
    
    // Show loading state
    const btn = document.getElementById('analyzeBtn');
    const btnText = document.getElementById('btnText');
    const btnLoader = document.getElementById('btnLoader');
    
    btn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-block';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to analyze text');
        }
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        showError(error.message || 'An error occurred while analyzing the text.');
    } finally {
        // Reset button
        btn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

function displayResults(data) {
    console.log('API Response:', data);  // Debug log
    
    const resultSection = document.getElementById('resultSection');
    const verdictBox = document.getElementById('verdictBox');
    const verdictLabel = document.getElementById('verdictLabel');
    const confidenceText = document.getElementById('confidenceText');
    const fakeProb = document.getElementById('fakeProb');
    const realProb = document.getElementById('realProb');
    const fakeBar = document.getElementById('fakeBar');
    const realBar = document.getElementById('realBar');
    
    // Determine if real or fake based on class prediction
    const isReal = data.class === 1;
    
    // Update verdict
    verdictBox.className = 'verdict ' + (isReal ? 'real' : 'fake');
    verdictLabel.textContent = isReal ? 'REAL' : 'FAKE';
    confidenceText.textContent = `${data.confidence}% Confidence`;
    
    // Update probability bars
    const fakePercent = data.probabilities?.fake || 0;
    const realPercent = data.probabilities?.real || 0;
    
    fakeProb.textContent = `${fakePercent}%`;
    realProb.textContent = `${realPercent}%`;
    
    // Animate bars
    setTimeout(() => {
        fakeBar.style.width = `${fakePercent}%`;
        realBar.style.width = `${realPercent}%`;
    }, 100);
    
    // Show result section
    resultSection.style.display = 'block';
    
    // Scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');
    
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    
    // Scroll to error
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Allow Enter key to submit (with Ctrl/Cmd)
document.getElementById('newsText').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        analyzeNews();
    }
});
