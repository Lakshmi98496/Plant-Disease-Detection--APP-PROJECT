// Variable declarations for elements (cache them once)
const imageInput = document.getElementById('imageInput');
const predictButton = document.getElementById('predictButton');
const loadingMessage = document.getElementById('loadingMessage');
const previewContainer = document.getElementById('imagePreviewContainer');
const previewImage = document.getElementById('imagePreview');

const resultCard = document.getElementById('resultCard');
// New/Updated element references for Confidence and Severity
const predictedClassPlaceholder = document.getElementById('predictedClassPlaceholder');
const confidencePlaceholder = document.getElementById('confidencePlaceholder');
const severityPlaceholder = document.getElementById('severityPlaceholder');

const resultFilenameSpan = document.getElementById('resultFilename');
const diagnosisDescription = document.getElementById('diagnosisDescription');
const treatmentRecommendation = document.getElementById('treatmentRecommendation');
const tableBody = document.getElementById('resultsTable').querySelector('tbody');

// Define Max File Size (4MB)
const MAX_SIZE_MB = 4; 

// Function to reset the result card and diagnosis block content
function resetResultsDisplay() {
    predictedClassPlaceholder.textContent = 'Awaiting image...';
    resultFilenameSpan.textContent = '';
    
    // Clear NEW placeholders
    confidencePlaceholder.textContent = '';
    severityPlaceholder.textContent = '---';
    // Remove all severity/color classes
    severityPlaceholder.className = 'severity-label'; 
    
    // Remove old color classes and set default background
    resultCard.classList.remove('healthy', 'diseased'); 

    diagnosisDescription.textContent = 'Upload an image to get a full diagnosis here.';
    treatmentRecommendation.textContent = 'The best treatment options will appear here.';
}

// Helper function to apply severity styling
function applySeverityStyle(severityText) {
    const placeholder = severityPlaceholder;
    placeholder.textContent = severityText;
    placeholder.className = 'severity-label'; // Reset base class

    if (severityText.includes('HIGH')) {
        placeholder.classList.add('high');
    } else if (severityText.includes('MODERATE')) {
        placeholder.classList.add('moderate');
    } else {
        // LOW or HEALTHY plants
        placeholder.classList.add('low');
    }
}


// 1. Handle File Selection and Image Preview
imageInput.addEventListener('change', function(event) {
    const file = event.target.files[0];

    resetResultsDisplay(); // Reset results when a new file is chosen

    if (file) {
        // === CLIENT-SIDE VALIDATION ===
        if (file.size > MAX_SIZE_MB * 1024 * 1024) {
            // FIX: Using safe concatenation (+) - NO DOLLAR SYMBOL
            alert('File is too large (max ' + MAX_SIZE_MB + 'MB). Please select a smaller image.'); 
            this.value = ''; // Clear the input
            predictButton.disabled = true;
            return;
        }
        // ==============================
        
        predictButton.disabled = false;
        
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block'; // Show the preview area
        }
        reader.readAsDataURL(file);
    } else {
        predictButton.disabled = true;
        previewContainer.style.display = 'none'; // Hide the preview area
    }
});


// 2. Handle Prediction Button Click
predictButton.addEventListener('click', async () => {
    const file = imageInput.files[0];
    
    if (!file) {
        alert("Please select an image file first.");
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    // UI State: Start Loading
    predictButton.disabled = true;
    loadingMessage.style.display = 'block';
    resetResultsDisplay();
    predictedClassPlaceholder.textContent = 'Analyzing...';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        // --- IMPROVED ERROR HANDLING ---
        if (!response.ok || result.error) {
            const errorMsg = result.error || "Server returned an unknown error.";
            // FIX: Using safe concatenation (+) - NO DOLLAR SYMBOL
            alert('Error: ' + errorMsg); 
            predictedClassPlaceholder.textContent = 'Prediction Failed';
            resultCard.classList.remove('healthy', 'diseased'); 
            resultCard.classList.add('diseased');
            diagnosisDescription.textContent = errorMsg;
            // Clear NEW placeholders on error
            confidencePlaceholder.textContent = '';
            severityPlaceholder.textContent = '---';
            severityPlaceholder.className = 'severity-label';
            return; // Stop execution on error
        }
        // --- END IMPROVED ERROR HANDLING ---
        
        // UI State: Success - Update all elements
        const predictedClass = result.predicted_class;
        
        // Update Result Card
        resultFilenameSpan.textContent = result.filename;
        predictedClassPlaceholder.textContent = predictedClass;
        
        // NEW: Confidence and Severity Display (This is safe as it's not in an alert)
        confidencePlaceholder.textContent = '('+result.confidence+')';
        applySeverityStyle(result.severity); // Apply color based on severity text
        
        // Update Diagnosis Block with data from app.py
        diagnosisDescription.textContent = result.diagnosis;
        treatmentRecommendation.textContent = result.treatment;

        // Apply the color code class (Healthy/Diseased)
        resultCard.classList.remove('healthy', 'diseased'); 
        if (predictedClass.toLowerCase().includes('healthy')) {
            resultCard.classList.add('healthy');
        } else {
            resultCard.classList.add('diseased');
        }
        
        // 4. Update History Table (IMPROVED: Includes Confidence and Severity)
        const newRow = tableBody.insertRow(0); // Insert at the top
        const filenameCell = newRow.insertCell(0);
        const classCell = newRow.insertCell(1);
        
        filenameCell.textContent = result.filename;
        
        // FIX: Safely strip special characters/emojis from severity
        const cleanedSeverity = result.severity.replace(/[\u274C\u2757\u2705\u26A0\uFE0F\uD83D\uDEA9]/g, '').trim();

        // Display Class, Confidence, and CLEANED Severity in one cell
        // FINAL FIX: Using safe string concatenation (+) to build the HTML string
classCell.innerHTML = '<strong>' + predictedClass + '</strong>' + 
                      '<br/>' + 
                      '<small>' +
                          'Conf: ' + result.confidence + ' | ' +
                          'Sev: ' + cleanedSeverity +
                      '</small>'; // <-- Final semicolon is critical for syntax!
        
        // *** The try block CLOSES here ***
        
    } catch (error) {
        // UI State: Network/Fetch Error
        console.error('Prediction failed:', error);
        // FIX: Using safe concatenation (+) - NO DOLLAR SYMBOL
        alert('A network error occurred. Check the console and server logs.');
        predictedClassPlaceholder.textContent = 'Network Error.';
        resultCard.classList.remove('healthy', 'diseased'); 
        resultCard.classList.add('diseased');
        diagnosisDescription.textContent = 'Error: Could not connect to the Flask server.';
    } finally {
        // UI State: End Loading
        predictButton.disabled = false;
        loadingMessage.style.display = 'none';
    }
});