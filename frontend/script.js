/**
 * Frontend JavaScript
 * Handles form submission, API communication, and results display
 */

// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// DOM Elements
const form = document.getElementById('prediction-form');
const submitBtn = document.getElementById('predict-btn');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');
const subjectSelect = document.getElementById('subject-select');
const modelSelect = document.getElementById('model-select');
const modelStatus = document.getElementById('model-status');

// Current subject configuration
let currentConfig = null;

// Form submission handler
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Clear previous results
    hideResults();
    hideError();

    // Show loading state
    setLoading(true);

    try {
        // Collect form data
        const formData = collectFormData();

        // Make API request
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || 'Prediction failed');
        }
        if (data.success) {
            // Display results
            displayResults(data.predictions, data.comparison, data.debug_info);
        } else {
            showError(data.error || 'An error occurred during prediction.');
        }
    } catch (error) {
        console.error('Error:', error);
        showError(error.message);
    } finally {
        setLoading(false);
    }
});

/**
 * Collect form data
 */
function collectFormData() {
    const formData = new FormData(form);
    const data = {};

    // Get selected subject
    const subjectId = subjectSelect.value;
    if (subjectId) {
        data.subject_id = subjectId;
    }

    for (let [key, value] of formData.entries()) {
        // Skip subject-select as it's already handled
        if (key === 'subject-select') continue;
        data[key] = parseFloat(value);
    }

    // Add selected model type
    if (modelSelect) {
        data.model_type = modelSelect.value;
        console.log("Sending model_type:", data.model_type);
    }

    return data;
}

/**
 * Display prediction results
 */
function displayResults(predictions, comparison = null, debugInfo = []) {
    console.log("Received comparison data:", comparison);
    if (debugInfo && debugInfo.length > 0) {
        console.log("-------------- BACKEND DEBUG INFO --------------");
        debugInfo.forEach(msg => console.log(msg));
        console.log("------------------------------------------------");
    }

    // Pass/Fail
    const passFail = predictions.pass_fail;
    document.getElementById('pass-fail-value').textContent = passFail.prediction;
    document.getElementById('pass-fail-confidence').textContent = `${passFail.confidence}%`;
    document.getElementById('pass-probability').textContent = `${passFail.probability_pass}%`;

    const passFallCard = document.getElementById('pass-fail-card');
    passFallCard.classList.remove('pass', 'fail');
    passFallCard.classList.add(passFail.prediction === 'Pass' ? 'pass' : 'fail');

    // Final Score
    // Final Score
    const finalExam = predictions.final_exam_score;

    // Display score with confidence interval if available
    let scoreDisplay = finalExam.predicted_score;
    if (finalExam.confidence_interval > 0) {
        scoreDisplay += ` <span style="font-size: 0.6em; color: var(--gray-500);">(±${finalExam.confidence_interval})</span>`;
    }

    document.getElementById('final-score-value').innerHTML = scoreDisplay;
    document.getElementById('final-grade').textContent = finalExam.grade;

    // Support
    const support = predictions.support_needed;
    document.getElementById('support-value').textContent = support.prediction;
    document.getElementById('support-confidence').textContent = `${support.confidence}%`;

    const supportCard = document.getElementById('support-card');
    supportCard.classList.remove('support-yes', 'support-no');
    supportCard.classList.add(support.prediction === 'Yes' ? 'support-yes' : 'support-no');

    // Overall Assessment
    document.getElementById('overall-assessment').textContent = predictions.overall_assessment;

    // Model Comparison (if available)
    const comparisonContainer = document.getElementById('comparison-container');
    try {
        if (comparison && comparisonContainer) {
            console.log("Attempting to render comparison card...", comparison);
            comparisonContainer.style.display = 'block';
            comparisonContainer.style.marginBottom = '20px'; // Force spacing
            console.log("Set display to block");

            const agreement = comparison.agreement;
            const color = agreement ? 'var(--success-600)' : 'var(--warning-600)';
            const icon = agreement ? '✓' : '⚠️';

            comparisonContainer.innerHTML = `
                <div style="background: var(--white); border-radius: var(--radius-md); padding: var(--space-4); border: 2px solid ${agreement ? 'var(--success-100)' : 'var(--warning-100)'}; box-shadow: var(--shadow-sm);">
                    <h4 style="margin-bottom: 0.5rem; font-size: 0.9rem; color: var(--gray-600);">Model Consensus</h4>
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <span style="font-weight: 600; color: ${color}; display: flex; align-items: center; gap: 0.5rem;">
                                ${icon} ${agreement ? 'Models Agree' : 'Models Diverge'}
                            </span>
                            <div style="font-size: 0.85rem; color: var(--gray-500); margin-top: 0.25rem;">
                                ${comparison.other_model_name} predicts: <strong>${comparison.other_pass_fail}</strong> (${comparison.other_score.toFixed(1)})
                            </div>
                        </div>
                    </div>
                </div>
            `;
            console.log("Rendered comparison HTML");
        } else if (comparisonContainer) {
            console.log("Hiding comparison container (no data or container missing)");
            comparisonContainer.style.display = 'none';
        }
    } catch (e) {
        console.error("CRITICAL ERROR Rendering Comparison Card:", e);
    }

    // Recommendation
    document.getElementById('recommendation').textContent = support.recommendation;

    // Features (if available)
    // Features (if available)
    if (predictions.features_used && predictions.num_features) {
        displayFeatures(predictions.features_used, predictions.num_features, predictions.feature_importance);
    }

    // Show results section
    showResults();
}

/**
 * Display features used by the model
 */
function displayFeatures(features, numFeatures, featureImportance = {}) {
    const featuresGrid = document.getElementById('features-grid');
    const featureCount = document.getElementById('feature-count');
    const modelNameDisplay = document.getElementById('model-name-display');

    // Update feature count
    if (featureCount) {
        featureCount.textContent = numFeatures;
    }

    // Update model name
    if (modelNameDisplay && modelSelect) {
        const selectedOption = modelSelect.options[modelSelect.selectedIndex];
        modelNameDisplay.textContent = selectedOption.text;
    }

    // Convert to array for sorting
    let featureEntries = Object.entries(features);

    // Sort by importance if available
    if (featureImportance && Object.keys(featureImportance).length > 0) {
        featureEntries.sort((a, b) => {
            const impA = featureImportance[a[0]] || 0;
            const impB = featureImportance[b[0]] || 0;
            return impB - impA; // Descending order
        });
    }

    // Create feature cards
    const featureCards = featureEntries.map(([key, value]) => {
        const displayName = formatFeatureName(key);
        const displayValue = typeof value === 'number' ? value.toFixed(2) : value;
        const importance = featureImportance ? (featureImportance[key] || 0) : 0;

        let importanceIndicator = '';
        if (importance > 0.05) { // Highlight important features
            importanceIndicator = `<div style="height: 4px; background: var(--primary-500); width: ${Math.min(importance * 500, 100)}%; border-radius: 2px; margin-top: 4px;"></div>`;
        }

        return `
            <div class="feature-item" style="flex-direction: column; align-items: stretch; gap: 4px;" title="Importance: ${(importance * 100).toFixed(1)}%">
                <div style="display: flex; justify-content: space-between; width: 100%;">
                    <span class="feature-name">${displayName}</span>
                    <span class="feature-value">${displayValue}</span>
                </div>
                ${importanceIndicator}
            </div>
        `;
    }).join('');

    featuresGrid.innerHTML = featureCards;
}

/**
 * Format feature names to be user-friendly
 */
function formatFeatureName(key) {
    const nameMap = {
        'attendance': 'Attendance',
        'quiz1': 'Quiz 1',
        'quiz2': 'Quiz 2',
        'quiz3': 'Quiz 3',
        'quiz4': 'Quiz 4',
        'assignment1': 'Assignment 1',
        'assignment2': 'Assignment 2',
        'assignment3': 'Assignment 3',
        'assignment4': 'Assignment 4',
        'midterm': 'Midterm',
        'quiz_avg': 'Quiz Average',
        'assignment_avg': 'Assignment Average',
        'overall_avg': 'Overall Average',
        'quiz_trend': 'Quiz Improvement',
        'assignment_trend': 'Assignment Improvement',
        'quiz_std': 'Quiz Consistency',
        'assignment_std': 'Assignment Consistency',
        'attendance_score_interaction': 'Attendance Impact',
        'high_performer': 'High Performer',
        'low_performer': 'Low Performer'
    };

    return nameMap[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Show results section
 */
function showResults() {
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Hide results section
 */
function hideResults() {
    resultsSection.style.display = 'none';
}

/**
 * Show error message
 */
function showError(message) {
    document.getElementById('error-message').textContent = message;
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Hide error section
 */
function hideError() {
    errorSection.style.display = 'none';
}

/**
 * Set loading state
 */
function setLoading(isLoading) {
    if (isLoading) {
        submitBtn.classList.add('loading');
        submitBtn.disabled = true;
        document.querySelector('.btn-text').textContent = 'Analyzing...';
    } else {
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
        document.querySelector('.btn-text').textContent = 'Predict Performance';
    }
}

/**
 * Auto-fill demo data (for testing)
 */
function fillDemoData() {
    document.getElementById('attendance').value = '85';
    document.getElementById('quiz1').value = '78';
    document.getElementById('quiz2').value = '82';
    document.getElementById('quiz3').value = '75';
    document.getElementById('quiz4').value = '88';
    document.getElementById('assignment1').value = '85';
    document.getElementById('assignment2').value = '90';
    document.getElementById('assignment3').value = '87';
    document.getElementById('assignment4').value = '92';
    document.getElementById('midterm').value = '80';
}

// Add keyboard shortcut for demo data (Ctrl+D)
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'd') {
        e.preventDefault();
        fillDemoData();
    }
});

// Load subject configurations on page load
window.addEventListener('load', async () => {
    // Load available models
    await loadAvailableModels();

    // Load subjects
    await loadSubjects();

    // Check API health
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        if (!data.models_loaded) {
            console.warn('Models not loaded. Please train models first.');
        } else {
            console.log('✓ API is healthy and models are loaded');
        }
    } catch (error) {
        console.error('API health check failed:', error);
        showError('Cannot connect to API. Please ensure the backend server is running.');
    }
});

/**
 * Load available ML models
 */
async function loadAvailableModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/models/available`);
        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error('Failed to load models');
        }

        // Populate model dropdown
        if (data.models && data.models.length > 0) {
            modelSelect.innerHTML = data.models.map(model =>
                `<option value="${model.id}">${model.name}</option>`
            ).join('');

            // Set default model
            if (data.default) {
                modelSelect.value = data.default;
            }

            // Update status
            if (modelStatus) {
                modelStatus.textContent = 'Ready';
            }

            console.log(`✓ Loaded ${data.models.length} model(s):`, data.models.map(m => m.name).join(', '));
        }

    } catch (error) {
        console.error('Error loading models:', error);
        // Fallback to Random Forest only
        modelSelect.innerHTML = '<option value="random_forest">Random Forest</option>';
        if (modelStatus) {
            modelStatus.textContent = 'Limited';
        }
    }
}


/**
 * Load available subject configurations
 */
async function loadSubjects() {
    try {
        const response = await fetch(`${API_BASE_URL}/subject/list`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || 'Failed to load subjects');
        }

        // Populate subject dropdown
        subjectSelect.innerHTML = data.configs.map(config =>
            `<option value="${config.id}">${config.name}</option>`
        ).join('');

        // Load default configuration
        await loadSubjectConfig('default');

    } catch (error) {
        console.error('Error loading subjects:', error);
        subjectSelect.innerHTML = '<option value="default">Default (Out of 100)</option>';
    }
}

/**
 * Load a specific subject configuration
 */
async function loadSubjectConfig(configId) {
    try {
        const response = await fetch(`${API_BASE_URL}/subject/${configId}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || 'Failed to load configuration');
        }

        currentConfig = data.config;
        updateFieldVisibility();
        updateTotalMarks();
        updateInputValidation();

    } catch (error) {
        console.error('Error loading configuration:', error);
    }
}

/**
 * Update field visibility based on configuration
 */
function updateFieldVisibility() {
    if (!currentConfig) return;

    const numQuizzes = currentConfig.num_quizzes || 4;
    const numAssignments = currentConfig.num_assignments || 4;
    const hasMidterm = currentConfig.has_midterm !== false;

    // Show/hide quiz fields
    for (let i = 1; i <= 4; i++) {
        const group = document.getElementById(`quiz${i}-group`);
        if (group) {
            if (i <= numQuizzes) {
                group.style.display = 'block';
                document.getElementById(`quiz${i}`).required = true;
            } else {
                group.style.display = 'none';
                document.getElementById(`quiz${i}`).required = false;
            }
        }
    }

    // Show/hide assignment fields
    for (let i = 1; i <= 4; i++) {
        const group = document.getElementById(`assignment${i}-group`);
        if (group) {
            if (i <= numAssignments) {
                group.style.display = 'block';
                document.getElementById(`assignment${i}`).required = true;
            } else {
                group.style.display = 'none';
                document.getElementById(`assignment${i}`).required = false;
            }
        }
    }

    // Show/hide midterm
    const midtermGroup = document.getElementById('midterm-group');
    if (midtermGroup) {
        if (hasMidterm) {
            midtermGroup.style.display = 'block';
            document.getElementById('midterm').required = true;
        } else {
            midtermGroup.style.display = 'none';
            document.getElementById('midterm').required = false;
        }
    }
}

/**
 * Update total marks display
 */
function updateTotalMarks() {
    if (!currentConfig) return;

    // Update quiz totals
    document.getElementById('quiz1-total').textContent = `(out of ${currentConfig.quiz1_total})`;
    document.getElementById('quiz2-total').textContent = `(out of ${currentConfig.quiz2_total})`;
    document.getElementById('quiz3-total').textContent = `(out of ${currentConfig.quiz3_total})`;
    document.getElementById('quiz4-total').textContent = `(out of ${currentConfig.quiz4_total})`;

    // Update assignment totals
    document.getElementById('assignment1-total').textContent = `(out of ${currentConfig.assignment1_total})`;
    document.getElementById('assignment2-total').textContent = `(out of ${currentConfig.assignment2_total})`;
    document.getElementById('assignment3-total').textContent = `(out of ${currentConfig.assignment3_total})`;
    document.getElementById('assignment4-total').textContent = `(out of ${currentConfig.assignment4_total})`;

    // Update midterm total
    document.getElementById('midterm-total').textContent = `(out of ${currentConfig.midterm_total})`;
}

/**
 * Update input field validation based on configuration
 */
function updateInputValidation() {
    if (!currentConfig) return;

    // Update max values for inputs
    document.getElementById('quiz1').max = currentConfig.quiz1_total;
    document.getElementById('quiz2').max = currentConfig.quiz2_total;
    document.getElementById('quiz3').max = currentConfig.quiz3_total;
    document.getElementById('quiz4').max = currentConfig.quiz4_total;

    document.getElementById('assignment1').max = currentConfig.assignment1_total;
    document.getElementById('assignment2').max = currentConfig.assignment2_total;
    document.getElementById('assignment3').max = currentConfig.assignment3_total;
    document.getElementById('assignment4').max = currentConfig.assignment4_total;

    document.getElementById('midterm').max = currentConfig.midterm_total;

    // Update placeholders
    document.getElementById('quiz1').placeholder = `0-${currentConfig.quiz1_total}`;
    document.getElementById('quiz2').placeholder = `0-${currentConfig.quiz2_total}`;
    document.getElementById('quiz3').placeholder = `0-${currentConfig.quiz3_total}`;
    document.getElementById('quiz4').placeholder = `0-${currentConfig.quiz4_total}`;

    document.getElementById('assignment1').placeholder = `0-${currentConfig.assignment1_total}`;
    document.getElementById('assignment2').placeholder = `0-${currentConfig.assignment2_total}`;
    document.getElementById('assignment3').placeholder = `0-${currentConfig.assignment3_total}`;
    document.getElementById('assignment4').placeholder = `0-${currentConfig.assignment4_total}`;

    document.getElementById('midterm').placeholder = `0 - ${currentConfig.midterm_total} `;
}

// Subject selection change handler
subjectSelect.addEventListener('change', async (e) => {
    await loadSubjectConfig(e.target.value);
});
