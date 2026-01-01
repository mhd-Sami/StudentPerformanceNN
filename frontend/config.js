/**
 * Configuration Page JavaScript
 * Handles subject configuration creation and management
 */

// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// DOM Elements (will be initialized after DOM loads)
let configForm;
let configsList;
let successSection;
let errorSection;

// Load configurations on page load
window.addEventListener('load', () => {
    // Initialize DOM elements
    configForm = document.getElementById('config-form');
    configsList = document.getElementById('configs-list');
    successSection = document.getElementById('success-section');
    errorSection = document.getElementById('error-section');

    // Setup event listeners
    setupFormListener();
    loadConfigurations();
    setupDynamicFields();
});

// Form submission handler
function setupFormListener() {
    console.log('Setting up form listener...', configForm);

    if (!configForm) {
        console.error('Config form not found!');
        return;
    }

    configForm.addEventListener('submit', async (e) => {
        console.log('Form submitted!');
        e.preventDefault();

        hideMessages();
        setLoading(true);

        try {
            // Collect form data
            const subjectName = document.getElementById('subject-name').value.trim();
            const hasMidterm = document.getElementById('has-midterm').value === 'true';

            const configData = {
                id: subjectName.toLowerCase().replace(/\s+/g, '-'),
                name: subjectName,
                description: document.getElementById('subject-description').value.trim() || undefined,
                num_quizzes: parseInt(document.getElementById('num-quizzes').value),
                num_assignments: parseInt(document.getElementById('num-assignments').value),
                has_midterm: hasMidterm,
                quiz1_total: parseInt(document.getElementById('quiz1-total').value),
                quiz2_total: parseInt(document.getElementById('quiz2-total').value),
                quiz3_total: parseInt(document.getElementById('quiz3-total').value),
                quiz4_total: parseInt(document.getElementById('quiz4-total').value),
                assignment1_total: parseInt(document.getElementById('assignment1-total').value),
                assignment2_total: parseInt(document.getElementById('assignment2-total').value),
                assignment3_total: parseInt(document.getElementById('assignment3-total').value),
                assignment4_total: parseInt(document.getElementById('assignment4-total').value),
                // Always include midterm_total (backend requires it), default to 100 when not used
                midterm_total: hasMidterm ? parseInt(document.getElementById('midterm-total').value) : 100
            };

            // Make API request
            const response = await fetch(`${API_BASE_URL}/subject/create`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(configData)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.message || 'Failed to create configuration');
            }

            // Show success message
            showSuccess('Configuration Saved', `Configuration "${configData.name}" created successfully!`);

            // Reset form
            configForm.reset();

            // Reload configurations list
            loadConfigurations();

        } catch (error) {
            console.error('Error:', error);
            showError(error.message);
        } finally {
            setLoading(false);
        }
    });
}

/**
 * Load all configurations from API
 */
async function loadConfigurations() {
    try {
        const response = await fetch(`${API_BASE_URL}/subject/list`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || 'Failed to load configurations');
        }

        displayConfigurations(data.configs);

    } catch (error) {
        console.error('Error loading configurations:', error);
        configsList.innerHTML = `
            <div class="error-message">
                Failed to load configurations: ${error.message}
            </div>
        `;
    }
}

/**
 * Display configurations list
 */
function displayConfigurations(configs) {
    if (!configs || configs.length === 0) {
        configsList.innerHTML = `
            <div class="empty-message">
                <p>No configurations found. Create your first configuration above!</p>
            </div>
        `;
        return;
    }

    configsList.innerHTML = configs.map(config => `
        <div class="config-card" data-config-id="${config.id}">
            <div class="config-header">
                <h3 class="config-name">${config.name}</h3>
                ${config.id !== 'default' ? `
                    <button class="delete-btn" onclick="deleteConfiguration('${config.id}')">
                        🗑️ Delete
                    </button>
                ` : '<span class="default-badge">Default</span>'}
            </div>
            ${config.description ? `<p class="config-description">${config.description}</p>` : ''}
            <div class="config-meta">
                <span class="meta-badge">📝 ${config.num_quizzes} Quiz${config.num_quizzes > 1 ? 'zes' : ''}</span>
                <span class="meta-badge">📄 ${config.num_assignments} Assignment${config.num_assignments > 1 ? 's' : ''}</span>
                <span class="meta-badge">📋 Midterm: ${config.has_midterm ? 'Yes' : 'No'}</span>
            </div>
            <div class="config-details">
                <div class="detail-group">
                    <h4>Quizzes</h4>
                    <div class="detail-values">
                        ${config.total_marks.quizzes.map((total, i) =>
        `<span class="badge">Q${i + 1}: ${total}</span>`
    ).join('')}
                    </div>
                </div>
                <div class="detail-group">
                    <h4>Assignments</h4>
                    <div class="detail-values">
                        ${config.total_marks.assignments.map((total, i) =>
        `<span class="badge">A${i + 1}: ${total}</span>`
    ).join('')}
                    </div>
                </div>
                <div class="detail-group">
                    <h4>Midterm</h4>
                    <div class="detail-values">
                        <span class="badge">Total: ${config.total_marks.midterm}</span>
                    </div>
                </div>
            </div>
        </div>
    `).join('');
}

/**
 * Delete a configuration
 */
async function deleteConfiguration(configId) {
    if (!confirm(`Are you sure you want to delete this configuration?`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/subject/${configId}/delete`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || 'Failed to delete configuration');
        }

        showSuccess('Configuration Deleted', 'Configuration deleted successfully!');
        loadConfigurations();

    } catch (error) {
        console.error('Error:', error);
        showError(error.message);
    }
}

/**
 * Show success message
 */
function showSuccess(title, message) {
    hideMessages(); // Clear any previous messages first
    document.getElementById('success-title').textContent = title;
    document.getElementById('success-message').textContent = message;
    successSection.style.display = 'block';
    successSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Auto-hide after 5 seconds
    setTimeout(() => {
        successSection.style.display = 'none';
    }, 5000);
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
 * Hide all messages
 */
function hideMessages() {
    successSection.style.display = 'none';
    errorSection.style.display = 'none';
}

/**
 * Set loading state
 */
function setLoading(isLoading) {
    const saveBtn = document.getElementById('save-config-btn');
    const btnText = saveBtn.querySelector('.btn-text');

    if (isLoading) {
        saveBtn.classList.add('loading');
        saveBtn.disabled = true;
        if (btnText) btnText.textContent = 'Saving...';
    } else {
        saveBtn.classList.remove('loading');
        saveBtn.disabled = false;
        if (btnText) btnText.textContent = 'Save Configuration';
    }
}

/**
 * Setup dynamic field visibility based on assessment counts
 */
function setupDynamicFields() {
    const numQuizzes = document.getElementById('num-quizzes');
    const numAssignments = document.getElementById('num-assignments');
    const hasMidterm = document.getElementById('has-midterm');

    // Initial update
    updateQuizFields(parseInt(numQuizzes.value));
    updateAssignmentFields(parseInt(numAssignments.value));
    updateMidtermField(hasMidterm.value === 'true');

    // Listen to changes
    numQuizzes.addEventListener('change', (e) => {
        updateQuizFields(parseInt(e.target.value));
    });

    numAssignments.addEventListener('change', (e) => {
        updateAssignmentFields(parseInt(e.target.value));
    });

    hasMidterm.addEventListener('change', (e) => {
        updateMidtermField(e.target.value === 'true');
    });
}

/**
 * Update quiz field visibility
 */
function updateQuizFields(count) {
    for (let i = 1; i <= 4; i++) {
        const field = document.getElementById(`quiz${i}-total`);
        const group = field.closest('.form-group');

        if (i <= count) {
            group.style.display = 'block';
            field.required = true;
        } else {
            group.style.display = 'none';
            field.required = false;
        }
    }
}

/**
 * Update assignment field visibility
 */
function updateAssignmentFields(count) {
    for (let i = 1; i <= 4; i++) {
        const field = document.getElementById(`assignment${i}-total`);
        const group = field.closest('.form-group');

        if (i <= count) {
            group.style.display = 'block';
            field.required = true;
        } else {
            group.style.display = 'none';
            field.required = false;
        }
    }
}

/**
 * Update midterm field visibility
 */
function updateMidtermField(hasMidterm) {
    const midtermCard = document.querySelector('.form-card:has(#midterm-total)');
    if (midtermCard) {
        midtermCard.style.display = hasMidterm ? 'block' : 'none';
        document.getElementById('midterm-total').required = hasMidterm;
    }
}

// Make deleteConfiguration available globally for onclick handlers
window.deleteConfiguration = deleteConfiguration;
