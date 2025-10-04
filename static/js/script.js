/**
 * ML Model Comparison Platform - Frontend JavaScript
 * Handles all client-side interactions and API calls
 */

// Global state
let currentDataset = null;
let columnNames = [];

// ==========================================
// Initialization
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
    // Setup navigation
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const section = link.getAttribute('data-section');
            
            // Update active state
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            // Scroll to section
            scrollToSection(section);
        });
    });
    
    // Setup file upload
    setupFileUpload();
    
    // Setup range sliders
    setupRangeSliders();
    
    // Setup model type selector
    setupModelSelector();
    
    // Setup tuning method selector
    setupTuningMethodSelector();
    
    // Load available models
    loadAvailableModels();
    
    // Load existing results
    loadResults();
});

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function formatTimestamp(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString();
}

function formatPercentage(value) {
    return (value * 100).toFixed(2) + '%';
}

// ==========================================
// Load Available Models
// ==========================================

async function loadAvailableModels() {
    try {
        const response = await fetch('/available-models');
        const data = await response.json();
        
        if (data.success) {
            const advancedGroup = document.getElementById('advancedModelsGroup');
            if (advancedGroup && data.advanced_models) {
                advancedGroup.innerHTML = '';
                Object.entries(data.advanced_models).forEach(([key, name]) => {
                    const option = document.createElement('option');
                    option.value = key;
                    option.textContent = name;
                    advancedGroup.appendChild(option);
                });
            }
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// ==========================================
// Tuning Method Setup
// ==========================================

function setupTuningMethodSelector() {
    const tuningSelect = document.getElementById('tuningMethod');
    const nTrialsGroup = document.getElementById('nTrialsGroup');
    
    if (tuningSelect && nTrialsGroup) {
        tuningSelect.addEventListener('change', (e) => {
            if (e.target.value === 'optuna') {
                nTrialsGroup.style.display = 'block';
            } else {
                nTrialsGroup.style.display = 'none';
            }
        });
    }
}

// ==========================================
// MLflow UI
// ==========================================

function openMLflowUI() {
    showToast('Starting MLflow UI...', 'success');
    
    // Show instructions
    const message = `
        To view MLflow experiments:
        
        1. Open Terminal
        2. Navigate to your project directory
        3. Run: mlflow ui --backend-store-uri file://./mlruns --port 5001
        4. Open: http://localhost:5001
    `;
    
    alert(message);
}

// ==========================================
// Navigation
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
    // Setup navigation
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const section = link.getAttribute('data-section');
            
            // Update active state
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            // Scroll to section
            scrollToSection(section);
        });
    });
    
    // Setup file upload
    setupFileUpload();
    
    // Setup range sliders
    setupRangeSliders();
    
    // Setup model type selector
    setupModelSelector();
    
    // Load existing results
    loadResults();
});

// ==========================================
// File Upload
// ==========================================

function setupFileUpload() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    
    // Click to upload
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });
    
    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    if (!file.name.endsWith('.csv')) {
        showToast('Please upload a CSV file', 'error');
        return;
    }
    
    showLoading();
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            currentDataset = data;
            columnNames = data.column_names;
            displayDatasetInfo(data);
            populateColumnSelectors();
            showToast('Dataset uploaded successfully!', 'success');
        } else {
            showToast(data.error || 'Upload failed', 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showToast('Upload failed. Please try again.', 'error');
    } finally {
        hideLoading();
    }
}

function displayDatasetInfo(data) {
    document.getElementById('fileName').textContent = data.filename;
    document.getElementById('rowCount').textContent = data.rows.toLocaleString();
    document.getElementById('colCount').textContent = data.columns;
    
    // Display preview table
    const table = document.getElementById('previewTable');
    table.innerHTML = '';
    
    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    data.column_names.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create body
    const tbody = document.createElement('tbody');
    data.preview.forEach(row => {
        const tr = document.createElement('tr');
        data.column_names.forEach(col => {
            const td = document.createElement('td');
            td.textContent = row[col] !== null ? row[col] : 'N/A';
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    
    document.getElementById('datasetInfo').style.display = 'block';
}

function populateColumnSelectors() {
    const selectors = ['targetColumn', 'autoTargetColumn'];
    
    selectors.forEach(selectorId => {
        const select = document.getElementById(selectorId);
        select.innerHTML = '<option value="">Select target column...</option>';
        
        columnNames.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            select.appendChild(option);
        });
    });
}

// ==========================================
// Range Sliders
// ==========================================

function setupRangeSliders() {
    const sliders = [
        { sliderId: 'testSize', valueId: 'testSizeValue', format: (v) => `${(v * 100).toFixed(0)}%` },
        { sliderId: 'autoTestSize', valueId: 'autoTestSizeValue', format: (v) => `${(v * 100).toFixed(0)}%` }
    ];
    
    sliders.forEach(({ sliderId, valueId, format }) => {
        const slider = document.getElementById(sliderId);
        const valueDisplay = document.getElementById(valueId);
        
        if (slider && valueDisplay) {
            slider.addEventListener('input', (e) => {
                valueDisplay.textContent = format(parseFloat(e.target.value));
            });
        }
    });
}

// ==========================================
// Model Parameter Configuration
// ==========================================

function setupModelSelector() {
    const modelSelect = document.getElementById('modelType');
    
    if (modelSelect) {
        modelSelect.addEventListener('change', (e) => {
            displayModelParams(e.target.value);
        });
        
        // Initialize with default
        displayModelParams(modelSelect.value);
    }
}

function displayModelParams(modelType) {
    const paramsContainer = document.getElementById('modelParams');
    paramsContainer.innerHTML = '';
    
    const paramConfigs = {
        logistic_regression: [
            { name: 'C', label: 'Regularization (C)', type: 'number', default: 1.0, min: 0.01, step: 0.01 },
            { name: 'max_iter', label: 'Max Iterations', type: 'number', default: 1000, min: 100, step: 100 }
        ],
        decision_tree: [
            { name: 'max_depth', label: 'Max Depth', type: 'number', default: 10, min: 1, step: 1 },
            { name: 'min_samples_split', label: 'Min Samples Split', type: 'number', default: 2, min: 2, step: 1 }
        ],
        random_forest: [
            { name: 'n_estimators', label: 'Number of Trees', type: 'number', default: 100, min: 10, step: 10 },
            { name: 'max_depth', label: 'Max Depth', type: 'number', default: 10, min: 1, step: 1 }
        ],
        gradient_boosting: [
            { name: 'n_estimators', label: 'Number of Estimators', type: 'number', default: 100, min: 10, step: 10 },
            { name: 'learning_rate', label: 'Learning Rate', type: 'number', default: 0.1, min: 0.01, step: 0.01 },
            { name: 'max_depth', label: 'Max Depth', type: 'number', default: 3, min: 1, step: 1 }
        ],
        svm: [
            { name: 'C', label: 'Regularization (C)', type: 'number', default: 1.0, min: 0.01, step: 0.01 },
            { name: 'kernel', label: 'Kernel', type: 'select', options: ['rbf', 'linear', 'poly'], default: 'rbf' }
        ],
        neural_network: [
            { name: 'hidden_layer_sizes', label: 'Hidden Layer Size', type: 'number', default: 100, min: 10, step: 10 },
            { name: 'learning_rate_init', label: 'Learning Rate', type: 'number', default: 0.001, min: 0.0001, step: 0.0001 },
            { name: 'max_iter', label: 'Max Iterations', type: 'number', default: 1000, min: 100, step: 100 }
        ]
    };
    
    const params = paramConfigs[modelType] || [];
    
    params.forEach(param => {
        const formGroup = document.createElement('div');
        formGroup.className = 'form-group';
        
        const label = document.createElement('label');
        label.className = 'form-label';
        label.textContent = param.label;
        formGroup.appendChild(label);
        
        let input;
        if (param.type === 'select') {
            input = document.createElement('select');
            input.className = 'form-select';
            param.options.forEach(opt => {
                const option = document.createElement('option');
                option.value = opt;
                option.textContent = opt;
                if (opt === param.default) option.selected = true;
                input.appendChild(option);
            });
        } else {
            input = document.createElement('input');
            input.type = param.type;
            input.className = 'form-input';
            input.value = param.default;
            if (param.min !== undefined) input.min = param.min;
            if (param.step !== undefined) input.step = param.step;
        }
        
        input.id = `param_${param.name}`;
        input.dataset.paramName = param.name;
        formGroup.appendChild(input);
        
        paramsContainer.appendChild(formGroup);
    });
}

function collectModelParams() {
    const params = {};
    const paramInputs = document.querySelectorAll('#modelParams input, #modelParams select');
    
    paramInputs.forEach(input => {
        const paramName = input.dataset.paramName;
        let value = input.value;
        
        // Convert to appropriate type
        if (input.type === 'number') {
            value = parseFloat(value);
        }
        
        // Special handling for neural network hidden_layer_sizes
        if (paramName === 'hidden_layer_sizes') {
            value = parseInt(value);
        }
        
        params[paramName] = value;
    });
    
    return params;
}

// ==========================================
// Manual Training
// ==========================================

async function trainModel() {
    if (!currentDataset) {
        showToast('Please upload a dataset first', 'error');
        return;
    }
    
    const targetColumn = document.getElementById('targetColumn').value;
    if (!targetColumn) {
        showToast('Please select a target column', 'error');
        return;
    }
    
    const modelType = document.getElementById('modelType').value;
    const testSize = parseFloat(document.getElementById('testSize').value);
    const params = collectModelParams();
    
    // Collect preprocessing options
    const missingStrategy = document.getElementById('missingStrategy').value;
    const encodingStrategy = document.getElementById('encodingStrategy').value;
    const scalingStrategy = document.getElementById('scalingStrategy').value;
    
    // Collect tuning options
    const tuningMethod = document.getElementById('tuningMethod').value;
    const cvFolds = parseInt(document.getElementById('cvFolds').value);
    const nTrials = parseInt(document.getElementById('nTrials').value);
    const enableMlflow = document.getElementById('enableMlflow').checked;
    
    const trainButton = document.getElementById('trainButton');
    trainButton.disabled = true;
    trainButton.classList.add('loading');
    showLoading();
    
    try {
        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                target_column: targetColumn,
                model_type: modelType,
                test_size: testSize,
                params: params,
                missing_strategy: missingStrategy,
                encoding_strategy: encodingStrategy,
                scaling_strategy: scalingStrategy,
                tuning_method: tuningMethod,
                cv_folds: cvFolds,
                n_trials: nTrials,
                enable_mlflow: enableMlflow
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            displayTrainResults(data.result);
            showToast('Model trained successfully!', 'success');
            loadResults();
        } else {
            showToast(data.error || 'Training failed', 'error');
        }
    } catch (error) {
        console.error('Training error:', error);
        showToast('Training failed. Please try again.', 'error');
    } finally {
        trainButton.disabled = false;
        trainButton.classList.remove('loading');
        hideLoading();
    }
}

function displayTrainResults(result) {
    const resultsContainer = document.getElementById('trainResults');
    resultsContainer.innerHTML = '';
    resultsContainer.style.display = 'block';
    
    const resultCard = createResultCard(result);
    resultsContainer.appendChild(resultCard);
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ==========================================
// Auto-Optimize
// ==========================================

async function autoOptimize() {
    if (!currentDataset) {
        showToast('Please upload a dataset first', 'error');
        return;
    }
    
    const targetColumn = document.getElementById('autoTargetColumn').value;
    if (!targetColumn) {
        showToast('Please select a target column', 'error');
        return;
    }
    
    const testSize = parseFloat(document.getElementById('autoTestSize').value);
    const searchType = document.getElementById('searchType').value;
    
    // Collect preprocessing options
    const missingStrategy = document.getElementById('autoMissingStrategy').value;
    const encodingStrategy = document.getElementById('autoEncodingStrategy').value;
    const scalingStrategy = document.getElementById('autoScalingStrategy').value;
    
    const autoButton = document.getElementById('autoButton');
    autoButton.disabled = true;
    autoButton.classList.add('loading');
    showLoading();
    
    try {
        const response = await fetch('/auto-optimize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                target_column: targetColumn,
                test_size: testSize,
                search_type: searchType,
                missing_strategy: missingStrategy,
                encoding_strategy: encodingStrategy,
                scaling_strategy: scalingStrategy
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            displayAutoResults(data.results, data.best_model);
            showToast('Auto-optimization completed!', 'success');
            loadResults();
        } else {
            showToast(data.error || 'Auto-optimization failed', 'error');
        }
    } catch (error) {
        console.error('Auto-optimize error:', error);
        showToast('Auto-optimization failed. Please try again.', 'error');
    } finally {
        autoButton.disabled = false;
        autoButton.classList.remove('loading');
        hideLoading();
    }
}

function displayAutoResults(results, bestModel) {
    const resultsContainer = document.getElementById('autoResults');
    resultsContainer.innerHTML = '';
    resultsContainer.style.display = 'block';
    
    // Add header with best model highlight
    const header = document.createElement('div');
    header.style.marginBottom = '2rem';
    header.innerHTML = `
        <h3 style="font-size: 1.5rem; margin-bottom: 1rem; color: var(--accent-purple);">
            🏆 Best Model: ${formatModelName(bestModel.model_type)}
        </h3>
        <p style="color: var(--text-secondary);">
            F1 Score: ${formatPercentage(bestModel.metrics.f1_score)} | 
            Accuracy: ${formatPercentage(bestModel.metrics.test_accuracy)}
        </p>
    `;
    resultsContainer.appendChild(header);
    
    // Display all results
    results.forEach((result, index) => {
        const resultCard = createResultCard(result, index === 0);
        resultsContainer.appendChild(resultCard);
    });
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ==========================================
// Result Card Creation
// ==========================================

function createResultCard(result, isBest = false) {
    const card = document.createElement('div');
    card.className = 'result-card';
    if (isBest) {
        card.style.border = '2px solid var(--accent-purple)';
        card.style.boxShadow = '0 0 30px rgba(139, 92, 246, 0.3)';
    }
    
    const modelName = formatModelName(result.model_type);
    const timestamp = result.timestamp ? formatTimestamp(result.timestamp) : 'N/A';
    
    card.innerHTML = `
        <div class="result-header">
            <h3 class="result-model-name">${isBest ? '🏆 ' : ''}${modelName}</h3>
            <span class="result-timestamp">${timestamp}</span>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <span class="metric-value">${formatPercentage(result.metrics.test_accuracy)}</span>
                <span class="metric-label">Test Accuracy</span>
            </div>
            <div class="metric-card">
                <span class="metric-value">${formatPercentage(result.metrics.train_accuracy)}</span>
                <span class="metric-label">Train Accuracy</span>
            </div>
            <div class="metric-card">
                <span class="metric-value">${formatPercentage(result.metrics.precision)}</span>
                <span class="metric-label">Precision</span>
            </div>
            <div class="metric-card">
                <span class="metric-value">${formatPercentage(result.metrics.recall)}</span>
                <span class="metric-label">Recall</span>
            </div>
            <div class="metric-card">
                <span class="metric-value">${formatPercentage(result.metrics.f1_score)}</span>
                <span class="metric-label">F1 Score</span>
            </div>
            ${result.cv_score ? `
            <div class="metric-card">
                <span class="metric-value">${formatPercentage(result.cv_score)}</span>
                <span class="metric-label">CV Score</span>
            </div>
            ` : ''}
        </div>
        
        ${result.params && Object.keys(result.params).length > 0 ? `
        <div class="params-display">
            <h4 class="params-title">Hyperparameters</h4>
            ${Object.entries(result.params).map(([key, value]) => `
                <div class="param-item">
                    <span class="param-key">${key}</span>
                    <span class="param-value">${formatParamValue(value)}</span>
                </div>
            `).join('')}
        </div>
        ` : ''}
        
        ${result.preprocessing ? `
        <div class="params-display">
            <h4 class="params-title">🔧 Preprocessing Applied</h4>
            ${result.preprocessing.missing_values ? `
                <div class="param-item">
                    <span class="param-key">Missing Values</span>
                    <span class="param-value">${result.preprocessing.missing_values.strategy}</span>
                </div>
            ` : ''}
            ${result.preprocessing.encoding ? `
                <div class="param-item">
                    <span class="param-key">Encoding</span>
                    <span class="param-value">${result.preprocessing.encoding.strategy}</span>
                </div>
            ` : ''}
            ${result.preprocessing.scaling ? `
                <div class="param-item">
                    <span class="param-key">Scaling</span>
                    <span class="param-value">${result.preprocessing.scaling.strategy}</span>
                </div>
            ` : ''}
        </div>
        ` : ''}
        
        ${result.tuning ? `
        <div class="params-display tuning-info">
            <h4 class="params-title">🎯 Hyperparameter Tuning</h4>
            <div class="param-item">
                <span class="param-key">Method</span>
                <span class="param-value">${result.tuning.method.toUpperCase()}</span>
            </div>
            <div class="param-item">
                <span class="param-key">Best Score</span>
                <span class="param-value">${formatPercentage(result.tuning.best_score)}</span>
            </div>
            ${result.tuning.n_trials ? `
                <div class="param-item">
                    <span class="param-key">Trials</span>
                    <span class="param-value">${result.tuning.n_trials}</span>
                </div>
            ` : ''}
            ${result.tuning.cv_folds ? `
                <div class="param-item">
                    <span class="param-key">CV Folds</span>
                    <span class="param-value">${result.tuning.cv_folds}</span>
                </div>
            ` : ''}
        </div>
        ` : ''}
        
        ${result.cv_scores && result.cv_scores.length > 0 ? `
        <div class="params-display">
            <h4 class="params-title">📊 Cross-Validation Scores</h4>
            <div class="param-item">
                <span class="param-key">Mean CV Score</span>
                <span class="param-value">${formatPercentage(result.cv_scores.reduce((a, b) => a + b, 0) / result.cv_scores.length)}</span>
            </div>
        </div>
        ` : ''}
        
        ${result.mlflow_run_id ? `
        <div class="params-display mlflow-info">
            <h4 class="params-title">📊 MLflow Tracking</h4>
            <div class="param-item">
                <span class="param-key">Run ID</span>
                <span class="param-value" style="font-size: 0.75rem;">${result.mlflow_run_id}</span>
            </div>
        </div>
        ` : ''}
        
        ${result.confusion_matrix_plot ? `
        <div class="confusion-matrix">
            <h4 class="params-title">Confusion Matrix</h4>
            <img src="${result.confusion_matrix_plot}" alt="Confusion Matrix">
        </div>
        ` : ''}
    `;
    
    return card;
}

function formatModelName(modelType) {
    const names = {
        'logistic_regression': 'Logistic Regression',
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest',
        'gradient_boosting': 'Gradient Boosting',
        'svm': 'Support Vector Machine',
        'neural_network': 'Neural Network'
    };
    return names[modelType] || modelType;
}

function formatParamValue(value) {
    if (typeof value === 'number') {
        return value.toFixed(4);
    }
    return String(value);
}

// ==========================================
// Results Management
// ==========================================

async function loadResults() {
    try {
        const response = await fetch('/results');
        const data = await response.json();
        
        if (data.success && data.results.length > 0) {
            displayResultsGrid(data.results);
        } else {
            displayEmptyState();
        }
    } catch (error) {
        console.error('Error loading results:', error);
    }
}

function displayResultsGrid(results) {
    const resultsGrid = document.getElementById('resultsGrid');
    resultsGrid.innerHTML = '';
    
    // Sort by F1 score
    const sortedResults = [...results].sort((a, b) => 
        b.metrics.f1_score - a.metrics.f1_score
    );
    
    sortedResults.forEach((result, index) => {
        const card = createCompactResultCard(result, index === 0);
        resultsGrid.appendChild(card);
    });
}

function createCompactResultCard(result, isBest) {
    const card = document.createElement('div');
    card.className = 'result-card';
    if (isBest) {
        card.style.border = '2px solid var(--accent-purple)';
    }
    
    const modelName = formatModelName(result.model_type);
    
    card.innerHTML = `
        <div class="result-header">
            <h3 class="result-model-name" style="font-size: 1.25rem;">
                ${isBest ? '🏆 ' : ''}${modelName}
            </h3>
        </div>
        
        <div class="metrics-grid" style="grid-template-columns: repeat(2, 1fr);">
            <div class="metric-card">
                <span class="metric-value" style="font-size: 1.5rem;">${formatPercentage(result.metrics.test_accuracy)}</span>
                <span class="metric-label">Accuracy</span>
            </div>
            <div class="metric-card">
                <span class="metric-value" style="font-size: 1.5rem;">${formatPercentage(result.metrics.f1_score)}</span>
                <span class="metric-label">F1 Score</span>
            </div>
        </div>
        
        ${result.cv_score ? `
        <div style="text-align: center; margin-top: 1rem; color: var(--text-secondary);">
            CV Score: ${formatPercentage(result.cv_score)}
        </div>
        ` : ''}
    `;
    
    return card;
}

function displayEmptyState() {
    const resultsGrid = document.getElementById('resultsGrid');
    resultsGrid.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">📈</div>
            <h3>No experiments yet</h3>
            <p>Train some models to see results here</p>
        </div>
    `;
}

async function clearResults() {
    if (!confirm('Are you sure you want to clear all results?')) {
        return;
    }
    
    try {
        const response = await fetch('/clear-results', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayEmptyState();
            document.getElementById('trainResults').style.display = 'none';
            document.getElementById('autoResults').style.display = 'none';
            showToast('All results cleared', 'success');
        }
    } catch (error) {
        console.error('Error clearing results:', error);
        showToast('Failed to clear results', 'error');
    }
}

// ==========================================
// Scroll Tracking for Navigation
// ==========================================

window.addEventListener('scroll', () => {
    const sections = ['upload', 'train', 'auto', 'results'];
    const navLinks = document.querySelectorAll('.nav-link');
    
    let currentSection = '';
    
    sections.forEach(sectionId => {
        const section = document.getElementById(sectionId);
        if (section) {
            const rect = section.getBoundingClientRect();
            if (rect.top <= 150 && rect.bottom >= 150) {
                currentSection = sectionId;
            }
        }
    });
    
    if (currentSection) {
        navLinks.forEach(link => {
            if (link.getAttribute('data-section') === currentSection) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }
});