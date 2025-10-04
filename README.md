# ML Model Comparison Platform

> 🧠 **AI-Assisted Development Notice**
>
> This project was developed by Surya Akurati with assistance from Anthropic's Claude AI for code generation, optimization, and documentation support.  
> All implementation decisions, testing, and integrations were conducted and verified by the developer.

A production-ready Flask web application for training, comparing, and optimizing machine learning models with an intuitive, modern interface. 

## Features

### 🚀 Core Functionality
- **Dataset Upload**: Drag-and-drop CSV file upload with instant preview
- **Manual Training**: Configure and train individual ML models with custom hyperparameters
- **Auto-Optimization**: Automatically find the best model and hyperparameters using Grid/Random Search
- **Comprehensive Metrics**: View accuracy, precision, recall, F1-score, and confusion matrices
- **Results Comparison**: Compare all trained models side-by-side

### 🤖 Supported Models

**Traditional Models:**
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Neural Network (Multi-layer Perceptron)

**Advanced Models (Phase 2):**
- XGBoost - Extreme Gradient Boosting
- LightGBM - Microsoft's gradient boosting framework
- CatBoost - Yandex's gradient boosting library
- Keras MLP - Deep learning neural networks

### 🔧 Advanced Preprocessing Options
- **Missing Values**: Mean, Median, Mode, or Drop rows
- **Categorical Encoding**: Auto (smart), Label Encoding, One-Hot Encoding
- **Feature Scaling**: Standard Scaler, Min-Max Scaler, or No Scaling

### 📊 Features
- Interactive hyperparameter configuration
- Cross-validation scoring
- Confusion matrix visualization
- Ranked model comparison
- Real-time training progress
- Responsive, modern UI design

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Extract the Project
```bash
unzip ml-comparison-platform.zip
cd ml-comparison-platform
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Create Required Directories
```bash
mkdir -p uploads static/plots static/css static/js templates
```

### Step 5: Run the Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage Guide

### 1. Upload Dataset
1. Navigate to the **Upload** section
2. Drag and drop a CSV file or click "Choose File"
3. View dataset preview and column information
4. Ensure your dataset has:
   - At least 10 rows
   - At least 2 columns (features + target)
   - A clear target column for prediction

### 2. Manual Training
1. Go to the **Train** section
2. Select your **target column**
3. Choose a **model type** (traditional or advanced)
4. **Configure preprocessing** (Phase 1):
   - Missing Values: mean/median/mode/drop
   - Encoding: auto/label/onehot
   - Scaling: standard/minmax/none
5. **Configure hyperparameter tuning** (Phase 2 - NEW):
   - **None**: Use default parameters
   - **Grid Search**: Exhaustive parameter search
   - **Random Search**: Fast random sampling
   - **Optuna**: Smart Bayesian optimization (recommended)
6. Set **Cross-Validation folds** (2-10, default: 5)
7. For Optuna: Set number of **trials** (10-100, default: 30)
8. Toggle **MLflow tracking** on/off
9. Configure model-specific hyperparameters (optional)
10. Click **Train Model**
11. View comprehensive results with tuning information

### 3. Auto-Optimization
1. Navigate to the **Auto-Optimize** section
2. Select your **target column**
3. Configure preprocessing settings
4. Choose search type (Grid or Random)
5. Click **Start Auto-Optimization**
6. The system will test multiple models and rank by performance

### 4. View Results
1. Go to the **Results** section
2. Compare all trained models
3. View metrics, confusion matrices, and tuning info
4. Click **View MLflow UI** to see detailed experiment logs
5. Models ranked automatically by F1-score

### 5. MLflow Integration (Phase 2 - NEW)
All experiments are automatically logged to MLflow with:
- Model parameters
- Performance metrics
- Cross-validation scores
- Preprocessing configuration
- Tuning method and results
- Confusion matrix artifacts
- Trained model artifacts

**To view MLflow dashboard:**
```bash
# In separate terminal
mlflow ui --backend-store-uri file://./mlruns --port 5001
# Open: http://localhost:5001
```

## Project Structure

```
ml-comparison-platform/
├── app.py                      # Flask backend with ML logic
├── preprocessing.py            # Preprocessing pipeline module
├── advanced_models.py          # XGBoost, LightGBM, CatBoost, Keras (Phase 2)
├── optuna_tuner.py            # Optuna optimization module (Phase 2)
├── mlflow_tracker.py          # MLflow tracking wrapper (Phase 2)
├── requirements.txt            # Python dependencies (updated)
├── README.md                   # This file
├── uploads/                    # Uploaded datasets
├── experiments/
│   └── optuna_studies/        # Optuna study results (Phase 2)
├── mlruns/                    # MLflow tracking data (Phase 2)
├── static/
│   ├── css/
│   │   └── style.css          # Modern, responsive styling
│   ├── js/
│   │   └── script.js          # Frontend interactions (updated)
│   └── plots/                 # Generated visualizations
└── templates/
    └── index.html             # Main HTML template (updated)
```

## Technical Details

### Backend (Flask)
- **Route Handlers**: Separate endpoints for upload, training, and optimization
- **Preprocessing Module**: Configurable data cleaning, encoding, and scaling
- **Data Preprocessing**: Automatic handling of missing values, encoding, and scaling
- **Model Factory**: Dynamic model instantiation with custom parameters
- **Error Handling**: Comprehensive try-catch blocks with informative error messages
- **Validation**: Input validation at every step to ensure data integrity

### Frontend (HTML/CSS/JavaScript)
- **Modern Design**: Dark theme with gradient accents inspired by contemporary web design
- **Responsive Layout**: Works on desktop, tablet, and mobile devices
- **Real-time Feedback**: Toast notifications and loading indicators
- **Interactive Elements**: Drag-and-drop upload, range sliders, dynamic forms
- **Smooth Animations**: CSS transitions and animations for better UX

## Phase 2 Features Explained

### 1. Hyperparameter Tuning Methods

**None (Default)**
- Uses model default parameters
- Fastest training
- Good for quick experimentation

**Grid Search**
- Tests all parameter combinations
- Exhaustive and thorough
- Slower but finds optimal parameters
- Best for small parameter spaces

**Random Search**
- Randomly samples parameter combinations
- Faster than Grid Search
- Good balance of speed and performance
- Recommended for large parameter spaces

**Optuna (Smart Optimization)**
- Uses Bayesian optimization
- Learns from previous trials
- Most efficient parameter search
- Recommended for best results
- Automatically saves study results

### 2. MLflow Experiment Tracking

**What gets logged:**
- Model type and parameters
- All performance metrics
- Cross-validation scores
- Preprocessing configuration
- Tuning method used
- Confusion matrix images
- Trained model artifacts
- Timestamps

**Benefits:**
- Full experiment reproducibility
- Compare runs over time
- Track model performance
- Share results with team
- Audit trail for compliance

**Accessing MLflow:**
```bash
# Start UI in separate terminal
mlflow ui --backend-store-uri file://./mlruns --port 5001

# Open browser
http://localhost:5001
```

### 3. Advanced Models

**XGBoost**
- Best for: Structured/tabular data
- Strengths: High accuracy, handles missing values
- Speed: Fast
- Use when: You need maximum performance

**LightGBM**
- Best for: Large datasets
- Strengths: Very fast, memory efficient
- Speed: Fastest
- Use when: Dataset > 10,000 rows

**CatBoost**
- Best for: Categorical features
- Strengths: Handles categories natively, robust
- Speed: Medium-Fast
- Use when: Many categorical columns

**Keras MLP**
- Best for: Complex patterns
- Strengths: Deep learning, flexible
- Speed: Slower
- Use when: Linear models fail

### 4. Cross-Validation

**What it does:**
- Splits data into K folds
- Trains K times on different splits
- Averages performance across folds
- Provides robust performance estimate

**When to use:**
- Always! (except for very large datasets)
- Default: 5 folds (recommended)
- Increase folds for small datasets (up to 10)
- Decrease for large datasets (down to 3)

**Benefits:**
- More reliable performance estimates
- Reduces overfitting
- Better model selection Validation**: CSV parsing with comprehensive checks
2. **Preprocessing** (Configurable):
   - Missing value handling (mean/median/mode/drop)
   - Categorical encoding (auto/label/one-hot)
   - Feature scaling (standard/minmax/none)
3. **Data Splitting**: Train-test split with stratification
4. **Model Training**: Fit models with configured hyperparameters
5. **Evaluation**: Calculate multiple metrics on test set
6. **Visualization**: Generate confusion matrix plots
7. **Optimization**: Grid/Random search with cross-validation

## Safety & Best Practices

This application follows medical-grade development standards:

- ✅ **Input Validation**: Every user input is validated
- ✅ **Error Handling**: Comprehensive try-catch blocks throughout
- ✅ **Type Safety**: Explicit type conversions and checks
- ✅ **Resource Management**: Proper file handling and cleanup
- ✅ **Security**: File upload restrictions, secure filenames
- ✅ **Logging**: Error messages logged for debugging
- ✅ **Code Quality**: PEP 8 compliant, well-documented code

## Troubleshooting

### Issue: "Module not found" error
**Solution**: Ensure virtual environment is activated and dependencies installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: TensorFlow/Keras installation fails
**Solution**: On macOS with Apple Silicon:
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

### Issue: Upload fails
**Solution**: 
- Verify CSV file is properly formatted
- Check file size (max 16MB)
- Ensure dataset has at least 10 rows and 2 columns

### Issue: Training takes forever
**Solution**:
- Use Random Search instead of Grid Search
- Reduce number of Optuna trials (try 10-20)
- Use LightGBM instead of other advanced models
- Reduce cross-validation folds to 3

### Issue: Port 5000 already in use
**Solution**: 
```bash
# Find and kill process on port 5000 (macOS/Linux)
lsof -ti:5000 | xargs kill -9

# Or use different port
# Modify app.py, last line: app.run(debug=True, port=5001)
```

### Issue: MLflow UI won't start
**Solution**:
```bash
# Make sure you're in project directory
cd ml-comparison-platform

# Activate virtual environment
source venv/bin/activate

# Use full path
mlflow ui --backend-store-uri file://./mlruns --port 5001
```

### Issue: Optuna optimization fails
**Solution**:
- Ensure model type supports Optuna (check error message)
- Reduce number of trials
- Check that dataset has enough samples

### Issue: Advanced models not showing
**Solution**: Libraries may not be installed:
```bash
pip install xgboost lightgbm catboost tensorflow
```

## Performance Tips

### For Best Accuracy
1. Use **Optuna** tuning method (30-50 trials)
2. Enable **cross-validation** (5 folds)
3. Try **advanced models** (XGBoost, LightGBM)
4. Use **Standard Scaler** for preprocessing
5. Ensure dataset is **clean and balanced**

### For Fastest Results
1. Use **None** or **Random Search** tuning
2. Reduce **CV folds** to 3
3. Use **LightGBM** model
4. Keep **default parameters**
5. Use smaller **test split** (10%)

### For Learning
1. Start with **sample_data.csv**
2. Try **different tuning methods** and compare
3. Experiment with **preprocessing options**
4. Compare **traditional vs advanced models**
5. Review **MLflow logs** to understand patterns

## Advanced Usage

### Custom Optuna Trials
Edit `optuna_tuner.py` to modify search spaces for each model.

### Custom Parameter Grids
Edit `advanced_models.py` or `app.py` function `get_traditional_param_grid()`.

### MLflow Experiment Names
Change experiment name in `app.py`:
```python
mlflow_tracker = MLflowTracker(experiment_name='my_experiment')
```

### Batch Experiments
Use the API endpoints to run experiments programmatically:
```python
import requests

# Train model
response = requests.post('http://localhost:5000/train', json={
    'target_column': 'approved',
    'model_type': 'xgboost',
    'tuning_method': 'optuna',
    'n_trials': 50
})
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Create your own tests in `tests/` directory following the examples.

## Dataset Requirements

Your CSV dataset should:
- Have a header row with column names
- Include a target column (what you want to predict)
- Have at least 10 rows (50+ recommended)
- Have at least 1 feature column
- Use numeric or categorical values
- Ideally have balanced classes

Example structure:
```csv
feature1,feature2,feature3,target
1.2,5.4,cat,yes
2.3,6.1,dog,no
...
```

## Browser Compatibility

Tested and optimized for:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Version History

- **v2.0.0** (Phase 2): Advanced models, Optuna, MLflow, hyperparameter tuning
- **v1.5.0** (Phase 1.5): Preprocessing pipeline
- **v1.0.0** (Phase 1): Core functionality

## What's New in Phase 2

✅ **Hyperparameter Tuning**: Grid, Random, and Optuna optimization
✅ **MLflow Tracking**: Complete experiment logging and reproducibility  
✅ **Advanced Models**: XGBoost, LightGBM, CatBoost, Keras MLP
✅ **Cross-Validation**: Robust performance evaluation
✅ **Optuna Integration**: Smart Bayesian optimization
✅ **Enhanced UI**: Tuning controls and MLflow access
✅ **Experiment Storage**: Optuna studies saved to JSON

## Support & Contribution

For issues or questions:
1. Check troubleshooting section
2. Review MLflow logs for training issues
3. Check browser console for frontend errors
4. Verify all dependencies installed

## License

This project is provided as-is for educational and commercial use.

---

**Built with modern ML best practices and production-grade code quality** 🚀

## Quick Command Reference

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run Flask App
python app.py
# Open: http://localhost:5000

# Run MLflow UI
mlflow ui --backend-store-uri file://./mlruns --port 5001
# Open: http://localhost:5001

# Run Tests
pytest tests/

# Deactivate Environment
deactivate
```

## File Checklist

Ensure you have all these files:
- ✅ `app.py` (updated with Phase 2)
- ✅ `preprocessing.py` (Phase 1)
- ✅ `advanced_models.py` (Phase 2 - NEW)
- ✅ `optuna_tuner.py` (Phase 2 - NEW)
- ✅ `mlflow_tracker.py` (Phase 2 - NEW)
- ✅ `requirements.txt` (updated)
- ✅ `templates/index.html` (updated)
- ✅ `static/css/style.css` (updated)
- ✅ `static/js/script.js` (updated)
- ✅ `sample_data.csv`
- ✅ `README.md` (this file)

**Total: 11 core files + directories**

### Issue: Upload fails
**Solution**: 
- Verify CSV file is properly formatted
- Check file size (max 16MB)
- Ensure dataset has at least 10 rows and 2 columns

### Issue: Training takes too long
**Solution**:
- Use smaller dataset for testing
- Reduce hyperparameter search space
- Use Random Search instead of Grid Search
- Reduce number of cross-validation folds

### Issue: Port 5000 already in use
**Solution**: 
```bash
# Find and kill process on port 5000 (macOS/Linux)
lsof -ti:5000 | xargs kill -9

# Or use a different port
# Modify app.py, line: app.run(debug=True, host='0.0.0.0', port=5001)
```

### Issue: Plots not displaying
**Solution**:
- Ensure `static/plots/` directory exists
- Check file permissions
- Verify matplotlib backend is set correctly

## Performance Tips

1. **For Large Datasets**: 
   - Use Random Search instead of Grid Search
   - Reduce number of hyperparameter combinations
   - Increase test split size to reduce training data

2. **For Better Accuracy**:
   - Try multiple models with Auto-Optimize
   - Experiment with different hyperparameters
   - Ensure balanced class distribution
   - Clean and preprocess data before upload

3. **For Faster Results**:
   - Start with simpler models (Logistic Regression, Decision Tree)
   - Use smaller hyperparameter ranges
   - Reduce cross-validation folds

## Dataset Requirements

Your CSV dataset should:
- Have a header row with column names
- Include a target column (the variable to predict)
- Have at least 10 rows of data
- Have at least 1 feature column (in addition to target)
- Use numeric or categorical values (will be auto-encoded)
- Ideally have balanced classes for classification

Example CSV structure:
```csv
feature1,feature2,feature3,target
1.2,5.4,cat,A
2.3,6.1,dog,B
...
```

## Browser Compatibility

Tested and optimized for:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Support & Contribution

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Review error messages in browser console
3. Check Flask terminal output for backend errors

## License

This project is provided as-is for educational and commercial use.

## Version History

- **v1.0.0** (2025-10-03): Initial release with full functionality

---

**Built with ❤️ using Flask, scikit-learn, and modern web technologies**
