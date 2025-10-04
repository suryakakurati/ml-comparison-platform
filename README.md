# ML Model Comparison Platform

A production-ready Flask web application for training, comparing, and optimizing machine learning models with an intuitive, modern interface.

## Features

### 🚀 Core Functionality
- **Dataset Upload**: Drag-and-drop CSV file upload with instant preview
- **Manual Training**: Configure and train individual ML models with custom hyperparameters
- **Auto-Optimization**: Automatically find the best model and hyperparameters using Grid/Random Search
- **Comprehensive Metrics**: View accuracy, precision, recall, F1-score, and confusion matrices
- **Results Comparison**: Compare all trained models side-by-side

### 🤖 Supported Models
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Neural Network (Multi-layer Perceptron)

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
2. Select your **target column** (the variable you want to predict)
3. Choose a **model type**
4. Adjust the **test split size** (default: 20%)
5. Configure model-specific **hyperparameters**
6. Click **Train Model**
7. View comprehensive results including:
   - Training and test accuracy
   - Precision, recall, F1-score
   - Confusion matrix visualization
   - Hyperparameter configuration

### 3. Auto-Optimization
1. Navigate to the **Auto-Optimize** section
2. Select your **target column**
3. Set **test split size**
4. Choose **search type**:
   - **Grid Search**: Exhaustive search (slower, more thorough)
   - **Random Search**: Sample-based search (faster)
5. Click **Start Auto-Optimization**
6. The system will:
   - Test multiple models automatically
   - Try different hyperparameter combinations
   - Rank results by performance
   - Highlight the best model

### 4. View Results
1. Go to the **Results** section
2. Compare all trained models
3. Models are automatically ranked by F1-score
4. Best model is highlighted with 🏆
5. Click **Clear All** to reset experiments

## Project Structure

```
ml-comparison-platform/
├── app.py                  # Flask backend with ML logic
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── uploads/               # Uploaded datasets (auto-created)
├── static/
│   ├── css/
│   │   └── style.css      # Modern, responsive styling
│   ├── js/
│   │   └── script.js      # Frontend interactions
│   └── plots/             # Generated visualizations
└── templates/
    └── index.html         # Main HTML template
```

## Technical Details

### Backend (Flask)
- **Route Handlers**: Separate endpoints for upload, training, and optimization
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

### Machine Learning Pipeline
1. **Data Upload & Validation**: CSV parsing with comprehensive checks
2. **Preprocessing**: 
   - Missing value imputation
   - Categorical encoding
   - Feature scaling (StandardScaler)
   - Train-test splitting with stratification
3. **Model Training**: Fit models with configured hyperparameters
4. **Evaluation**: Calculate multiple metrics on test set
5. **Visualization**: Generate confusion matrix plots
6. **Optimization**: Grid/Random search with cross-validation

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
**Solution**: Ensure you've activated the virtual environment and installed all dependencies:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

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