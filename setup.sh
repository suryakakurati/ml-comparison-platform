#!/bin/bash

# ML Model Comparison Platform - Setup Script
# For macOS/Linux systems

echo "========================================="
echo "ML Model Comparison Platform - Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create required directories
echo "Creating required directories..."
mkdir -p uploads
mkdir -p static/plots
mkdir -p static/css
mkdir -p static/js
mkdir -p templates
echo "✓ Directories created"
echo ""

# Set permissions
echo "Setting permissions..."
chmod +x app.py
echo "✓ Permissions set"
echo ""

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To start the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run the app: python app.py"
echo "3. Open browser: http://localhost:5000"
echo ""
echo "To deactivate virtual environment: deactivate"
echo ""