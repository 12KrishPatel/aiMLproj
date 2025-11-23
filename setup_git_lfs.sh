#!/bin/bash
# Script to set up Git LFS and pull the actual model/data files

echo "=========================================="
echo "Git LFS Setup for Spam Detector Project"
echo "=========================================="
echo ""

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo " Git LFS is not installed."
    echo ""
    echo "To install Git LFS, run one of the following:"
    echo ""
    echo "  Option 1 (Homebrew - recommended for macOS):"
    echo "    brew install git-lfs"
    echo ""
    echo "  Option 2 (MacPorts):"
    echo "    sudo port install git-lfs"
    echo ""
    echo "  Option 3 (Download installer):"
    echo "    Visit: https://git-lfs.github.com/"
    echo ""
    echo "After installing, run this script again."
    exit 1
fi

echo " Git LFS is installed"
echo ""

# Initialize Git LFS in the repository
echo " Initializing Git LFS..."
git lfs install

# Pull the actual LFS files
echo ""
echo "⬇️  Pulling actual model and data files from Git LFS..."
git lfs pull

echo ""
echo " Done! The actual files should now be available."
echo ""
echo "You can now run:"
echo "  python evaluate_model.py"
echo "  python messagesmodel.py  (to retrain the model)"
echo "  streamlit run app.py     (to start the web app)"

