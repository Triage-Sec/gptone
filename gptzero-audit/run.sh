#!/bin/bash
set -e

# GPTZero Model Extraction Pipeline
# ===================================
# Usage: ./run.sh YOUR_API_KEY
#
# Get a key at: https://app.gptzero.me/app/api
# Starter plan: $45/mo for 300K words
# This pipeline uses ~20K words (~$3)

if [ -z "$1" ] && [ -z "$GPTZERO_API_KEY" ]; then
    echo "Usage: ./run.sh YOUR_GPTZERO_API_KEY"
    echo ""
    echo "Or:    export GPTZERO_API_KEY=your-key"
    echo "       ./run.sh"
    echo ""
    echo "Get an API key at: https://app.gptzero.me/app/api"
    exit 1
fi

API_KEY="${1:-$GPTZERO_API_KEY}"

echo "============================================"
echo "GPTZero Model Extraction Pipeline"
echo "Triage Security"
echo "============================================"
echo ""

# Install deps
echo "Installing dependencies..."
pip install -r requirements.txt -q 2>/dev/null || pip install -r requirements.txt -q --break-system-packages 2>/dev/null
echo "Done."
echo ""

# Run full pipeline
python main.py --api-key "$API_KEY" --delay 0.5

echo ""
echo "============================================"
echo "DELIVERABLES"
echo "============================================"
echo ""
echo "  1. Trained surrogate model:"
echo "     models/surrogate_model.joblib"
echo ""
echo "  2. Test results (surrogate vs GPTZero):"
echo "     data/test_results.json"
echo ""
echo "  3. Full extraction report:"
echo "     report/gptzero_extraction_report.txt"
echo ""
echo "  4. Raw API data:"
echo "     data/extraction_results.json"
echo ""
echo "============================================"
echo "QUICK TEST"
echo "============================================"
echo ""
echo "Test the surrogate on any text:"
echo ""
echo '  python predict.py "Your text here"'
echo ""
echo "Compare surrogate vs GPTZero live:"
echo ""
echo "  python predict.py --compare --api-key $API_KEY \"Your text here\""
echo ""
