#!/usr/bin/env bash
# Bootstrap script for the Audiblez + Kokoro-82M pipeline
# Usage: bash setup.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=== Audiblez + Kokoro-82M Pipeline Setup ==="
echo ""

# 1. Check system dependencies
echo "[1/5] Checking system dependencies..."

MISSING=""

if ! command -v espeak-ng &>/dev/null && ! command -v espeak &>/dev/null; then
    MISSING="$MISSING espeak-ng"
fi

if ! command -v ffmpeg &>/dev/null; then
    MISSING="$MISSING ffmpeg"
fi

if [ -n "$MISSING" ]; then
    echo ""
    echo "ERROR: Missing system dependencies:$MISSING"
    echo ""
    echo "Install them with:"
    echo "  sudo apt install$MISSING"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

echo "  espeak-ng: $(command -v espeak-ng || command -v espeak)"
echo "  ffmpeg:    $(command -v ffmpeg)"

# Optional: check for Calibre
if command -v ebook-convert &>/dev/null; then
    echo "  calibre:   $(command -v ebook-convert) (optional, for format conversion)"
else
    echo "  calibre:   not found (optional, install for PDF/MOBI support)"
fi

# 2. Check Python version
echo ""
echo "[2/5] Checking Python..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "  Python: $PYTHON_VERSION"

# 3. Create virtual environment
echo ""
echo "[3/5] Creating virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "  .venv already exists, reusing"
else
    python3 -m venv "$VENV_DIR"
    echo "  Created: $VENV_DIR"
fi

# 4. Install Python dependencies
echo ""
echo "[4/5] Installing Python dependencies (this may take a while)..."
"$VENV_DIR/bin/pip" install --upgrade pip -q

# Install kokoro first (latest), then audiblez with --no-deps to avoid
# version conflict (audiblez 0.4.9 pins kokoro<0.8.0 but 0.9.4 works fine)
"$VENV_DIR/bin/pip" install "kokoro>=0.9.4" torch torchaudio -q
"$VENV_DIR/bin/pip" install audiblez --no-deps -q
"$VENV_DIR/bin/pip" install ebooklib beautifulsoup4 lxml soundfile numpy misaki tqdm rich pytest -q

# 5. Validate installation
echo ""
echo "[5/5] Validating installation..."

"$VENV_DIR/bin/python" -c "
import sys
print(f'  Python:    {sys.version}')

try:
    import torch
    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else 'CPU only'
    print(f'  PyTorch:   {torch.__version__} (CUDA: {cuda}, Device: {device})')
except ImportError:
    print('  PyTorch:   NOT INSTALLED')

try:
    from kokoro import KPipeline
    print(f'  Kokoro:    OK')
except ImportError:
    print('  Kokoro:    NOT INSTALLED')

try:
    import audiblez
    print(f'  Audiblez:  OK')
except ImportError:
    print('  Audiblez:  NOT INSTALLED')

try:
    import soundfile
    print(f'  soundfile: OK')
except ImportError:
    print('  soundfile: NOT INSTALLED')
"

# Create directories if missing
mkdir -p "$PROJECT_DIR/input" "$PROJECT_DIR/output"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Convert a book:"
echo "  python -m src.pipeline book.epub -v af_sky"
echo ""
echo "Or place .epub files in input/ and run batch mode:"
echo "  python -m src.pipeline --batch input/ -o output/"
