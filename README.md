# Whisper.cpp for macOS with Python Integration

This README provides step-by-step instructions for installing and using [whisper.cpp](https://github.com/ggerganov/whisper.cpp) on macOS, including Python integration options.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Download a Model](#download-a-model)
- [Using Whisper.cpp](#using-whispercpp)
- [Python Integration](#python-integration)
  - [Option 1: Official Python Bindings](#option-1-official-python-bindings)
  - [Option 2: Custom Python Wrapper](#option-2-custom-python-wrapper)
  - [Option 3: Advanced Integration with PyTorch](#option-3-advanced-integration-with-pytorch)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before starting, ensure you have the following installed on your macOS system:
- Xcode Command Line Tools
- Homebrew
- Python 3.7+

## Installation Steps

1. **Install required dependencies:**

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install necessary tools
brew install cmake
brew install ffmpeg
```

2. **Clone the whisper.cpp repository:**

```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
```

3. **Build the project:**

```bash
make
```

If you want to enable additional optimizations (like AVX, AVX2, or Metal acceleration):

```bash
# For better CPU performance
make clean
WHISPER_NO_AVX=0 WHISPER_NO_AVX2=0 make -j

# For Metal GPU acceleration (Apple Silicon)
make clean
WHISPER_COREML=1 make -j
```

## Download a Model

Download one of the ggml-formatted Whisper models:

```bash
# Download the tiny model (fastest, least accurate)
bash ./models/download-ggml-model.sh tiny

# Other options: base, small, medium, large
# bash ./models/download-ggml-model.sh base
# bash ./models/download-ggml-model.sh small
# bash ./models/download-ggml-model.sh medium
# bash ./models/download-ggml-model.sh large
```

## Using Whisper.cpp

Test your installation by transcribing a sample audio file:

```bash
# Convert an audio file to 16kHz WAV format
ffmpeg -i your-audio-file.mp3 -ar 16000 -ac 1 -c:a pcm_s16le sample.wav

# Run transcription
./main -m models/ggml-tiny.bin -f sample.wav
```

## Python Integration

There are several ways to integrate whisper.cpp with Python:

### Option 1: Official Python Bindings

Use the unofficial but well-maintained Python bindings:

```bash
# Install the Python package
pip install pywhispercpp
uv add pywhispercpp
```

Example usage:

```python
from pywhispercpp.model import Model

# Initialize the model
model = Model(
    "/Users/vesaalexandru/Workspaces/cube/RealTyme/stt-benchmark/whisper.cpp/models/ggml-tiny.bin"
)

# Transcribe an audio file
segments = model.transcribe(
    "/Users/vesaalexandru/Workspaces/cube/RealTyme/stt-benchmark/data/M18_05_01.wav",
    new_segment_callback=print,
    print_progress=False,
)
for i, segment in enumerate(segments):
    print(f"{i}: {segment.text}")

```

