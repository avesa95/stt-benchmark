from pywhispercpp.model import Model

# Initialize the model
model = Model(
    "/Users/ristoc/Workspaces/cube/stt-benchmark/whisper.cpp/models/ggml-tiny.bin"
)

# Transcribe an audio file
segments = model.transcribe(
    "/Users/ristoc/Workspaces/cube/stt-benchmark/data/M18_05_01.wav",
    new_segment_callback=print,
    print_progress=False,
)

for i, segment in enumerate(segments):
    print(f"{i}: {segment.text}")
