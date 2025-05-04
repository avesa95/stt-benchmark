import whisperx
import gc

model = whisperx.load_model("tiny", "cpu", compute_type="int8")

audio = whisperx.load_audio("/Users/ristoc/Workspaces/cube/stt-benchmark/data/M18_05_01.wav")
result = model.transcribe(audio, batch_size=4)

print(result["segments"])