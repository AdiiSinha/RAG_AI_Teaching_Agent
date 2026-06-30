import whisper , os , json
# for file in os.listdir("audios"):
#     print(repr(file))

model = whisper.load_model("large-v2")

# print("Audio file exists:", os.path.exists(
#     r"audios/6_SEO and Core Web Vitals in HTML ｜ Sigma Web Development Course - Tutorial #6 [CyRlWlaJnTY].webm"
# ))


result = model.transcribe(
    audio="audios/6_SEO and Core Web Vitals in HTML ｜ Sigma Web Development Course - Tutorial #6 [CyRlWlaJnTY].webm.mp3",
    language="hi",
    task="translate",
    word_timestamps = False
)

chunks = []
for segment in result["segments"]:
    chunks.append({"start" : segment["start"] , "end" : segment["end"] , "text" : segment["text"]})

print(chunks)

with open("output.json" , "w") as f:
    json.dump(chunks , f)
    
    
# print(result["text"]) 