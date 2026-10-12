from context import context_processing
import edge_tts
import asyncio
import os
import numpy as np
import sounddevice as sd
import pygame
from faster_whisper import WhisperModel

cp = context_processing

kfc_dialog_list = [
    "Dạ KFC xin chào anh/chị ạ! Hôm nay mình muốn dùng món gì nè?",
    "Anh/chị có lưu ý gì về dị ứng hoặc thành phần nào không muốn dùng hôm nay không ạ?",
    "Dạ hôm nay nhà mình/nhóm mình đi bao nhiêu người ạ?",
    "Anh/chị dự định chi tiêu trong tầm giá khoảng bao nhiêu cho bữa ăn hôm nay ạ?",
]
thank_you = "Em cảm ơn thông tin của anh/chị! Hệ thống đang chọn lọc các món phù hợp nhất, anh/chị đợi em một chút nhé."
VOICE = 'vi-VN-HoaiMyNeural'
SAMPLE_RATE = 16000         
model = WhisperModel("small", device="cpu", compute_type="int8")

pygame.mixer.init()

async def generate_audio(text, output_file):
    if os.path.exists(output_file):     
        return
    communicate = edge_tts.Communicate(text=text, voice=VOICE)
    await communicate.save(output_file)

def play_audio(file):
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy(): 
        pygame.time.wait(100)
    pygame.mixer.music.unload()


def record_until_silence(threshold=0.01, silence_sec=1.5, max_sec=15, wait_sec=8):
    block = int(SAMPLE_RATE * 0.1)       
    frames = []
    started = False
    silent_blocks = 0
    total_blocks = 0
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        while True:
            data, _ = stream.read(block)
            data = data.flatten()
            total_blocks += 1
            volume = np.sqrt(np.mean(data ** 2))   
            if volume > threshold:
                started = True
                silent_blocks = 0
            elif started:
                silent_blocks += 1
            if started:
                frames.append(data)
            if started and silent_blocks * 0.1 >= silence_sec:
                break                            
            if not started and total_blocks * 0.1 >= wait_sec:
                break                               
            if total_blocks * 0.1 >= max_sec:
                break

    if not frames:
        return None
    return np.concatenate(frames)

def speech_to_text(audio):
    segments, _ = model.transcribe(
        audio,                 
        language="vi",
        beam_size=5,
        vad_filter=True,     
    )
    return " ".join(s.text for s in segments).strip()

async def greet_and_ask():
    for i, text in enumerate(kfc_dialog_list, start=1):
        await generate_audio(text, f"question_{i}.mp3")
    await generate_audio(thank_you, "thank_you.mp3")
    answers = []
    for i, question in enumerate(kfc_dialog_list, start=1):
        print(f"🤖 {question}")
        play_audio(f"question_{i}.mp3")
        text = ""
        for attempt in range(2):                     
            audio = record_until_silence()
            text = speech_to_text(audio) if audio is not None else ""
            if text:
                break
            await generate_audio("Dạ em chưa nghe rõ, anh chị nói lại giúp em nhé.", "repeat.mp3")
            play_audio("repeat.mp3")
        answers.append({"question": question, "answer": text})
    play_audio("thank_you.mp3")
    cp.context_extraction(answers)


if __name__ == "__main__":
    asyncio.run(greet_and_ask())