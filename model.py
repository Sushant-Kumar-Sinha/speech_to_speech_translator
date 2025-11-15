import torch
import numpy as np
import librosa
import tempfile
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoModelForSeq2SeqLM, AutoTokenizer
from gtts import gTTS
import pygame
import threading
from collections import OrderedDict

def warm_up_models(asr_model, asr_processor, translator_model, translator_tokenizer):
    print("🚀 Warming up models... please wait (first-time latency only)")

    # Warm up ASR model
    dummy_audio = np.zeros(16000)
    inputs = asr_processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        _ = asr_model.generate(**inputs)

    # Warm up translation model
    dummy_text = "Hello"
    inputs = translator_tokenizer(dummy_text, return_tensors="pt")
    with torch.no_grad():
        _ = translator_model.generate(**inputs)

    # Warm up TTS
    try:
        tts = gTTS(text="Test", lang="en", slow=False)
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            temp_path = tmp_file.name
        tts.save(temp_path)
        os.unlink(temp_path)
    except Exception as e:
        print(f"⚠️ gTTS warm-up skipped: {e}")

    print("✅ Warm-up complete. Models are ready.\n")


class GoogleTTSWrapper:
    def __init__(self):
        pygame.mixer.init()
        self.language_map = {
            'hindi': 'hi', 'english': 'en', 'bengali': 'bn', 'tamil': 'ta',
            'telugu': 'te', 'marathi': 'mr', 'gujarati': 'gu', 'kannada': 'kn',
            'malayalam': 'ml', 'punjabi': 'pa', 'odia': 'or', 'assamese': 'as', 'urdu': 'ur'
        }

    def get_language_code(self, language_name):
        return self.language_map.get(language_name.lower(), 'hi')

    def text_to_speech(self, text, language_name):
        try:
            lang_code = self.get_language_code(language_name)
            print(f"🔊 Generating {language_name.title()} TTS...")

            tts = gTTS(text=text, lang=lang_code, slow=False)
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                temp_path = tmp_file.name
            tts.save(temp_path)

            threading.Thread(target=self._play_audio, args=(temp_path,), daemon=True).start()
            return True

        except Exception as e:
            print(f"❌ Google TTS error: {e}")
            return False

    def _play_audio(self, audio_file):
        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            pygame.mixer.music.unload()
            pygame.time.wait(150)
            os.unlink(audio_file)
        except Exception as e:
            print(f"❌ Audio playback error: {e}")


class NLLBTranslator:
    def __init__(self):
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.translation_cache = OrderedDict()
        self.cache_size = 500
        self.tokenizer = None
        self.model = None

    def load_models(self):
        print("🔄 Loading NLLB translation model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.eval()
            print("✅ NLLB translation model loaded")
        except Exception as e:
            print(f"❌ NLLB loading failed: {e}")
            raise

    def translate_text(self, text, source_lang, target_lang):
        try:
            if not text.strip():
                return text

            cache_key = f"{source_lang}{target_lang}{text.strip().lower()}"
            if cache_key in self.translation_cache:
                return self.translation_cache[cache_key]

            lang_map = {
                'english': 'eng_Latn', 'hindi': 'hin_Deva', 'bengali': 'ben_Beng',
                'tamil': 'tam_Taml', 'telugu': 'tel_Telu', 'marathi': 'mar_Deva',
                'gujarati': 'guj_Gujr', 'kannada': 'kan_Knda', 'malayalam': 'mal_Mlym',
                'punjabi': 'pan_Guru', 'odia': 'ory_Orya', 'assamese': 'asm_Beng', 'urdu': 'urd_Arab'
            }

            src_code = lang_map.get(source_lang.lower(), 'eng_Latn')
            tgt_code = lang_map.get(target_lang.lower(), 'hin_Deva')

            self.tokenizer.src_lang = src_code
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=128,
                    num_beams=3,
                    early_stopping=True
                )

            translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            self.translation_cache[cache_key] = translated_text
            if len(self.translation_cache) > self.cache_size:
                self.translation_cache.popitem(last=False)

            return translated_text

        except Exception as e:
            print(f"❌ Translation error: {e}")
            return text


class LowLatencyTranslator:
    def __init__(self):
        self.asr_model = None
        self.asr_processor = None
        self.translator = None
        self.tts = None

        self.sample_rate = 16000
        self.chunk_duration = 2.5
        self.source_lang = 'english'
        self.target_lang = 'hindi'
        self.is_recording = False

        self.load_models()

    def load_models(self):
        print("🔄 Loading models...")
        try:
            # Using faster Whisper model
            self.asr_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            self.asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            self.asr_model.eval()
            print("✅ ASR model loaded")

            self.translator = NLLBTranslator()
            self.translator.load_models()

            self.tts = GoogleTTSWrapper()
            print("✅ TTS initialized")

        except Exception as e:
            print(f"❌ Error loading models: {e}")

    def speech_to_text(self, audio_data):
        try:
            inputs = self.asr_processor(audio_data, sampling_rate=self.sample_rate, return_tensors="pt")
            with torch.no_grad():
                generated_ids = self.asr_model.generate(
                    inputs.input_features,
                    language="en",
                    task="transcribe",
                    max_length=100
                )
            transcription = self.asr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return transcription.strip()
        except Exception as e:
            print(f"❌ ASR Error: {e}")
            return ""