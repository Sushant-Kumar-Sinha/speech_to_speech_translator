import torch
import numpy as np
import tempfile
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoModelForSeq2SeqLM, AutoTokenizer
from gtts import gTTS
from collections import OrderedDict

# -------------------- WARM-UP --------------------
def warm_up_models(asr_model, asr_processor, translator_model, translator_tokenizer):
    print("🚀 Warming up models... please wait (first-time latency only)")

    # Warm up ASR model (using English model for warm-up)
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


# -------------------- TTS --------------------
class GoogleTTSWrapper:
    def __init__(self):
        self.language_map = {
            'hindi': 'hi', 'english': 'en', 'bengali': 'bn', 'tamil': 'ta',
            'telugu': 'te', 'marathi': 'mr', 'gujarati': 'gu', 'kannada': 'kn',
            'malayalam': 'ml', 'punjabi': 'pa', 'odia': 'or', 'assamese': 'as', 'urdu': 'ur'
        }

    def get_language_code(self, language_name):
        return self.language_map.get(language_name.lower(), 'hi')

    def text_to_speech_file(self, text, language_name):
        try:
            lang_code = self.get_language_code(language_name)
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tmp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tts.save(tmp_file.name)
            return tmp_file.name
        except Exception as e:
            print(f"❌ Google TTS error: {e}")
            return None

    def text_to_speech(self, text, language_name):
        return self.text_to_speech_file(text, language_name)


# -------------------- TRANSLATOR --------------------
class NLLBTranslator:
    def __init__(self):
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.translation_cache = OrderedDict()
        self.cache_size = 500
        self.tokenizer = None
        self.model = None

    def load_models(self):
        print("🔄 Loading NLLB translation model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.eval()
        print("✅ NLLB translation model loaded")

    def translate_text(self, text, source_lang, target_lang):
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

# -------------------- LOW LATENCY TRANSLATOR --------------------
class LowLatencyTranslator:
    def __init__(self):
        # Keep the same attribute names for app.py compatibility
        self.asr_model = None  # This will be the active model based on language
        self.asr_processor = None  # This will be the active processor
        self.asr_model_en = None   # Fast model for English
        self.asr_processor_en = None
        self.asr_model_other = None   # Model for other languages
        self.asr_processor_other = None
        self.translator = None
        self.tts = None

        self.sample_rate = 16000
        self.source_lang = 'english'
        self.target_lang = 'hindi'

        self.whisper_lang_map = {
            'english': 'en',
            'hindi': 'hi',
            'bengali': 'bn', 'tamil': 'ta', 'telugu': 'te', 'marathi': 'mr',
            'gujarati': 'gu', 'kannada': 'kn', 'malayalam': 'ml', 'punjabi': 'pa',
            'odia': 'or', 'assamese': 'as', 'urdu': 'ur'
        }

        self.load_models()
        # Use English model for warm-up (maintains compatibility)
        warm_up_models(self.asr_model_en, self.asr_processor_en, self.translator.model, self.translator.tokenizer)

    def load_models(self):
        print("🔄 Loading optimized ASR models...")
        
        # Fast model for English (whisper-tiny)
        print("📥 Loading whisper-tiny for English...")
        self.asr_processor_en = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.asr_model_en = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        self.asr_model_en.eval()

        # Model for other languages (whisper-medium)
        print("📥 Loading whisper-medium for other languages...")
        self.asr_processor_other = WhisperProcessor.from_pretrained("openai/whisper-medium")
        self.asr_model_other = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
        self.asr_model_other.eval()
        
        # Set default to English model for compatibility
        self.asr_model = self.asr_model_en
        self.asr_processor = self.asr_processor_en
        
        print("✅ ASR models loaded (tiny for English, medium for other languages)")

        self.translator = NLLBTranslator()
        self.translator.load_models()

        self.tts = GoogleTTSWrapper()
        print("✅ All models initialized")

    def speech_to_text(self, audio_data):
        try:
            # Select the appropriate model based on current language
            if self.source_lang.lower() == 'hindi':
                processor = self.asr_processor_other
                model = self.asr_model_other
                forced_language = 'hi'
                forced_task = "transcribe"
                model_type = "medium"
            elif self.source_lang.lower() == 'english':
                processor = self.asr_processor_en
                model = self.asr_model_en
                forced_language = 'en'
                forced_task = "transcribe"
                model_type = "tiny"
            else:
                # For other languages, use medium model without forcing language
                processor = self.asr_processor_other
                model = self.asr_model_other
                forced_language = None  # No forced language - let Whisper auto-detect
                forced_task = "transcribe"
                model_type = "medium"
            
            # Update the main attributes for any code that might access them
            self.asr_processor = processor
            self.asr_model = model
            
            inputs = processor(
                audio_data, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            )
            
            with torch.no_grad():
                if forced_language:
                    # For English and Hindi - force specific language
                    generated_ids = model.generate(
                        inputs.input_features,
                        language=forced_language,
                        task=forced_task,
                        max_length=100,
                        num_beams=3,  # Reduced for speed
                        temperature=0.0,
                    )
                else:
                    # For other languages - no forced language (auto-detect)
                    generated_ids = model.generate(
                        inputs.input_features,
                        task=forced_task,
                        max_length=100,
                        num_beams=3,  # Reduced for speed
                        temperature=0.0,
                    )
            
            transcription = processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            print(f"🔊 ASR Result [{self.source_lang}, model:{model_type}]: {transcription}")
            return transcription.strip()
            
        except Exception as e:
            print(f"❌ ASR Error: {e}")
            return ""

    def translate_and_tts(self, text, target_lang=None):
        source_lang = self.source_lang
        target_lang = target_lang or self.target_lang

        translated_text = self.translator.translate_text(text, source_lang, target_lang)
        tts_file = self.tts.text_to_speech(translated_text, target_lang)

        if tts_file is None:
            print("❌ TTS generation failed.")
        return translated_text, tts_file
