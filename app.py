import os
import tempfile
import numpy as np
import gradio as gr
import logging
from typing import Dict, Any
import subprocess
import time
import threading
import queue
import sounddevice as sd
import librosa

# ✅ Change to correct directory
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
os.chdir(current_directory)
print(f"📁 Server running from: {current_directory}")

# Import your model exactly as it is
from model import LowLatencyTranslator, warm_up_models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ GLOBAL MODELS
class GlobalTranslator:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print("🚀 LOADING MODELS (First time only)...")
                    cls._instance = LowLatencyTranslator()
                    warm_up_models(
                        cls._instance.asr_model,
                        cls._instance.asr_processor,
                        cls._instance.translator.model,
                        cls._instance.translator.tokenizer
                    )
                    print("✅ MODELS LOADED!")
        return cls._instance

# Pre-load models
global_translator = GlobalTranslator.get_instance()

class AudioProcessor:
    def __init__(self, translator: LowLatencyTranslator):
        self.translator = translator
        self.is_processing = False
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        self.current_transcription = ""
        self.current_translation = ""
        self.last_update_time = 0
        self.translation_history = []
        self.max_history_items = 10
        self.status = "disconnected"
        self.current_source_lang = "english"
        self.current_target_lang = "hindi"

    def change_languages(self, source_lang: str, target_lang: str):
        """Change languages"""
        try:
            # Update the translator instance
            self.translator.source_lang = source_lang
            self.translator.target_lang = target_lang
            self.current_source_lang = source_lang
            self.current_target_lang = target_lang
            
            print(f"🌐 LANGUAGES CHANGED: {source_lang} → {target_lang}")
            
            # Also update the global instance
            global_translator = GlobalTranslator.get_instance()
            global_translator.source_lang = source_lang
            global_translator.target_lang = target_lang
            
            return f"✅ Languages changed to {source_lang.title()} → {target_lang.title()}", "connected"
            
        except Exception as e:
            print(f"❌ Language change error: {e}")
            return f"❌ Error changing languages: {str(e)}", "error"

    def start_processing(self):
        """Start live translation - WITH CHUNKING"""
        if self.is_processing:
            return "⚠️ Already processing", "recording"
        
        self.is_processing = True
        self.current_transcription = ""
        self.current_translation = ""
        self.status = "recording"
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio_chunks)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start recording thread
        recording_thread = threading.Thread(target=self._record_audio)
        recording_thread.daemon = True
        recording_thread.start()
        
        return "🎤 Live translation started! Speak now...", "recording"

    def _record_audio(self):
        """Record audio chunks and add to queue"""
        try:
            print(f"🎙 LIVE TRANSLATION STARTED: {self.translator.source_lang} → {self.translator.target_lang}")
            self.translator.is_recording = True

            while self.translator.is_recording and self.is_processing:
                # Record audio CHUNK
                recording = sd.rec(
                    int(self.translator.chunk_duration * self.translator.sample_rate),
                    samplerate=self.translator.sample_rate, 
                    channels=1, 
                    dtype='float32'
                )
                sd.wait()

                # Process audio CHUNK
                audio_data = recording.flatten()
                self.audio_queue.put(audio_data)
                
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Recording error: {e}")
            self.status = "error"

    def _process_audio_chunks(self):
        """Process audio chunks from queue"""
        while self.is_processing:
            try:
                # Get audio chunk with timeout
                audio_data = self.audio_queue.get(timeout=1.0)
                
                # Process audio CHUNK
                transcription = self.translator.speech_to_text(audio_data)
                
                if transcription and transcription.strip():
                    print(f"🎯 [{self.translator.source_lang}] ASR: {transcription}")
                    
                    # Get translation
                    translated_text = self.translator.translator.translate_text(
                        transcription, 
                        self.translator.source_lang, 
                        self.translator.target_lang
                    )
                    
                    print(f"🌐 [{self.translator.target_lang}] Translation: {translated_text}")
                    
                    # Update current results
                    self.current_transcription = transcription
                    self.current_translation = translated_text
                    self.last_update_time = time.time()
                    
                    # Add to history
                    self.add_to_history(transcription, translated_text)
                    
                    # ✅ TTS
                    print(f"🔊 Playing {self.translator.target_lang} TTS: {translated_text}")
                    self.translator.tts.text_to_speech(translated_text, self.translator.target_lang)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                self.status = "error"

    def add_to_history(self, original, translated):
        """Add translation to history"""
        history_item = {
            'original': original,
            'translated': translated,
            'timestamp': time.strftime('%H:%M:%S'),
            'source_lang': self.current_source_lang,
            'target_lang': self.current_target_lang
        }
        self.translation_history.insert(0, history_item)
        
        # Keep only recent items
        if len(self.translation_history) > self.max_history_items:
            self.translation_history.pop()

    def get_history_display(self):
        """Get formatted history for display"""
        if not self.translation_history:
            return "No translations yet. Start speaking or upload a file!"
        
        history_html = "<div style='max-height: 300px; overflow-y: auto;'>"
        for item in self.translation_history:
            history_html += f"""
            <div style="background: white; border: 1px solid #e9ecef; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem; font-size: 0.9rem;">
                <div style="color: #495057; margin-bottom: 0.25rem;"><strong>Original ({item['source_lang']}):</strong> {item['original']}</div>
                <div style="color: #28a745; font-weight: 500; margin-bottom: 0.25rem;"><strong>Translated ({item['target_lang']}):</strong> {item['translated']}</div>
                <small style="color: #6c757d;">{item['timestamp']}</small>
            </div>
            """
        history_html += "</div>"
        return history_html

    def stop_processing(self):
        """Stop live translation"""
        self.is_processing = False
        self.translator.is_recording = False
        self.status = "connected"
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        return "⏹️ Live translation stopped", "connected"

    def get_current_results(self):
        """Get current transcription and translation"""
        current_time = time.strftime('%H:%M:%S')
        
        original_info = f"Detected speech ({current_time})" if self.current_transcription else "Waiting for input..."
        translated_info = f"Translated to {self.current_target_lang} ({current_time})" if self.current_translation else "Waiting for translation..."
        
        return (
            self.current_transcription or "Your speech will be transcribed here...", 
            self.current_translation or "Translation will appear here...",
            original_info,
            translated_info
        )

    def process_audio_file(self, file_path: str):
        """Process audio file - NO CHUNKING"""
        try:
            if not file_path:
                return "❌ Please select an audio file first.", "error"

            print(f"🎵 PROCESSING AUDIO FILE: {self.translator.source_lang} → {self.translator.target_lang}")
            self.status = "processing"

            # Load entire audio file
            audio, sr = librosa.load(file_path, sr=self.translator.sample_rate)
            duration = len(audio) / sr
            print(f"✅ Loaded audio file ({duration:.1f}s)")
            
            # ✅ PROCESS ENTIRE FILE AT ONCE - NO CHUNKING
            transcription = self.translator.speech_to_text(audio)
            
            if transcription and transcription.strip():
                print(f"🎯 [{self.translator.source_lang}] File ASR: {transcription}")
                
                # Get translation
                translated = self.translator.translator.translate_text(
                    transcription, 
                    self.translator.source_lang, 
                    self.translator.target_lang
                )
                print(f"🌐 [{self.translator.target_lang}] File Translation: {translated}")
                
                # Update current results
                self.current_transcription = transcription
                self.current_translation = translated
                self.last_update_time = time.time()
                
                # Add to history
                self.add_to_history(transcription, translated)
                
                # ✅ TTS
                print(f"🔊 Playing {self.translator.target_lang} TTS for file: {translated}")
                self.translator.tts.text_to_speech(translated, self.translator.target_lang)
            
            print("✅ Audio file processing completed")
            self.status = "connected"
            return "✅ Audio file processed successfully!", "connected"

        except Exception as e:
            logger.error(f"Audio file processing error: {e}")
            self.status = "error"
            return f"❌ Error processing audio: {str(e)}", "error"

    def process_video_file(self, file_path: str):
        """Process video file - NO CHUNKING"""
        try:
            if not file_path:
                return "❌ Please select a video file first.", "error"

            print(f"🎥 PROCESSING VIDEO FILE: {self.translator.source_lang} → {self.translator.target_lang}")
            self.status = "processing"

            # Extract audio from video
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
                audio_path = audio_tmp.name
            
            command = [
                "ffmpeg", "-i", file_path, "-ac", "1", "-ar", "16000", 
                "-loglevel", "quiet", "-y", audio_path
            ]
            subprocess.run(command, check=True)
            
            print("✅ Video audio extracted")
            
            # Process the extracted audio as WHOLE file
            result, status = self.process_audio_file(audio_path)
            
            # Cleanup
            os.unlink(audio_path)
            
            self.status = "connected"
            return "✅ Video file processed successfully!", "connected"

        except Exception as e:
            logger.error(f"Video file processing error: {e}")
            self.status = "error"
            return f"❌ Error processing video: {str(e)}", "error"

    def get_status_info(self):
        """Get current status information"""
        status_texts = {
            "disconnected": "Disconnected",
            "connected": "Connected", 
            "recording": "Live Translation Active",
            "processing": "Processing File...",
            "error": "Error"
        }
        
        status_colors = {
            "disconnected": "red",
            "connected": "green",
            "recording": "red", 
            "processing": "orange",
            "error": "red"
        }
        
        current_languages = f"{self.current_source_lang.title()} → {self.current_target_lang.title()}"
        
        return status_texts.get(self.status, "Unknown"), status_colors.get(self.status, "gray"), current_languages

# Create global processor instance
processor = AudioProcessor(GlobalTranslator.get_instance())

# Full final UI (drop-in). Assumes `processor` already exists in the module scope.
def create_interface():
    # --- Strong CSS overrides to force colors and shapes ---
    custom_css = r"""
    body { background: linear-gradient(180deg,#efe9ff 0%,#efeef8 100%); font-family: "Inter", "Segoe UI", Roboto, Arial, sans-serif; }
    .gradio-container { padding:28px 36px !important; }
    
    /* Header / status */
    .header-row { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:18px; }
    .status-block { display:flex; align-items:center; gap:12px; }
    .status-indicator { width:14px; height:14px; border-radius:50%; box-shadow:0 0 0 4px rgba(0,0,0,0.04); display:inline-block; }
    .status-disconnected { background:#e74c3c !important; }
    .status-connected { background:#2ecc71 !important; }
    .status-processing { background:#f1c40f !important; }
    .top-status { font-weight:800; font-size:16px; color:#374151; }
    .top-sub { font-size:13px; color:#7b8aa0; }
    .languages-inline { font-weight:800; font-size:18px; color:#111827; margin-left:18px; }
    .languages-inline i { font-weight:400; color:#4b5563; margin:0 8px; }

    .change-lang-btn .gr-button { background:#ffffff !important; border:1px solid rgba(15,76,129,0.12) !important; border-radius:10px !important; padding:10px 14px !important; color:#0f4c81 !important; font-weight:700 !important; box-shadow:0 8px 22px rgba(15,76,129,0.06) !important; }

    .card { background:#ffffff; border-radius:14px; padding:18px; box-shadow:0 8px 30px rgba(18,20,30,0.04); margin-bottom:18px; }
    .section-header { font-size:20px; font-weight:800; display:flex; align-items:center; gap:12px; padding-bottom:10px; border-bottom:1px solid #eef2f6; margin-bottom:14px; }
    .section-sub { color:#6b7280; font-size:14px; margin-bottom:14px; }

    /* Waveform */
    .waveform { width:140px; height:46px; display:flex; gap:8px; align-items:center; justify-content:center; margin:auto; }
    .waveform .bar { width:8px; border-radius:5px; background:#0b63ff; transform-origin:bottom center; animation:bounce 0.9s infinite ease-in-out; opacity:0.95; height:16px; }
    .waveform .bar:nth-child(2) { animation-delay:0.12s; }
    .waveform .bar:nth-child(3) { animation-delay:0.24s; height:28px; }
    .waveform .bar:nth-child(4) { animation-delay:0.36s; }
    .waveform .bar:nth-child(5) { animation-delay:0.48s; }
    @keyframes bounce { 0%{ transform:scaleY(0.55); } 50%{ transform:scaleY(1.6);} 100%{ transform:scaleY(0.55); } }

    /* BUTTON STYLES - strong selectors with !important to override Gradio defaults */
    /* Start = green pill */
    .start-btn .gr-button, button.start-btn, .start-btn button {
        background:#22C55E !important;
        color:#ffffff !important;
        border-radius:999px !important;
        padding:12px 28px !important;
        font-weight:800 !important;
        font-size:16px !important;
        border:none !important;
        box-shadow:0 6px 18px rgba(34,197,94,0.18) !important;
    }
    /* Stop = red pill */
    .stop-btn .gr-button, button.stop-btn, .stop-btn button {
        background:#DC2626 !important;
        color:#ffffff !important;
        border-radius:999px !important;
        padding:12px 28px !important;
        font-weight:800 !important;
        font-size:16px !important;
        border:none !important;
        box-shadow:0 6px 18px rgba(220,38,38,0.18) !important;
    }

    /* Translate audio (green rectangle) */
    .translate-audio-btn .gr-button, .translate-audio-btn button, button.translate-audio-btn {
        background:#22C55E !important;
        color:#ffffff !important;
        border-radius:10px !important;
        padding:10px 18px !important;
        font-weight:700 !important;
        border:none !important;
        box-shadow:0 6px 14px rgba(34,197,94,0.10) !important;
    }

    /* Translate video (blue rectangle) */
    .translate-video-btn .gr-button, .translate-video-btn button, button.translate-video-btn {
        background:#2563EB !important;
        color:#ffffff !important;
        border-radius:10px !important;
        padding:10px 18px !important;
        font-weight:700 !important;
        border:none !important;
        box-shadow:0 6px 14px rgba(37,99,235,0.10) !important;
    }

    /* muted / disabled style */
    .muted-btn .gr-button, .muted-btn button { background:#E5E7EB !important; color:#374151 !important; border-radius:10px !important; }

    /* Files row */
    .files-row { display:flex; gap:18px; }
    .file-card { flex:1; border-radius:12px; padding:20px; border:2px dashed #e7eef8; text-align:center; background:linear-gradient(180deg, rgba(255,255,255,0.98), rgba(255,255,255,0.96)); }
    .file-card h5 { font-size:18px; font-weight:800; margin-top:8px; }
    .file-card p { color:#6b7280; font-size:13px; margin-top:6px; margin-bottom:16px; }

    /* ASR and translation boxes */
    .asr-box { background:#fbfcfe !important; border:1px solid #e6eef6 !important; border-radius:12px !important; padding:16px !important; min-height:90px !important; font-size:16px !important; color:#111827 !important; }
    .trans-box { background:#f0fff3 !important; border:2px solid #22C55E !important; border-radius:12px !important; padding:16px !important; min-height:90px !important; font-size:16px !important; color:#165a2b !important; }

    .history-card { margin-top:12px; max-height:320px; overflow-y:auto; }
    .history-item { background:white; border:1px solid #eef2f6; border-radius:10px; padding:10px; margin-bottom:10px; }
    @media (max-width:900px){ .files-row { flex-direction:column; } .waveform { margin:12px 0; } }

    /* === CLICK ANIMATION FOR TRANSLATE BUTTONS === */
    @keyframes clickPulse {
        0% { transform: scale(1); }
        50% { transform: scale(0.94); }
        100% { transform: scale(1); }
    }

    .translate-audio-btn .gr-button:active,
    .translate-video-btn .gr-button:active {
        animation: clickPulse 0.20s ease-in-out;
    }

    /* Spinner loading class for buttons */
    .button-loading {
        position: relative !important;
        opacity: 0.75 !important;
        pointer-events: none !important;
    }
    .button-loading:after {
        content: "";
        position: absolute;
        right: 10px;
        top: 50%;
        width: 14px;
        height: 14px;
        margin-top: -7px;
        border: 2px solid rgba(255,255,255,0.6);
        border-top-color: white;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); }
    }
    """

    with gr.Blocks(css=custom_css, title="Low Latency Speech Translator") as demo:
        # ---- top status & languages ----
        with gr.Row(elem_classes="header-row"):
            with gr.Row(elem_classes="status-block"):
                status_display = gr.HTML(value=(
                    "<div style='display:flex;align-items:center;gap:12px;'>"
                    "<span class='status-indicator status-disconnected'></span>"
                    "<div>"
                    "<div class='top-status'>Disconnected</div>"
                    "<div class='top-sub'>Not connected</div>"
                    "</div></div>"
                ))
                languages_display = gr.HTML(value=(
                    f"<div class='languages-inline'><strong>{processor.current_source_lang.title()}</strong> <i>→</i> <strong>{processor.current_target_lang.title()}</strong></div>"
                ))
            language_btn = gr.Button("⚙️ Change Languages", elem_classes="change-lang-btn")

        # ---- Live Translation card ----
        with gr.Column(elem_classes="card"):
            gr.HTML('<div class="section-header"><i class="fas fa-microphone"></i>Live Translation</div>')
            gr.HTML('<div class="section-sub">Click start to begin real-time speech translation. Speak clearly into your microphone.</div>')

            # waveform + start/stop column
            with gr.Row():
                waveform_display = gr.HTML("<div class='waveform' id='waveform'><div class='bar'></div><div class='bar'></div><div class='bar'></div><div class='bar'></div><div class='bar'></div></div>")
                # Column for start/stop buttons
                with gr.Column(scale=0):
                    start_btn = gr.Button("Start Live Translation", elem_classes="start-btn")
                    stop_btn = gr.Button("Stop Live Translation", elem_classes="stop-btn", visible=False)

        # ---- File Translation card ----
        with gr.Column(elem_classes="card"):
            gr.HTML('<div class="section-header"><i class="fas fa-file"></i>File Translation</div>')
            with gr.Row(elem_classes="files-row"):
                with gr.Column(elem_classes="file-card"):
                    gr.HTML('<div style="font-size:44px;color:#0b63ff;"><i class="fas fa-file-audio"></i></div>')
                    gr.HTML("<h5>Audio File</h5>")
                    gr.HTML("<p>Upload WAV, MP3, or other audio files</p>")
                    audio_file = gr.Audio(sources=["upload", "microphone"], type="filepath")
                    audio_status = gr.HTML(value="", visible=False)
                    audio_btn = gr.Button("Translate Audio", elem_classes="translate-audio-btn")
                with gr.Column(elem_classes="file-card"):
                    gr.HTML('<div style="font-size:44px;color:#0b7a45;"><i class="fas fa-file-video"></i></div>')
                    gr.HTML("<h5>Video File</h5>")
                    gr.HTML("<p>Upload MP4, AVI, or other video files</p>")
                    video_file = gr.Video()
                    video_status = gr.HTML(value="", visible=False)
                    video_btn = gr.Button("Translate Video", elem_classes="translate-video-btn")

        # ---- Results card (stacked ASR -> Translated -> History) ----
        with gr.Column(elem_classes="card"):
            gr.HTML('<div class="section-header"><i class="fas fa-list"></i>Translation Results</div>')
            # ASR output
            asr_title = gr.HTML("<div style='font-weight:700;color:#6b7280;margin-bottom:6px;'>Detected Speech</div>")
            asr_output = gr.Textbox(value="Your speech will be transcribed here...", interactive=False, elem_classes="asr-box", lines=4)

            # Translated output
            trans_title = gr.HTML("<div style='font-weight:700;color:#16a34a;margin-top:14px;margin-bottom:6px;'>Translated Text</div>")
            trans_output = gr.Textbox(value="Translation will appear here...", interactive=False, elem_classes="trans-box", lines=4)

            # History
            history_html = gr.HTML(value=processor.get_history_display(), visible=True, elem_classes="history-card")

        # ---- Language modal (hidden) ----
        with gr.Column(visible=False) as language_modal:
            source_lang = gr.Dropdown(
                choices=["english","hindi","bengali","tamil","telugu","marathi","gujarati","kannada","malayalam","punjabi","odia","assamese","urdu"],
                value=processor.current_source_lang, label="Source Language"
            )
            target_lang = gr.Dropdown(
                choices=["hindi","english","bengali","tamil","telugu","marathi","gujarati","kannada","malayalam","punjabi","odia","assamese","urdu"],
                value=processor.current_target_lang, label="Target Language"
            )
            save_lang_btn = gr.Button("Save", variant="primary")
            cancel_btn = gr.Button("Cancel", variant="secondary")

        # hidden triggers for events
        status_trigger = gr.Textbox(visible=False)
        audio_trigger = gr.Textbox(visible=False)
        video_trigger = gr.Textbox(visible=False)

        # ---------------- Handlers ----------------
        def open_language_modal():
            return gr.update(visible=True)

        def close_language_modal():
            return gr.update(visible=False)

        def save_languages(src, tgt):
            msg, status = processor.change_languages(src, tgt)
            processor.current_source_lang = src
            processor.current_target_lang = tgt
            processor.status = status
            # return updates: close modal, update languages display, and a dummy trigger
            langs_html = f"<div class='languages-inline'><strong>{src.title()}</strong> <i>→</i> <strong>{tgt.title()}</strong></div>"
            return gr.update(visible=False), langs_html, ""

        # === Updated wrappers that show spinner animation on the clicked button ===
        def process_audio_wrapper(fp):
            # add spinner class to audio_btn (button will show spinner)
            try:
                audio_btn.add_class("button-loading")
            except Exception:
                pass
            try:
                res_msg, stat = processor.process_audio_file(fp)
                processor.status = stat
                if stat and "processing" in str(stat).lower():
                    badge = "<span style='padding:6px 10px;border-radius:999px;background:#f59e0b;color:white;font-weight:700;'>Processing</span>"
                elif stat and str(stat).lower() in ("connected", "done", "success"):
                    badge = "<span style='padding:6px 10px;border-radius:999px;background:#22C55E;color:white;font-weight:700;'>Done</span>"
                else:
                    badge = "<span style='padding:6px 10px;border-radius:999px;background:#ef4444;color:white;font-weight:700;'>Error</span>"
                return f"<div style='display:flex;align-items:center;gap:10px'>{badge}<div style='font-weight:700;color:#374151;margin-left:6px'>{res_msg}</div></div>", ""
            finally:
                try:
                    audio_btn.remove_class("button-loading")
                except Exception:
                    pass

        def process_video_wrapper(fp):
            # add spinner class to video_btn (button will show spinner)
            try:
                video_btn.add_class("button-loading")
            except Exception:
                pass
            try:
                res_msg, stat = processor.process_video_file(fp)
                processor.status = stat
                if stat and "processing" in str(stat).lower():
                    badge = "<span style='padding:6px 10px;border-radius:999px;background:#f59e0b;color:white;font-weight:700;'>Processing</span>"
                elif stat and str(stat).lower() in ("connected", "done", "success"):
                    badge = "<span style='padding:6px 10px;border-radius:999px;background:#22C55E;color:white;font-weight:700;'>Done</span>"
                else:
                    badge = "<span style='padding:6px 10px;border-radius:999px;background:#ef4444;color:white;font-weight:700;'>Error</span>"
                return f"<div style='display:flex;align-items:center;gap:10px'>{badge}<div style='font-weight:700;color:#374151;margin-left:6px'>{res_msg}</div></div>", ""
            finally:
                try:
                    video_btn.remove_class("button-loading")
                except Exception:
                    pass

        def start_live():
            msg, stat = processor.start_processing()
            processor.status = stat
            return "started"

        def stop_live():
            msg, stat = processor.stop_processing()
            processor.status = stat
            return "stopped"

        # UI tick function (runs periodically)
        def ui_tick(trigger=""):
            st = getattr(processor, "status", "disconnected") or "disconnected"
            st_low = str(st).lower()
            # status display
            if "processing" in st_low:
                dot_class = "status-processing"
                main_text = "Processing file..."
                sub_text = "Please wait"
            elif "record" in st_low or st_low == "recording":
                dot_class = "status-connected"
                main_text = "Connected"
                sub_text = "Live translation active"
            elif "connected" in st_low:
                dot_class = "status-connected"
                main_text = "Connected"
                sub_text = "Ready"
            else:
                dot_class = "status-disconnected"
                main_text = "Disconnected"
                sub_text = "Not connected"
            status_html = (
                "<div style='display:flex;align-items:center;gap:12px;'>"
                f"<span class='status-indicator {dot_class}'></span>"
                "<div>"
                f"<div class='top-status'>{main_text}</div>"
                f"<div class='top-sub'>{sub_text}</div>"
                "</div></div>"
            )

            # languages html
            langs_html = f"<div class='languages-inline'><strong>{processor.current_source_lang.title()}</strong> <i>→</i> <strong>{processor.current_target_lang.title()}</strong></div>"

            # waveform: show full when recording, muted otherwise
            if "record" in st_low or st_low == "recording":
                waveform_html = "<div class='waveform'><div class='bar'></div><div class='bar'></div><div class='bar'></div><div class='bar'></div><div class='bar'></div></div>"
            else:
                waveform_html = "<div class='waveform' style='opacity:0.25;'><div class='bar'></div><div class='bar'></div><div class='bar'></div><div class='bar'></div><div class='bar'></div></div>"

            # ASR & translation
            orig = getattr(processor, "current_transcription", "") or "Your speech will be transcribed here..."
            trans = getattr(processor, "current_translation", "") or "Translation will appear here..."
            # history
            try:
                history_val = processor.get_history_display()
            except Exception:
                history_val = "<div>No translations yet. Start speaking or upload a file!</div>"

            # button visibility + class updates
            if "record" in st_low or st_low == "recording":
                start_update = gr.update(visible=False)
                stop_update = gr.update(visible=True, value="Stop Live Translation")
            else:
                start_update = gr.update(visible=True, value="Start Live Translation")
                stop_update = gr.update(visible=False)

            return (
                gr.update(value=status_html),
                gr.update(value=orig),
                gr.update(value=trans),
                gr.update(value=history_val, visible=True),
                gr.update(value=langs_html),
                gr.update(value=waveform_html),
                start_update,
                stop_update
            )

        # ---- Bind events ----
        language_btn.click(open_language_modal, outputs=language_modal)
        cancel_btn.click(close_language_modal, outputs=language_modal)
        save_lang_btn.click(save_languages, inputs=[source_lang, target_lang], outputs=[language_modal, languages_display, status_trigger])

        audio_btn.click(process_audio_wrapper, inputs=[audio_file], outputs=[audio_status, audio_trigger])
        video_btn.click(process_video_wrapper, inputs=[video_file], outputs=[video_status, video_trigger])

        start_btn.click(start_live, outputs=[status_trigger])
        stop_btn.click(stop_live, outputs=[status_trigger])

        # Timer: 0.5s tick (works in your environment)
        update_timer = gr.Timer(0.5)
        update_timer.tick(
            fn=ui_tick,
            outputs=[
                status_display,   # status html
                asr_output,       # asr textbox
                trans_output,     # translation textbox
                history_html,     # history html
                languages_display,# languages html
                waveform_display, # waveform HTML
                start_btn,        # start button update
                stop_btn          # stop button update
            ]
        )

        return demo
    
# Create and launch the interface
if __name__ == "__main__":
    print("🚀 GRADIO TRANSLATOR STARTING...")
    
    demo = create_interface()
    
    # For Hugging Face Spaces, use this:
    # demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    
    # For local development, use this:
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=True
    )