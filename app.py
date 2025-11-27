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
import librosa

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
os.chdir(current_directory)
print(f"📁 Server running from: {current_directory}")

from model import LowLatencyTranslator, warm_up_models

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

global_translator = GlobalTranslator.get_instance()

class AudioProcessor:
    def __init__(self, translator: LowLatencyTranslator):
        self.translator = translator
        # These will be reset per session
        self.status = "connected"
        self.current_source_lang = "english"
        self.current_target_lang = "hindi"
        self.last_tts_file = None

    def get_session_state(self):
        """Get or create session state for current user"""
        return {
            'current_transcription': "",
            'current_translation': "", 
            'last_update_time': 0,
            'translation_history': [],
            'max_history_items': 10
        }

    def cleanup_previous_tts(self):
        """Clean up only the previous TTS file"""
        if self.last_tts_file and os.path.exists(self.last_tts_file):
            try:
                os.unlink(self.last_tts_file)
                print(f"🧹 Cleaned up previous TTS file: {self.last_tts_file}")
            except Exception as e:
                print(f"⚠️ Could not cleanup TTS file: {e}")
            finally:
                self.last_tts_file = None

    def change_languages(self, source_lang: str, target_lang: str):
        """Change languages"""
        try:
            self.translator.source_lang = source_lang
            self.translator.target_lang = target_lang
            self.current_source_lang = source_lang
            self.current_target_lang = target_lang
            
            print(f"🌐 LANGUAGES CHANGED: {source_lang} → {target_lang}")
            
            global_translator = GlobalTranslator.get_instance()
            global_translator.source_lang = source_lang
            global_translator.target_lang = target_lang
            
            return f"✅ Languages changed to {source_lang.title()} → {target_lang.title()}", "connected"
            
        except Exception as e:
            print(f"❌ Language change error: {e}")
            return f"❌ Error changing languages: {str(e)}", "error"

    def add_to_history(self, state, original, translated):
        """Add translation to history"""
        history_item = {
            'original': original,
            'translated': translated,
            'timestamp': time.strftime('%H:%M:%S'),
            'source_lang': self.current_source_lang,
            'target_lang': self.current_target_lang
        }
        state['translation_history'].insert(0, history_item)
        
        if len(state['translation_history']) > state['max_history_items']:
            state['translation_history'].pop()

    def get_history_display(self, state):
        """Get formatted history for display"""
        if not state['translation_history']:
            return "No translations yet. Upload an audio or video file!"
        
        history_html = "<div style='max-height: 300px; overflow-y: auto;'>"
        for item in state['translation_history']:
            history_html += f"""
            <div style="background: white; border: 1px solid #e9ecef; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem; font-size: 0.9rem;">
                <div style="color: #495057 !important ; margin-bottom: 0.25rem;"><strong>Original ({item['source_lang']}):</strong> {item['original']}</div>
                <div style="color: #28a745 !important; font-weight: 500; margin-bottom: 0.25rem;"><strong>Translated ({item['target_lang']}):</strong> {item['translated']}</div>
                <small style="color: #6c757d;">{item['timestamp']}</small>
            </div>
            """
        history_html += "</div>"
        return history_html

    def process_audio_file(self, state, file_path: str):
        try:
            if not file_path:
                return state, "❌ Please select an audio file first.", "error", None
    
            print(f"🎵 PROCESSING AUDIO FILE: {self.translator.source_lang} → {self.translator.target_lang}")
            self.status = "processing"
    
            # Clean up previous TTS
            self.cleanup_previous_tts()
    
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.translator.sample_rate)
            duration = len(audio) / sr
            print(f"✅ Loaded audio file ({duration:.1f}s)")
            
            # ✅ FIX: Explicitly set source language for ASR
            if self.current_source_lang.lower() == "hindi":
                self.translator.source_lang = "hindi"
                forced_lang = "hindi"
            else:
                self.translator.source_lang = "english" 
                forced_lang = "english"
                
            print(f"🔊 FORCING ASR LANGUAGE: {forced_lang}")
            
            # Perform ASR
            transcription = self.translator.speech_to_text(audio)
            
            tts_file = None
            
            if transcription and transcription.strip():
                print(f"🎯 [{self.translator.source_lang}] File ASR: {transcription}")
                
                # Check if we got proper Hindi script
                if forced_lang == "hindi":
                    # Basic check for Devanagari script
                    devanagari_range = range(0x0900, 0x097F)
                    has_devanagari = any(ord(char) in devanagari_range for char in transcription)
                    if not has_devanagari:
                        print("⚠️ WARNING: ASR output doesn't contain Hindi Devanagari script!")
                
                # Translation
                translated = self.translator.translator.translate_text(
                    transcription, 
                    self.translator.source_lang, 
                    self.translator.target_lang
                )
                print(f"🌐 [{self.translator.target_lang}] File Translation: {translated}")
                
                # Update session state
                state['current_transcription'] = transcription
                state['current_translation'] = translated
                state['last_update_time'] = time.time()
                
                self.add_to_history(state, transcription, translated)
                
                # TTS with cleanup
                tts_file = self.translator.tts.text_to_speech(translated, self.translator.target_lang)
                self.last_tts_file = tts_file
                
                # Ensure TTS file is accessible
                if tts_file and os.path.exists(tts_file):
                    print(f"🔊 TTS file generated: {tts_file}")
                    # Convert to WAV if needed
                    if not tts_file.endswith('.wav'):
                        wav_file = tts_file.replace('.mp3', '.wav')
                        command = [
                            "ffmpeg", "-i", tts_file, "-ac", "1", "-ar", "22050",
                            "-loglevel", "quiet", "-y", wav_file
                        ]
                        subprocess.run(command, check=True)
                        if os.path.exists(wav_file):
                            tts_file = wav_file
                            self.last_tts_file = wav_file
                else:
                    print("❌ TTS file not generated properly")
                    tts_file = None
            
            print("✅ Audio file processing completed")
            self.status = "connected"
            return state, "✅ Audio file processed successfully!", "connected", tts_file
    
        except Exception as e:
            logger.error(f"Audio file processing error: {e}")
            self.status = "error"
            return state, f"❌ Error processing audio: {str(e)}", "error", None

    def process_video_file(self, state, file_path: str):
        """Process video file - WORKS ON HUGGING FACE (ffmpeg available)"""
        try:
            if not file_path:
                return state, "❌ Please select a video file first.", "error", None

            print(f"🎥 PROCESSING VIDEO FILE: {self.translator.source_lang} → {self.translator.target_lang}")
            self.status = "processing"

            # Clean up previous TTS
            self.cleanup_previous_tts()

            # Extract audio from video
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
                audio_path = audio_tmp.name
            
            command = [
                "ffmpeg", "-i", file_path, "-ac", "1", "-ar", "16000", 
                "-loglevel", "quiet", "-y", audio_path
            ]
            subprocess.run(command, check=True)
            
            print("✅ Video audio extracted")
            
            # Process the extracted audio and get TTS file
            state, res_msg, status, tts_file = self.process_audio_file(state, audio_path)
            
            # Cleanup
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            
            self.status = "connected"
            return state, "✅ Video file processed successfully!", "connected", tts_file

        except Exception as e:
            logger.error(f"Video file processing error: {e}")
            self.status = "error"
            return state, f"❌ Error processing video: {str(e)}", "error", None

# Create global processor instance
processor = AudioProcessor(GlobalTranslator.get_instance())

def create_interface():
    custom_css = r"""
    /* your existing CSS remains the same */
    body { background: linear-gradient(180deg,#efe9ff 0%,#efeef8 100%); font-family: "Inter", "Segoe UI", Roboto, Arial, sans-serif; }
    .gradio-container { padding:28px 36px !important; }
    
    /* Header / status */
    .header-row { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:18px; }
    .status-block { display:flex; align-items:center; gap:12px; }
    .status-indicator { width:14px; height:14px; border-radius:50%; box-shadow:0 0 0 4px rgba(0,0,0,0.04); display:inline-block; }
    .status-disconnected { background:#e74c3c !important; }
    .status-connected { background:#2ecc71 !important; }
    .status-processing { background:#f1c40f !important; }
    .top-status { font-weight:800; font-size:16px; color:#1f2937 !important; }
    .top-sub { font-size:13px; color:#4b5563 !important; }
    .languages-inline { 
        font-weight:800 !important;
        font-size:18px;
        color:#000000 !important;
        margin-left:18px;
        opacity: 1 !important;
    }
    
    .languages-inline * {
        color:#000000 !important;
        opacity: 1 !important;
    }
    
    .languages-inline i { 
        font-weight:400 !important;
        color:#4b5563 !important;
        opacity: 1 !important;
    }


    .change-lang-btn .gr-button { background:#ffffff !important; border:1px solid rgba(15,76,129,0.12) !important; border-radius:10px !important; padding:10px 14px !important; color:#0f4c81 !important; font-weight:700 !important; box-shadow:0 8px 22px rgba(15,76,129,0.06) !important; }

    .card { background:#ffffff; border-radius:14px; padding:18px; box-shadow:0 8px 30px rgba(18,20,30,0.04); margin-bottom:18px; }
    .section-header { font-size:20px; font-weight:800; display:flex; align-items:center; gap:12px; padding-bottom:10px; border-bottom:1px solid #eef2f6; margin-bottom:14px; color: #1f2937 !important; }
    .section-sub { color:#4b5563 !important; font-size:14px; margin-bottom:14px; }

    /* BUTTON STYLES */
    .translate-audio-btn .gr-button, .translate-audio-btn button, button.translate-audio-btn {
        background:#22C55E !important;
        color:#ffffff !important;
        border-radius:10px !important;
        padding:10px 18px !important;
        font-weight:700 !important;
        border:none !important;
        box-shadow:0 6px 14px rgba(34,197,94,0.10) !important;
    }

    .translate-video-btn .gr-button, .translate-video-btn button, button.translate-video-btn {
        background:#2563EB !important;
        color:#ffffff !important;
        border-radius:10px !important;
        padding:10px 18px !important;
        font-weight:700 !important;
        border:none !important;
        box-shadow:0 6px 14px rgba(37,99,235,0.10) !important;
    }

    .muted-btn .gr-button, .muted-btn button { background:#E5E7EB !important; color:#374151 !important; border-radius:10px !important; }

    .files-row { display:flex; gap:18px; }
    .file-card { flex:1; border-radius:12px; padding:20px; border:2px dashed #e7eef8; text-align:center; background:linear-gradient(180deg, rgba(255,255,255,0.98), rgba(255,255,255,0.96)); }
    .file-card h5 { font-size:18px; font-weight:800; margin-top:8px; color: #1f2937 !important; }
    .file-card p { color:#4b5563 !important; font-size:13px; margin-top:6px; margin-bottom:16px; }

    .asr-box { background:#fbfcfe !important; border:1px solid #e6eef6 !important; border-radius:12px !important; padding:16px !important; min-height:90px !important; font-size:16px !important; color:#111827 !important; }
    .trans-box { background:#f0fff3 !important; border:2px solid #22C55E !important; border-radius:12px !important; padding:16px !important; min-height:90px !important; font-size:16px !important; color:#165a2b !important; }

    .history-card { margin-top:12px; max-height:320px; overflow-y:auto; }
    .history-item { background:white; border:1px solid #eef2f6; border-radius:10px; padding:10px; margin-bottom:10px; }
    @media (max-width:900px){ .files-row { flex-direction:column; } }

    @keyframes clickPulse {
        0% { transform: scale(1); }
        50% { transform: scale(0.94); }
        100% { transform: scale(1); }
    }

    .translate-audio-btn .gr-button:active,
    .translate-video-btn .gr-button:active {
        animation: clickPulse 0.20s ease-in-out;
    }

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

    with gr.Blocks(css=custom_css, title="File Speech Translator") as demo:
        # Use Gradio's session state for user-specific data
        session_state = gr.State(processor.get_session_state())
        
        gr.HTML(
            """
            <div style="
                width: 100%; 
                background: white; 
                padding: 32px 20px; 
                border-radius: 18px; 
                box-shadow: 0 8px 32px rgba(0,0,0,0.06);
                margin-bottom: 24px;
                text-align: center;
            ">
                <div style="
                    display: inline-flex; 
                    align-items: center; 
                    gap: 14px; 
                    font-size: 36px; 
                    font-weight: 700; 
                    color: #1f2937;
                ">
                    <span style="
                        background: #f3f4f6;
                        padding: 10px 14px;
                        border-radius: 12px;
                        font-size: 28px;
                    ">🈺</span>
                    Low Latency Translator
                </div>
                
                <div style="
                    font-size: 18px; 
                    color: #4b5563; 
                    margin-top: 10px;
                ">
                    Real-time speech-to-text translation with multilingual support
                </div>
            </div>
            """
        )

        # ---- top status & languages ----
        with gr.Row(elem_classes="header-row"):
            with gr.Row(elem_classes="status-block"):
                status_display = gr.HTML(value=(
                    "<div style='display:flex;align-items:center;gap:12px;'>"
                    "<span class='status-indicator status-connected'></span>"
                    "<div>"
                    "<div class='top-status'>Connected</div>"
                    "<div class='top-sub'>Ready for file translation</div>"
                    "</div></div>"
                ))
                languages_display = gr.HTML(value=(
                    f"<div class='languages-inline'><strong>{processor.current_source_lang.title()}</strong> <i>→</i> <strong>{processor.current_target_lang.title()}</strong></div>"
                ))
            language_btn = gr.Button("⚙️ Change Languages", elem_classes="change-lang-btn")

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

        # ---- Results card ----
        with gr.Column(elem_classes="card"):
            gr.HTML('<div class="section-header"><i class="fas fa-list"></i>Translation Results</div>')
            asr_title = gr.HTML("<div style='font-weight:700;color:#1f2937 !important;margin-bottom:6px;'>Detected Speech</div>")
            asr_output = gr.Textbox(value="Your speech will be transcribed here...", interactive=False, elem_classes="asr-box", lines=4)

            trans_title = gr.HTML("<div style='font-weight:700;color:#1f2937 !important;margin-top:14px;margin-bottom:6px;'>Translated Text</div>")
            trans_output = gr.Textbox(value="Translation will appear here...", interactive=False, elem_classes="trans-box", lines=4)
            
            tts_output = gr.Audio(
                label="Translated Speech Output",
                visible=False,
                elem_id="tts-player"
            )
            
            history_html = gr.HTML(value="No translations yet. Upload an audio or video file!", visible=True, elem_classes="history-card")

        # ---- Language modal ----
        with gr.Column(visible=False) as language_modal:
            source_lang = gr.Dropdown(
                choices=["english","hindi"],
                value=processor.current_source_lang, label="Source Language"
            )
            target_lang = gr.Dropdown(
                choices=["hindi","english","bengali","tamil","telugu","marathi","gujarati","kannada","malayalam","punjabi","odia","assamese","urdu"],
                value=processor.current_target_lang, label="Target Language"
            )
            save_lang_btn = gr.Button("Save", variant="primary")
            cancel_btn = gr.Button("Cancel", variant="secondary")

        # hidden triggers
        status_trigger = gr.Textbox(visible=False)

        # ---------------- Handlers ----------------
        def open_language_modal():
            return gr.update(visible=True)

        def close_language_modal():
            return gr.update(visible=False)

        def save_languages(src, tgt):
            src = src.lower()
            if src not in ["english", "hindi"]:
                src = "english"
        
            msg, status = processor.change_languages(src, tgt)
            processor.current_source_lang = src
            processor.current_target_lang = tgt
            processor.status = status
            langs_html = f"<div class='languages-inline'><strong>{src.title()}</strong> <i>→</i> <strong>{tgt.title()}</strong></div>"
            return gr.update(visible=False), langs_html, ""

        def process_audio_wrapper(state, fp):
            try:
                audio_btn.add_class("button-loading")
            except Exception:
                pass
            try:
                state, res_msg, stat, tts_file = processor.process_audio_file(state, fp)
                processor.status = stat
                
                if stat and "processing" in str(stat).lower():
                    badge = "<span style='padding:6px 10px;border-radius:999px;background:#f59e0b;color:white;font-weight:700;'>Processing</span>"
                elif stat and str(stat).lower() in ("connected", "done", "success"):
                    badge = "<span style='padding:6px 10px;border-radius:999px;background:#22C55E;color:white;font-weight:700;'>Done</span>"
                else:
                    badge = "<span style='padding:6px 10px;border-radius:999px;background:#ef4444;color:white;font-weight:700;'>Error</span>"
                
                status_html = f"<div style='display:flex;align-items:center;gap:10px'>{badge}<div style='font-weight:700;color:#374151;margin-left:6px'>{res_msg}</div></div>"
                
                # Get texts from session state
                orig_text = state['current_transcription'] or "Your speech will be transcribed here..."
                trans_text = state['current_translation'] or "Translation will appear here..."
                
                return (
                    state,
                    status_html, 
                    orig_text, 
                    trans_text, 
                    gr.update(value=tts_file, visible=tts_file is not None),
                    processor.get_history_display(state)
                )
            finally:
                try:
                    audio_btn.remove_class("button-loading")
                except Exception:
                    pass

        def process_video_wrapper(state, fp):
            try:
                video_btn.add_class("button-loading")
            except Exception:
                pass
            try:
                state, res_msg, stat, tts_file = processor.process_video_file(state, fp)
                processor.status = stat
                
                if stat and "processing" in str(stat).lower():
                    badge = "<span style='padding:6px 10px;border-radius:999px;background:#f59e0b;color:white;font-weight:700;'>Processing</span>"
                elif stat and str(stat).lower() in ("connected", "done", "success"):
                    badge = "<span style='padding:6px 10px;border-radius:999px;background:#22C55E;color:white;font-weight:700;'>Done</span>"
                else:
                    badge = "<span style='padding:6px 10px;border-radius:999px;background:#ef4444;color:white;font-weight:700;'>Error</span>"
                
                status_html = f"<div style='display:flex;align-items:center;gap:10px'>{badge}<div style='font-weight:700;color:#374151;margin-left:6px'>{res_msg}</div></div>"
                
                # Get texts from session state
                orig_text = state['current_transcription'] or "Your speech will be transcribed here..."
                trans_text = state['current_translation'] or "Translation will appear here..."
                
                return (
                    state,
                    status_html, 
                    orig_text, 
                    trans_text, 
                    gr.update(value=tts_file, visible=tts_file is not None),
                    processor.get_history_display(state)
                )
            finally:
                try:
                    video_btn.remove_class("button-loading")
                except Exception:
                    pass

        # UI tick function
        def ui_tick(state, trigger=""):
            st = getattr(processor, "status", "connected") or "connected"
            st_low = str(st).lower()
            
            if "processing" in st_low:
                dot_class = "status-processing"
                main_text = "Processing file..."
                sub_text = "Please wait"
            elif "connected" in st_low:
                dot_class = "status-connected"
                main_text = "Connected"
                sub_text = "Ready for file translation"
            else:
                dot_class = "status-disconnected"
                main_text = "Disconnected"
                sub_text = "Disconnected"
                
            status_html = (
                "<div style='display:flex;align-items:center;gap:12px;'>"
                f"<span class='status-indicator {dot_class}'></span>"
                "<div>"
                f"<div class='top-status'>{main_text}</div>"
                f"<div class='top-sub'>{sub_text}</div>"
                "</div></div>"
            )

            langs_html = f"<div class='languages-inline'><strong>{processor.current_source_lang.title()}</strong> <i>→</i> <strong>{processor.current_target_lang.title()}</strong></div>"

            # Get texts from session state
            orig_text = state['current_transcription'] or "Your speech will be transcribed here..."
            trans_text = state['current_translation'] or "Translation will appear here..."
            
            history_val = processor.get_history_display(state)

            return (
                state,
                gr.update(value=status_html),
                gr.update(value=orig_text),
                gr.update(value=trans_text),
                gr.update(value=history_val, visible=True),
                gr.update(value=langs_html)
            )

        # ---- Bind events ----
        language_btn.click(open_language_modal, outputs=language_modal)
        cancel_btn.click(close_language_modal, outputs=language_modal)
        save_lang_btn.click(save_languages, inputs=[source_lang, target_lang], outputs=[language_modal, languages_display, status_trigger])

        # ✅ UPDATED: Connect file processing with proper outputs including session state
        audio_btn.click(
            process_audio_wrapper, 
            inputs=[session_state, audio_file], 
            outputs=[session_state, audio_status, asr_output, trans_output, tts_output, history_html]
        )
        
        video_btn.click(
            process_video_wrapper, 
            inputs=[session_state, video_file], 
            outputs=[session_state, video_status, asr_output, trans_output, tts_output, history_html]
        )

        # Timer for UI updates
        update_timer = gr.Timer(1.0)
        update_timer.tick(
            fn=ui_tick,
            inputs=[session_state],
            outputs=[
                session_state,
                status_display,
                asr_output,
                trans_output,
                history_html,
                languages_display
            ]
        )

        return demo
    
# Create and launch the interface
if __name__ == "__main__":
    print("🚀 FILE TRANSLATOR STARTING...")
    
    demo = create_interface()
    
    # For Hugging Face Spaces
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
