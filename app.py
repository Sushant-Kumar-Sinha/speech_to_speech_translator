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
            return "<div style='text-align: center; color: #666; padding: 40px; font-size: 16px;'>No translations yet. Upload a file to get started! 🚀</div>"
        
        history_html = "<div style='max-height: 400px; overflow-y: auto; padding-right: 10px;'>"
        for i, item in enumerate(state['translation_history']):
            history_html += f"""
            <div style="
                background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 20px;
                margin-bottom: 16px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.04);
                transition: all 0.3s ease;
            " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px rgba(0,0,0,0.1)';" 
            onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 10px rgba(0,0,0,0.04)';">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
                    <div style="font-weight: 700; color: #1e293b; font-size: 14px; background: #f1f5f9; padding: 4px 12px; border-radius: 20px;">
                        {item['source_lang'].title()} → {item['target_lang'].title()}
                    </div>
                    <div style="color: #64748b; font-size: 12px; font-weight: 600;">{item['timestamp']}</div>
                </div>
                <div style="color: #374151; margin-bottom: 12px; font-size: 15px; line-height: 1.5;">
                    <strong style="color: #1e293b;">Original:</strong> {item['original']}
                </div>
                <div style="color: #059669; font-size: 15px; line-height: 1.5; font-weight: 500;">
                    <strong style="color: #047857;">Translated:</strong> {item['translated']}
                </div>
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
            
            # ✅ FIX: Use the actual selected source language
            self.translator.source_lang = self.current_source_lang
            forced_lang = self.current_source_lang
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
    /* Modern Gradient Background */
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
        min-height: 100vh;
    }
    
    .gradio-container {
        width: 83% !important;
        margin: 0 auto !important;
        padding: 20px !important;
        background: transparent !important;
    }
    
    /* Main Container */
    .main-container {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1) !important;
        padding: 32px !important;
        margin-bottom: 24px !important;
    }
    
    /* Header Styles */
    .header-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 20px !important;
        padding: 30px 25px !important;
        margin-bottom: 24px !important;
        color: white !important;
        text-align: center !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .header-gradient::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        animation: float 20s infinite linear;
    }

    /* Language Panel Styles - FIXED POSITIONING */
    .language-panel {
        background: rgba(255, 255, 255, 0.98) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
        margin-top: 16px !important;
        margin-bottom: 24px !important;
        transition: all 0.3s ease !important;
    }
    
    @keyframes float {
        0% { transform: translate(0, 0); }
        100% { transform: translate(-60px, -60px); }
    }
    
    .main-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 12px !important;
        background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        position: relative !important;
        z-index: 2 !important;
    }
    
    .subtitle {
        font-size: 1.2rem !important;
        opacity: 0.9 !important;
        font-weight: 400 !important;
        position: relative !important;
        z-index: 2 !important;
    }
    
    /* Status Bar - IMPROVED VISIBILITY */
    .status-bar {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(15px) !important;
        border-radius: 16px !important;
        padding: 20px 24px !important;
        margin-bottom: 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        position: relative !important;
        z-index: 50 !important;
    }
    
    .status-content {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        width: 100% !important;
    }
    
    .status-left {
        display: flex !important;
        align-items: center !important;
        gap: 16px !important;
    }
    
    .status-right {
        display: flex !important;
        align-items: center !important;
        gap: 16px !important;
    }
    
    .status-indicator {
        width: 14px !important;
        height: 14px !important;
        border-radius: 50% !important;
        display: inline-block !important;
        animation: pulse 2s infinite !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .status-connected { background: #10b981 !important; box-shadow: 0 0 15px #10b981 !important; }
    .status-processing { background: #f59e0b !important; box-shadow: 0 0 15px #f59e0b !important; }
    .status-error { background: #ef4444 !important; box-shadow: 0 0 15px #ef4444 !important; }
    
    .status-text {
        font-weight: 700 !important;
        color: #000000 !important; /* PURE BLACK */
        font-size: 16px !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    }
    
    .language-display {
        font-weight: 700 !important;
        color: #000000 !important; /* PURE BLACK */
        font-size: 16px !important;
        background: rgba(255,255,255,0.2) !important;
        padding: 8px 16px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    }
    
    /* Card Styles */
    .card-modern {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        padding: 20px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        z-index: 10 !important;
    }
    
    .card-modern:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12) !important;
    }
    
    .section-header {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
        margin-bottom: 16px !important;
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
    }
    
    .section-icon {
        font-size: 1.8rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    
    /* COMPACT File Upload Cards */
    .file-upload-grid {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 20px !important;
        margin-bottom: 20px !important;
    }
    /* Improved Radio Button Styling - WORKING VERSION */
    .gradio-radio {
        --ring-color: transparent !important;
        --background-fill-primary: transparent !important;
        --block-background-fill: transparent !important;
        --block-border-color: transparent !important;
    }

    .gradio-radio > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        margin: 8px 0 !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }

    .gradio-radio > div:hover {
        background: rgba(102, 126, 234, 0.1) !important;
        border-color: #667eea !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15) !important;
    }

    .gradio-radio > div.selected {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-color: #667eea !important;
        color: white !important;
        font-weight: 600 !important;
    }

    .gradio-radio label {
        font-weight: 500 !important;
        font-size: 15px !important;
        cursor: pointer !important;
    }

    /* Make language panel more compact */
    .language-panel {
        max-height: 500px !important;
        overflow-y: auto !important;
    }

    .language-panel .gradio-row {
        gap: 20px !important;
    }
    
    @media (max-width: 768px) {
        .file-upload-grid {
            grid-template-columns: 1fr !important;
        }
    }
    
    .upload-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        text-align: center !important;
        border: 2px dashed rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
        min-height: auto !important;
    }
    
    .upload-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.6), transparent);
        transition: left 0.7s;
    }
    
    .upload-card:hover::before {
        left: 100%;
    }
    
    .upload-card:hover {
        border-color: #667eea !important;
        transform: translateY(-4px) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.15) !important;
    }
    
    .upload-icon {
        font-size: 2.5rem !important;
        margin-bottom: 8px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    
    /* Compact upload card content */
    .upload-card h3 {
        margin-bottom: 8px !important;
        color: #1f2937 !important;
        font-size: 18px !important;
        font-weight: 700 !important;
    }
    
    .upload-card p {
        color: #6b7280 !important;
        margin-bottom: 16px !important;
        font-size: 14px !important;
        line-height: 1.4 !important;
    }
    
    /* Button Styles - IMPROVED WITH CLICK EFFECTS */
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        color: white !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        font-size: 16px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .btn-primary::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .btn-primary:hover::before {
        left: 100%;
    }
    
    .btn-primary:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6) !important;
    }
    
    .btn-primary:active {
        transform: translateY(-1px) scale(0.98) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.8) !important;
        transition: all 0.1s ease !important;
    }
    
    .btn-secondary {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid #667eea !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        color: #667eea !important;
        transition: all 0.3s ease !important;
        font-size: 15px !important;
    }
    
    .btn-secondary:hover {
        background: #667eea !important;
        color: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Result Boxes - IMPROVED VISIBILITY */
    .result-box {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        margin-bottom: 16px !important;
        border-left: 5px solid #667eea !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06) !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    .result-title {
        font-weight: 700 !important;
        color: #1f2937 !important;
        margin-bottom: 10px !important;
        font-size: 16px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    
    .result-content {
        color: #1f2937 !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        font-weight: 500 !important;
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: white;
        animation: spin 1s ease-in-out infinite;
        margin-right: 10px;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Progress Bar */
    .progress-bar {
        height: 6px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 3px;
        overflow: hidden;
        margin: 15px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 3px;
        animation: progress 2s ease-in-out infinite;
    }
    
    @keyframes progress {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Status Messages */
    .status-message {
        padding: 12px 20px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    
    .status-success {
        background: #dcfce7 !important;
        color: #166534 !important;
        border: 1px solid #bbf7d0 !important;
    }
    
    .status-processing {
        background: #fef3c7 !important;
        color: #92400e !important;
        border: 1px solid #fde68a !important;
    }
    
    .status-error {
        background: #fee2e2 !important;
        color: #991b1b !important;
        border: 1px solid #fecaca !important;
    }

    /* Audio and Video component styling for compact layout */
    .gradio-audio, .gradio-video {
        min-height: 80px !important;
        margin-bottom: 12px !important;
    }

    .gradio-audio .wrap, .gradio-video .wrap {
        min-height: 80px !important;
    }
    """

    with gr.Blocks(css=custom_css, title="Speech Translator Pro") as demo:
        # Use Gradio's session state for user-specific data
        session_state = gr.State(processor.get_session_state())
        
        # Main container
        with gr.Column(elem_classes="main-container"):
            # Header with gradient
            with gr.Column(elem_classes="header-gradient"):
                gr.HTML("""
                    <div style="position: relative; z-index: 2;">
                        <h1 class="main-title">🎙️ Speech Translator Pro</h1>
                        <p class="subtitle">Real-time multilingual speech translation with AI-powered accuracy</p>
                    </div>
                """)
            
            # Status Bar - IMPROVED VISIBILITY
            with gr.Row(elem_classes="status-bar"):
                with gr.Row(elem_classes="status-content"):
                    with gr.Row(elem_classes="status-left"):
                        status_display = gr.HTML(value=(
                            "<div style='display: flex; align-items: center; gap: 12px;'>"
                            "<span class='status-indicator status-connected'></span>"
                            "<span class='status-text'>System Ready</span>"
                            "</div>"
                        ))
                    
                    with gr.Row(elem_classes="status-right"):
                        languages_display = gr.HTML(value=(
                            f"<div class='language-display'>"
                            f"<span style='opacity: 0.9;'>🌐 </span>"
                            f"{processor.current_source_lang.title()} → {processor.current_target_lang.title()}"
                            f"</div>"
                        ))
                        language_btn = gr.Button("⚙️ Change Languages", elem_classes="btn-secondary")
        
            # Language Panel (initially hidden) - NOW BELOW THE STATUS BAR
            with gr.Column(visible=False, elem_classes="language-panel") as language_panel:
                gr.HTML("<h3 style='margin-bottom: 16px; color: #1f2937; text-align: center; font-size: 1.3rem;'>🌐 Language Settings</h3>")
                
                with gr.Row():
                    with gr.Column():
                        # Clear source language section
                        gr.HTML("""
                            <div style="text-align: center; margin-bottom: 16px;">
                                <div style="font-size: 18px; font-weight: 700; color: #1f2937; margin-bottom: 8px;">🎤 Source Language</div>
                                <div style="font-size: 14px; color: #6b7280;">Select the language of your audio/video files</div>
                            </div>
                        """)
                        source_lang = gr.Radio(
                            choices=["english", "hindi", "bengali", "tamil", "telugu", "marathi", 
                                "gujarati", "kannada", "malayalam", "punjabi", "urdu"],
                            value=processor.current_source_lang, 
                            label="",  # Remove label since we have header
                            show_label=False
                        )
                    with gr.Column():
                        # Clear target language section  
                        gr.HTML("""
                            <div style="text-align: center; margin-bottom: 16px;">
                                <div style="font-size: 18px; font-weight: 700; color: #1f2937; margin-bottom: 8px;">🌍 Target Language</div>
                                <div style="font-size: 14px; color: #6b7280;">Select the language for translation output</div>
                            </div>
                        """)
                        target_lang = gr.Radio(
                            choices=["hindi", "english", "bengali", "tamil", "telugu", "marathi", 
                                "gujarati", "kannada", "malayalam", "punjabi", "urdu"],
                            value=processor.current_target_lang, 
                            label="",  # Remove label since we have header
                            show_label=False
                        )
                
                with gr.Row():
                    save_lang_btn = gr.Button("💾 Save Settings", variant="primary", elem_classes="btn-primary", size="lg")
                    cancel_btn = gr.Button("❌ Cancel", variant="secondary", elem_classes="btn-secondary", size="lg")
            # File Upload Section - MORE COMPACT
            with gr.Column(elem_classes="card-modern"):
                gr.HTML("""
                    <div class="section-header">
                        <span class="section-icon">📁</span>
                        Upload Media Files
                    </div>
                    <p style="color: #6b7280; margin-bottom: 16px; font-size: 15px; line-height: 1.5;">
                        Upload audio or video files for instant translation. Supports multiple formats including WAV, MP3, MP4, and more.
                    </p>
                """)
                
                with gr.Row(elem_classes="file-upload-grid"):
                    # Audio Upload Card - MORE COMPACT
                    with gr.Column(elem_classes="upload-card"):
                        gr.HTML('<div class="upload-icon">🎵</div>')
                        gr.HTML("<h3>Audio File</h3>")
                        gr.HTML("<p>WAV, MP3, FLAC, and other audio formats</p>")
                        audio_file = gr.Audio(
                            sources=["upload", "microphone"], 
                            type="filepath",
                            label="",
                            show_label=False
                        )
                        audio_btn = gr.Button("🚀 Translate Audio", elem_classes="btn-primary", size="lg")
                        audio_status = gr.HTML(visible=False)
                    
                    # Video Upload Card - MORE COMPACT
                    with gr.Column(elem_classes="upload-card"):
                        gr.HTML('<div class="upload-icon">🎥</div>')
                        gr.HTML("<h3>Video File</h3>")
                        gr.HTML("<p>MP4, AVI, MOV, and other video formats</p>")
                        video_file = gr.Video(
                            label="",
                            show_label=False
                        )
                        video_btn = gr.Button("🎬 Translate Video", elem_classes="btn-primary", size="lg")
                        video_status = gr.HTML(visible=False)
            
            # Results Section
            with gr.Column(elem_classes="card-modern"):
                gr.HTML("""
                    <div class="section-header">
                        <span class="section-icon">📊</span>
                        Translation Results
                    </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("""
                            <div class="result-title">
                                <span>🎯</span>
                                DETECTED SPEECH
                            </div>
                        """)
                        asr_output = gr.Textbox(
                            value="Your speech will be transcribed here...", 
                            interactive=False, 
                            lines=3,
                            show_label=False,
                            elem_classes="result-box"
                        )
                    
                    with gr.Column(scale=1):
                        gr.HTML("""
                            <div class="result-title">
                                <span>🌐</span>
                                TRANSLATED TEXT
                            </div>
                        """)
                        trans_output = gr.Textbox(
                            value="Translation will appear here...", 
                            interactive=False, 
                            lines=3,
                            show_label=False,
                            elem_classes="result-box"
                        )
                
                # TTS Output
                tts_output = gr.Audio(
                    label="🔊 Translated Speech Output",
                    visible=False,
                    elem_classes="result-box"
                )
            
            # History Section
            with gr.Column(elem_classes="card-modern"):
                gr.HTML("""
                    <div class="section-header">
                        <span class="section-icon">🕒</span>
                        Translation History
                    </div>
                """)
                history_html = gr.HTML(
                    value=processor.get_history_display(processor.get_session_state()), 
                    visible=True
                )

        # Hidden triggers
        status_trigger = gr.Textbox(visible=False)

        # ---------------- Handlers ----------------
        def toggle_language_panel():
            """Toggle language panel visibility"""
            return gr.update(visible=True)

        def close_language_panel():
            """Close language panel"""
            return gr.update(visible=False)

        def save_languages(src, tgt):
            """Save language settings"""
            src = src.lower()
            msg, status = processor.change_languages(src, tgt)
            processor.current_source_lang = src
            processor.current_target_lang = tgt
            processor.status = status
            
            # Update status display with better visibility
            status_html = f"""
            <div style='display: flex; align-items: center; gap: 12px;'>
                <span class='status-indicator status-connected'></span>
                <span class='status-text'>System Ready</span>
            </div>
            """
            
            # Update language display with better visibility
            langs_html = f"""
            <div class='language-display'>
                <span style='opacity: 0.9;'>🌐 </span>
                {src.title()} → {tgt.title()}
            </div>
            """
            
            return gr.update(visible=False), status_html, langs_html, ""

        def process_audio_wrapper(state, fp):
            try:
                # Add click effect
                audio_btn.add_class("button-loading")
            except Exception:
                pass
            try:
                state, res_msg, stat, tts_file = processor.process_audio_file(state, fp)
                processor.status = stat
                
                # Enhanced status display with better visibility
                if stat and "processing" in str(stat).lower():
                    badge_class = "status-processing"
                    badge_text = "🔄 Processing"
                    message_class = "status-processing"
                elif stat and str(stat).lower() in ("connected", "done", "success"):
                    badge_class = "status-connected"
                    badge_text = "✅ Completed"
                    message_class = "status-success"
                else:
                    badge_class = "status-error"
                    badge_text = "❌ Error"
                    message_class = "status-error"
                
                status_html = f"""
                <div style='display: flex; align-items: center; gap: 12px;'>
                    <span class='status-indicator {badge_class}'></span>
                    <span class='status-text'>{badge_text}</span>
                </div>
                """
                
                # Status message
                status_msg = f"<div class='status-message {message_class}'>{res_msg}</div>"
                
                # Get texts from session state
                orig_text = state['current_transcription'] or "Your speech will be transcribed here..."
                trans_text = state['current_translation'] or "Translation will appear here..."
                
                return (
                    state,
                    status_msg, 
                    orig_text, 
                    trans_text, 
                    gr.update(value=tts_file, visible=tts_file is not None),
                    processor.get_history_display(state),
                    status_html
                )
            finally:
                try:
                    audio_btn.remove_class("button-loading")
                except Exception:
                    pass

        def process_video_wrapper(state, fp):
            try:
                # Add click effect
                video_btn.add_class("button-loading")
            except Exception:
                pass
            try:
                state, res_msg, stat, tts_file = processor.process_video_file(state, fp)
                processor.status = stat
                
                # Enhanced status display with better visibility
                if stat and "processing" in str(stat).lower():
                    badge_class = "status-processing"
                    badge_text = "🔄 Processing"
                    message_class = "status-processing"
                elif stat and str(stat).lower() in ("connected", "done", "success"):
                    badge_class = "status-connected"
                    badge_text = "✅ Completed"
                    message_class = "status-success"
                else:
                    badge_class = "status-error"
                    badge_text = "❌ Error"
                    message_class = "status-error"
                
                status_html = f"""
                <div style='display: flex; align-items: center; gap: 12px;'>
                    <span class='status-indicator {badge_class}'></span>
                    <span class='status-text'>{badge_text}</span>
                </div>
                """
                
                # Status message
                status_msg = f"<div class='status-message {message_class}'>{res_msg}</div>"
                
                # Get texts from session state
                orig_text = state['current_transcription'] or "Your speech will be transcribed here..."
                trans_text = state['current_translation'] or "Translation will appear here..."
                
                return (
                    state,
                    status_msg, 
                    orig_text, 
                    trans_text, 
                    gr.update(value=tts_file, visible=tts_file is not None),
                    processor.get_history_display(state),
                    status_html
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
                main_text = "Processing..."
            elif "connected" in st_low:
                dot_class = "status-connected"
                main_text = "System Ready"
            else:
                dot_class = "status-error"
                main_text = "System Error"
                
            status_html = f"""
            <div style='display: flex; align-items: center; gap: 12px;'>
                <span class='status-indicator {dot_class}'></span>
                <span class='status-text'>{main_text}</span>
            </div>
            """

            langs_html = f"""
            <div class='language-display'>
                <span style='opacity: 0.9;'>🌐 </span>
                {processor.current_source_lang.title()} → {processor.current_target_lang.title()}
            </div>
            """

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
        language_btn.click(
            toggle_language_panel, 
            outputs=[language_panel]
        )
        cancel_btn.click(
            close_language_panel, 
            outputs=[language_panel]
        )
        save_lang_btn.click(
            save_languages, 
            inputs=[source_lang, target_lang], 
            outputs=[language_panel, status_display, languages_display, status_trigger]
        )

        # Connect file processing with proper outputs including session state
        audio_btn.click(
            process_audio_wrapper, 
            inputs=[session_state, audio_file], 
            outputs=[session_state, audio_status, asr_output, trans_output, tts_output, history_html, status_display]
        )
        
        video_btn.click(
            process_video_wrapper, 
            inputs=[session_state, video_file], 
            outputs=[session_state, video_status, asr_output, trans_output, tts_output, history_html, status_display]
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
