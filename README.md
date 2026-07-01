# Speech-to-Speech Translator 🔊

A real-time, multilingual speech translation system that seamlessly bridges communication gaps by converting spoken language between English, Hindi, and 10 regional Indian languages using state-of-the-art AI models.

[![Deployment Status](https://img.shields.io/badge/Deployment-Hugging%20Face-blue)](https://huggingface.co/spaces/Sushant-Kumar-Sinha/speech-to-speech-translator)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

* **🎤 Advanced Speech Recognition:** High-fidelity audio/video transcription using OpenAI's Whisper ASR.
* **🌍 Multilingual Translation:** Context-aware translation across 11 major Indian languages via Facebook's NLLB.
* **🔊 Natural Text-to-Speech:** Generates expressive, human-like synthesized voice output in the target language.
* **📁 Diverse File Support:** Native support for processing both audio files (`.mp3`, `.wav`) and video containers (`.mp4`).
* **⚡ Dual-Engine Low Latency:** Optimized model orchestration (`Whisper-tiny` for swift English processing; `Whisper-medium` for dense regional accuracy).
* **💾 Smart Translation Cache:** On-the-fly caching to eliminate redundant API/model computation for repeated phrases.
* **📱 Premium UI/UX:** Responsive, intuitive web interface built natively on Gradio.
* **🔄 Production Ready:** Features built-in model warm-ups to guarantee immediate real-time processing inference.

---

## 🗣️ Supported Languages

| Source Languages | Target Languages |
| :--- | :--- |
| 🇬🇧 English | 🇬🇧 English |
| 🇮🇳 Hindi | 🇮🇳 Hindi |
| 🇮🇳 Bengali | 🇮🇳 Bengali |
| 🇮🇳 Tamil | 🇮🇳 Tamil |
| 🇮🇳 Telugu | 🇮🇳 Telugu |
| 🇮🇳 Marathi | 🇮🇳 Marathi |
| 🇮🇳 Gujarati | 🇮🇳 Gujarati |
| 🇮🇳 Kannada | 🇮🇳 Kannada |
| 🇮🇳 Malayalam | 🇮🇳 Malayalam |
| 🇮🇳 Punjabi | 🇮🇳 Punjabi |
| 🇮🇳 Urdu | 🇮🇳 Urdu |

---

## 🏗️ System Architecture

```text
  [ Audio/Video Input ]
           │
           ▼
     [Whisper ASR]    ──> (Extracts & Transcribes Speech to Text)
           │
           ▼
   [NLLB Translator]  ──> (Translates Context Across Indian Languages)
           │
           ▼
     [Google TTS]     ──> (Synthesizes Text into Natural Speech)
           │
           ▼
 [ Translated Audio Output ]

### Installation

1. **Clone the repository**
   
git clone https://github.com/Sushant-Kumar-Sinha/speech_to_speech_translator.git

cd speech_to_speech_translation

Usage:-

1. Upload Files: Select audio (MP3, WAV) or video (MP4) files
2. Choose Languages: Select source (English/Hindi) and target language
3. Translate: Click "Translate Audio" or "Translate Video"
4. Get Results: View transcribed text, translation, and listen to TTS output

📁 Project Structure

speech_to_speech_translator/

├── app.py       # Main Gradio application

├── model.py              # AI models (Whisper, NLLB, TTS)

├── requirements.txt       # Python dependencies

└── README.md             # Project documentation

Models Used:

1. ASR: OpenAI Whisper (tiny for English, Medium for others)
2. Translation: Facebook NLLB-200 Distilled 600M
3. TTS: Google Text-to-Speech


📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
OpenAI Whisper for speech recognition
Facebook NLLB for translation
Hugging Face for model hosting and Spaces
Gradio for the web interface

📞 Contact
Sushant Kumar Sinha

GitHub: @Sushant-Kumar-Sinha

Hugging Face: @Sushant-Kumar-Sinha

Project Link: https://github.com/Sushant-Kumar-Sinha/speech_to_speech_translator
