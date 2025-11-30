# Speech-to-Speech Translator 🈺

A real-time multilingual speech translation system that converts speech between English, Hindi, and 10 other Indian languages using state-of-the-art AI models.

## 🌟 Live Demo

🚀 **Try it now!**: [Speech-to-Speech Translator on Hugging Face](https://huggingface.co/spaces/Sushant-Kumar-Sinha/speech-to-speech-translator)

## ✨ Features

- **🎤 Speech Recognition**: Convert audio/video files to text using Whisper ASR
- **🌍 Multilingual Translation**: Translate between 11 Indian languages using Facebook NLLB
- **🔊 Text-to-Speech**: Generate natural sounding speech in target language
- **📁 File Support**: Process both audio (MP3, WAV) and video (MP4) files
- **⚡ Low Latency**: Optimized model selection (Whisper-tiny for English, Whisper-medium for other language for better accuracy)
- **💾 Translation Cache**: Smart caching for faster repeated translations
- **📱 Beautiful UI**: Gradio-based intuitive web interface
- **🔄 Real-time Processing**: Fast inference with model warm-up

## 🗣️ Supported Languages

| Source Languages | Target Languages |
|-----------------|------------------|
| 🇮🇳 English      | 🇮🇳 Hindi         |
| 🇮🇳 Hindi        | 🇮🇳 English       |
| 🇮🇳 Bengali      | 🇮🇳 Bengali       |
| 🇮🇳 Tamil        | 🇮🇳 Tamil         |
| 🇮🇳 Telugu       | 🇮🇳 Telugu        |
| 🇮🇳 Marathi      | 🇮🇳 Marathi       |
| 🇮🇳 Gujarati     | 🇮🇳 Gujarati      |
| 🇮🇳 Kannada      | 🇮🇳 Kannada       |
| 🇮🇳 Malayalam    | 🇮🇳 Malayalam     |
| 🇮🇳 Punjabi      | 🇮🇳 Punjabi       |
| 🇮🇳 Urdu         | 🇮🇳 Urdu          |

## 🏗️ System Architecture

Audio/Video Input
↓
[Whisper ASR] → Speech to Text
↓
[NLLB Translator] → Text Translation
↓
[Google TTS] → Text to Speech
↓
Translated Audio Output
## 🚀 Quick Start

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

1. ASR: OpenAI Whisper (tiny for English, Medium for Hindi)
2. Translation: Facebook NLLB-200 Distilled 600M
3. TTS: Google Text-to-Speech

   metadata
   
title: Speech To Speech Translator

emoji: 🔊

colorFrom: blue

colorTo: green

sdk: gradio

sdk_version: 5.49.1

app_file: app.py

pinned: false

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
