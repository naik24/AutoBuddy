# AutoBuddy: A Car Sales 🚗 Assistant ![project-stage-badge: Experimental](https://img.shields.io/badge/Project%20Stage-Experimental-yellow.svg)

## Workflow
<p align = "center"><img src = "https://github.com/user-attachments/assets/ed346515-5445-44be-beff-cbe23247b9f4"></p>

## Demo
<p align = "center"><a href = "https://www.youtube.com/watch?v=sXzgfXwr6Yo"><img src = "https://img.youtube.com/vi/sXzgfXwr6Yo/0.jpg"></a></p>

## Keys
The project uses to APIs: <a href = "https://fireworks.ai/">Fireworks</a> and <a href = "https://openai.com/">OpenAI</a>. The Fireworks-AI employs the LLM to classify the documents and OpenAI TTS (Text-To-Speech) is used to convert text to speech for the AI assistant.

## Setup

1. Clone the repository:
```python
git clone https://github.com/naik24/AutoBuddy.git
cd AutoBuddy
```

2. Create and activate virtual environment
```
python3 -m venv autobuddy
source autobuddy/bin/activate
```

3. Install Dependencies
```
pip3 install -r requirements.txt
```

4. The project uses the ```pyaudio``` dependency which has additional dependencies based on your OS.

For MacOS:
```python
brew install portaudio
```

5. Run the app
```python
streamlit run app.py
```
## add instruction about installing pyaudio
