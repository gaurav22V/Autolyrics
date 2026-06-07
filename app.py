import gradio as gr
import torch
import librosa
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# LOCAL CONFIG
MODEL_ID = "openai/whisper-small"
OUTPUT_DIR = "./models/autolyrics_lora"  
SAMPLING_RATE = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initializing AutoLyrics Application on {device.upper()}...")

# Load standard processor
processor = WhisperProcessor.from_pretrained(MODEL_ID)
base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

print(f"Attaching Fine-Tuned LoRA Weights from {OUTPUT_DIR}...")
try:
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model.to(device)
    print("AutoLyrics Engine Ready")
except Exception as e:
    print(f"ERROR: Could not load LoRA adapter.\nError: {e}")
    model = base_model.to(device)   # Fallback to base model

def transcribe_audio(audio_filepath):
    if audio_filepath is None:
        return "ERRORr: Please upload or record an audio file first.", "0.00 seconds"

    print("Processing incoming audio stream...")
    start_time = time.time()

    try:
        # Load and force resample to 16kHz
        audio_array, _ = librosa.load(audio_filepath, sr=SAMPLING_RATE)

        # Feature Extraction
        input_features = processor(
            audio_array, 
            sampling_rate=SAMPLING_RATE, 
            return_tensors="pt"
        ).input_features.to(device)

        # Generation
        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_new_tokens=400)

        # Decode output
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip().lower()
        
        # Clean up punctuations
        clean_text = "".join(char for char in transcription if char.isalnum() or char.isspace())

        end_time = time.time()
        latency = round(end_time - start_time, 2)

        print(f"Transcribed in {latency}s")
        return clean_text, f"{latency} seconds"

    except Exception as e:
        return f"Audio Processing Error: {str(e)}", "N/A"

# Styling Customizations
custom_css = """
.gradio-container {font-family: 'Inter', system-ui, sans-serif;}
.header-text {text-align: center; color: #2D3748; margin-bottom: 0.2rem;}
.sub-text {text-align: center; color: #718096; margin-bottom: 2rem; font-size: 1.1rem;}
.footer {text-align: center; margin-top: 2rem; color: #A0AEC0; font-size: 0.9rem;}
"""

with gr.Blocks(css=custom_css) as demo:
    
    # App Header
    gr.Markdown("<h1 class='header-text'> AutoLyrics </h1>")
    gr.Markdown("<p class='sub-text'>Vocal Transcription Engine | Fine-Tuned on NUS-48E</p>")
    
    with gr.Row():
        # Left Column: Inputs
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath", 
                label="Input Audio Stream"
            )
            transcribeButton = gr.Button("Generate Lyrics", variant="primary", size="lg")
            
        # Right Column: Outputs
        with gr.Column(scale=1):
            text_output = gr.Textbox(
                label="Transcribed Output", 
                placeholder="AI generation will appear here...", 
                lines=5
            )
            latency_output = gr.Textbox(
                label="Engine Latency", 
                placeholder="0.00s"
            )

    transcribeButton.click(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=[text_output, latency_output]
    )

# Starting server
if __name__ == "__main__":
    print("Starting web server...")
    demo.launch(
        share=False, 
        theme=gr.themes.Soft(primary_hue="indigo")
    )