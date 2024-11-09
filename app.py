# Step 2: Import required libraries
import streamlit as st
import whisper
from transformers import pipeline
from moviepy.editor import VideoFileClip

# Step 3: Load the Whisper model and Summarization pipeline
transcription_model = whisper.load_model("base")
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

# Step 4: Define a function to transcribe video
def transcribe_video(video_path):
    # Extract audio from video
    video = VideoFileClip(video_path)
    audio_path = "audio.wav"
    video.audio.write_audiofile(audio_path)
    
    # Transcribe audio
    transcription = transcription_model.transcribe(audio_path)
    return transcription['text']

# Step 5: Define a function to summarize the transcribed text
def summarize_text(text):
    # Use the summarization pipeline to summarize the text
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

# Step 6: Create the Streamlit app
def main():
    st.title("Video Transcription and Summarization App")
    st.write("Upload a video file to get its text transcription and summary.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_file is not None:
        # Save the uploaded video
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Transcribe the video
        st.write("Transcribing the video...")
        transcription_text = transcribe_video(video_path)
        st.write("Transcription:")
        st.write(transcription_text)
        
        # Summarize the transcription
        st.write("Summarizing the transcription...")
        summary_text = summarize_text(transcription_text)
        st.write("Summary:")
        st.write(summary_text)

if __name__ == "__main__":
    main()