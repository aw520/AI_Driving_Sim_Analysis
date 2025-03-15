from pydub import AudioSegment
import whisper as whisper
import os
import pandas as pd
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

#classify sentiment
def classify_sentiment(text):
    scores = sia.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"
    

#single video file
def process_video(mp4_file):
    """
    Processes an MP4 file: extracts audio, segments it, transcribes text, 
    analyzes sentiment, and saves the results in a CSV file.

    Args:
        mp4_file (str): The path to the video file.

    Returns:
        str: Path to the generated CSV file.
    """
    print(f"Processing: {mp4_file}")

    #check the exsistence
    if not os.path.exists(mp4_file):
        print(f"Error: File '{mp4_file}' not found.")
        return None

    #MP4 to WAV (16kHz, mono)
    wav_file = mp4_file.replace(".mp4", ".wav")
    audio = AudioSegment.from_file(mp4_file, format="mp4")
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_file, format="wav")

    #Whisper model
    model = whisper.load_model("base")

    #segment audio into 5s chunks
    chunk_length_ms = 5000  # 5 seconds
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    #store results
    data = []

    for i, chunk in enumerate(chunks):
        start_time = i * 5  # Start time in seconds
        chunk_filename = f"chunk_{i}.wav"
        chunk.export(chunk_filename, format="wav")

        #transcribe with Whisper
        result = model.transcribe(chunk_filename)
        text = result["text"].strip()

        #sentiment analysis
        sentiment = classify_sentiment(text) if text else "Neutral"

        #store data
        data.append({
            "Timestamp (s)": start_time,
            "Transcription": text,
            "Sentiment": sentiment
        })

        print(f"[{start_time}s] {text} - Sentiment: {sentiment}")

        os.remove(chunk_filename)  #cleanup

    df = pd.DataFrame(data)

    #csv
    output_dir = "transcriptions"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dir, f"transcription_{timestamp}.csv")

    #save
    df.to_csv(csv_filename, index=False, encoding="utf-8")
    print(f"\nTranscription complete. Saved to: {csv_filename}")

    return csv_filename

#process multiple videos
def process_all_videos(video_folder):
    """
    Processes all MP4 files in a given folder.

    Args:
        video_folder (str): Path to the directory containing video files.
    """
    print(f"Scanning directory: {video_folder}")

    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    
    if not video_files:
        print("No video files found in the directory.")
        return

    for video in video_files:
        video_path = os.path.join(video_folder, video)
        process_video(video_path)

#process for single video
if __name__ == "__main__":
    video_file = "Experimenter_CREW.mp4" 
    csv_output = process_video(video_file)