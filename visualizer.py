import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def load_transcription_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return None
    return pd.read_csv(filepath)

def plot_word_histogram(df):
    df["Bucket"] = df["Timestamp (s)"] // 5 * 5  #group into 5s intervals

    #clean transcriptions(ensure all String)
    df["Transcription"] = df["Transcription"].astype(str).fillna("")

    #count words per time bucket
    word_counts = df.groupby("Bucket")["Transcription"].apply(lambda x: sum(len(str(t).split()) for t in x)).reset_index()

    #Histogram
    plt.figure(figsize=(12, 6))
    plt.bar(word_counts["Bucket"], word_counts["Transcription"], width=5, align="edge")
    plt.xlabel("Time Buckets (seconds)")
    plt.ylabel("Word Count")
    plt.title("Histogram of Transcribed Words per 5-Second Interval")
    x_ticks = word_counts["Bucket"].values
    plt.xticks(x_ticks[::3]) 
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def plot_sentiment_distribution(df):
    sentiment_counts = df["Sentiment"].value_counts()

    plt.figure(figsize=(8, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=140, colors=["green", "red", "gray"])
    plt.title("Sentiment Distribution")
    plt.show()


def process_all_csv():
    transcriptions_folder = "transcriptions"

    if not os.path.exists(transcriptions_folder):
        print(f"Error: Folder '{transcriptions_folder}' does not exist.")
        return

    csv_files = [f for f in os.listdir(transcriptions_folder) if f.endswith(".csv")]

    if not csv_files:
        print("No transcription CSV files found in the folder.")
        return

    #process each CSV
    for csv_file in csv_files:
        csv_path = os.path.join(transcriptions_folder, csv_file)
        df = load_transcription_data(csv_path)

        if df is not None:
            print(f"Processing {csv_file}...")
            plot_word_histogram(df) #Histogram
            plot_sentiment_distribution(df) #Sentiment Distribution

if __name__ == "__main__":
    process_all_csv()