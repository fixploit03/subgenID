import argparse
import whisper
from googletrans import Translator


def format_time(seconds: float):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds * 1000) % 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def generate_subtitle(video_file, output_file, verbose=False):
    if verbose:
        print("Memuat model Whisper...")
    model = whisper.load_model("small")

    if verbose:
        print("Sedang transkrip dan deteksi bahasa...")
    result = model.transcribe(video_file)

    translator = Translator()

    with open(output_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], start=1):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]

            translated = translator.translate(text, dest="id").text

            f.write(f"{i}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{translated.strip()}\n\n")

            if verbose:
                print(f"[{format_time(start)} - {format_time(end)}] {translated}")

    print(f"Subtitle Bahasa Indonesia berhasil dibuat: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Indonesian subtitles from video using Whisper + Google Translate.")
    parser.add_argument("-i", "--input", required=True, help="Path ke file video input (misalnya: video.mp4)")
    parser.add_argument("-o", "--output", default="subtitle_id.srt", help="Path file output subtitle (default: subtitle_id.srt)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Tampilkan progress detail")
    args = parser.parse_args()

    generate_subtitle(args.input, args.output, args.verbose)
      
