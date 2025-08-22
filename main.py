#!/usr/bin/env python3
"""
Enhanced Indonesian Subtitle Generator
Menggunakan Whisper untuk speech-to-text dan Google Translate untuk terjemahan
dengan optimasi performa dan error handling yang lebih baik.
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import json

try:
    import whisper
    from googletrans import Translator
    import torch
except ImportError as e:
    print(f"Error: Library yang dibutuhkan tidak ditemukan: {e}")
    print("Install dengan: pip install openai-whisper googletrans==4.0.0-rc1 torch")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SubtitleGenerator:
    """
    Enhanced subtitle generator dengan optimasi dan error handling
    """
    
    def __init__(self, model_size: str = "small", max_workers: int = 3):
        self.model_size = model_size
        self.max_workers = max_workers
        self.translator = None
        self.model = None
        self.translate_lock = Lock()  # Untuk thread safety Google Translate
        
        # Cache untuk menghindari terjemahan berulang
        self.translation_cache: Dict[str, str] = {}
        
        # Supported model sizes dengan memory requirements
        self.model_info = {
            "tiny": {"memory": "~39 MB", "speed": "fastest", "accuracy": "lowest"},
            "base": {"memory": "~74 MB", "speed": "fast", "accuracy": "low"},
            "small": {"memory": "~244 MB", "speed": "medium", "accuracy": "medium"},
            "medium": {"memory": "~769 MB", "speed": "slow", "accuracy": "high"},
            "large": {"memory": "~1550 MB", "speed": "slowest", "accuracy": "highest"}
        }
        
    def _setup_model(self, verbose: bool = False) -> None:
        """Load Whisper model dengan error handling"""
        try:
            if verbose:
                model_details = self.model_info.get(self.model_size, {})
                logger.info(f"Memuat model Whisper '{self.model_size}'...")
                logger.info(f"Memory requirement: {model_details.get('memory', 'Unknown')}")
                logger.info(f"Speed: {model_details.get('speed', 'Unknown')}")
                
            # Check CUDA availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if verbose:
                logger.info(f"Menggunakan device: {device.upper()}")
                
            self.model = whisper.load_model(self.model_size, device=device)
            
        except Exception as e:
            logger.error(f"Gagal memuat model Whisper: {e}")
            raise
            
    def _setup_translator(self) -> None:
        """Setup Google Translator dengan retry logic"""
        try:
            self.translator = Translator()
            # Test connection
            test_translation = self.translator.translate("test", dest="id")
            logger.info("Google Translator berhasil diinisialisasi")
        except Exception as e:
            logger.error(f"Gagal menginisialisasi translator: {e}")
            raise
            
    def format_time(self, seconds: float) -> str:
        """Format waktu ke format SRT dengan presisi lebih baik"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds * 1000) % 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
        
    def _translate_text(self, text: str, max_retries: int = 3) -> str:
        """
        Translate text dengan caching dan retry logic
        """
        # Check cache first
        text_key = text.strip().lower()
        if text_key in self.translation_cache:
            return self.translation_cache[text_key]
            
        for attempt in range(max_retries):
            try:
                with self.translate_lock:  # Thread safety
                    time.sleep(0.1)  # Rate limiting
                    translated = self.translator.translate(text, dest="id").text
                    
                # Cache the result
                self.translation_cache[text_key] = translated
                return translated
                
            except Exception as e:
                logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Gagal menerjemahkan setelah {max_retries} percobaan: '{text}'")
                    return text  # Return original text if translation fails
                    
    def _translate_segments_parallel(self, segments: List[Dict], verbose: bool = False) -> List[Dict]:
        """
        Translate segments menggunakan parallel processing
        """
        translated_segments = [None] * len(segments)
        
        def translate_segment(index_segment_tuple):
            index, segment = index_segment_tuple
            try:
                translated_text = self._translate_text(segment["text"])
                return index, {
                    **segment,
                    "translated_text": translated_text
                }
            except Exception as e:
                logger.error(f"Error translating segment {index}: {e}")
                return index, {
                    **segment,
                    "translated_text": segment["text"]  # Fallback to original
                }
        
        if verbose:
            logger.info(f"Menerjemahkan {len(segments)} segmen menggunakan {self.max_workers} workers...")
            
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all translation tasks
            future_to_index = {
                executor.submit(translate_segment, (i, segment)): i 
                for i, segment in enumerate(segments)
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_index):
                try:
                    index, translated_segment = future.result()
                    translated_segments[index] = translated_segment
                    completed += 1
                    
                    if verbose and completed % 10 == 0:
                        logger.info(f"Progress: {completed}/{len(segments)} segmen diterjemahkan")
                        
                except Exception as e:
                    logger.error(f"Error dalam parallel translation: {e}")
                    
        return [seg for seg in translated_segments if seg is not None]
        
    def validate_input_file(self, video_file: str) -> bool:
        """Validate input video file"""
        if not os.path.exists(video_file):
            logger.error(f"File tidak ditemukan: {video_file}")
            return False
            
        # Check file size (warn if > 2GB)
        file_size = os.path.getsize(video_file) / (1024 * 1024 * 1024)  # GB
        if file_size > 2:
            logger.warning(f"File berukuran besar ({file_size:.1f} GB). Proses mungkin memakan waktu lama.")
            
        # Check file extension
        valid_extensions = {'.mp4', '.mp3', '.wav', '.m4a', '.flac', '.avi', '.mkv', '.mov'}
        file_ext = Path(video_file).suffix.lower()
        if file_ext not in valid_extensions:
            logger.warning(f"Ekstensi file '{file_ext}' mungkin tidak didukung oleh Whisper")
            
        return True
        
    def save_subtitle_file(self, segments: List[Dict], output_file: str, verbose: bool = False) -> None:
        """Save segments to SRT file dengan error handling"""
        try:
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, start=1):
                    start = segment["start"]
                    end = segment["end"]
                    translated_text = segment.get("translated_text", segment["text"])
                    
                    f.write(f"{i}\n")
                    f.write(f"{self.format_time(start)} --> {self.format_time(end)}\n")
                    f.write(f"{translated_text.strip()}\n\n")
                    
                    if verbose and i % 20 == 0:
                        logger.info(f"Menulis segmen {i}/{len(segments)}")
                        
            logger.info(f"Subtitle berhasil disimpan: {output_file}")
            
            # Show file info
            file_size = os.path.getsize(output_file)
            logger.info(f"Ukuran file subtitle: {file_size} bytes ({len(segments)} segmen)")
            
        except Exception as e:
            logger.error(f"Gagal menyimpan file subtitle: {e}")
            raise
            
    def save_cache(self, cache_file: str) -> None:
        """Save translation cache untuk penggunaan selanjutnya"""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.translation_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Cache terjemahan disimpan: {cache_file}")
        except Exception as e:
            logger.warning(f"Gagal menyimpan cache: {e}")
            
    def load_cache(self, cache_file: str) -> None:
        """Load translation cache dari file sebelumnya"""
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.translation_cache = json.load(f)
                logger.info(f"Cache terjemahan dimuat: {len(self.translation_cache)} entri")
        except Exception as e:
            logger.warning(f"Gagal memuat cache: {e}")
            
    def generate_subtitle(self, video_file: str, output_file: str, verbose: bool = False, 
                         use_cache: bool = True) -> bool:
        """
        Main function untuk generate subtitle dengan optimasi
        """
        try:
            start_time = time.time()
            
            # Validate input
            if not self.validate_input_file(video_file):
                return False
                
            # Setup cache
            cache_file = f"{Path(output_file).stem}_cache.json"
            if use_cache:
                self.load_cache(cache_file)
                
            # Initialize model and translator
            self._setup_model(verbose)
            self._setup_translator()
            
            # Transcribe audio
            if verbose:
                logger.info("Memulai transkripsi audio...")
                
            result = self.model.transcribe(
                video_file,
                verbose=verbose,
                language=None,  # Auto-detect language
                task="transcribe",
                fp16=torch.cuda.is_available()  # Use FP16 if CUDA available
            )
            
            # Log detected language
            detected_lang = result.get("language", "unknown")
            if verbose:
                logger.info(f"Bahasa terdeteksi: {detected_lang}")
                logger.info(f"Jumlah segmen: {len(result['segments'])}")
                
            # Translate segments in parallel
            if verbose:
                logger.info("Memulai terjemahan ke Bahasa Indonesia...")
                
            translated_segments = self._translate_segments_parallel(
                result["segments"], verbose
            )
            
            # Save subtitle file
            self.save_subtitle_file(translated_segments, output_file, verbose)
            
            # Save cache
            if use_cache:
                self.save_cache(cache_file)
                
            # Show completion info
            elapsed_time = time.time() - start_time
            logger.info(f"Proses selesai dalam {elapsed_time:.2f} detik")
            logger.info(f"Rata-rata: {len(translated_segments) / elapsed_time:.2f} segmen/detik")
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Proses dibatalkan oleh user")
            return False
        except Exception as e:
            logger.error(f"Error dalam generate_subtitle: {e}")
            return False


def main():
    """Main function dengan enhanced argument parsing"""
    parser = argparse.ArgumentParser(
        description="Enhanced Indonesian Subtitle Generator menggunakan Whisper + Google Translate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  %(prog)s -i video.mp4 -o subtitle.srt -v
  %(prog)s -i audio.mp3 -m medium --max-workers 5
  %(prog)s -i file.wav -o output/subtitle.srt --no-cache
        """
    )
    
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path ke file video/audio input (mp4, mp3, wav, dll)"
    )
    
    parser.add_argument(
        "-o", "--output", 
        default="subtitle_id.srt", 
        help="Path file output subtitle (default: subtitle_id.srt)"
    )
    
    parser.add_argument(
        "-m", "--model", 
        choices=["tiny", "base", "small", "medium", "large"],
        default="small",
        help="Ukuran model Whisper (default: small). Model lebih besar = akurasi lebih tinggi tapi lebih lambat"
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Tampilkan progress detail dan informasi debug"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Jumlah worker untuk parallel translation (default: 3)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching untuk terjemahan"
    )
    
    args = parser.parse_args()
    
    # Show model information
    if args.verbose:
        generator = SubtitleGenerator()
        model_info = generator.model_info.get(args.model, {})
        logger.info(f"Model yang dipilih: {args.model}")
        logger.info(f"Memory requirement: {model_info.get('memory', 'Unknown')}")
        logger.info(f"Expected speed: {model_info.get('speed', 'Unknown')}")
        logger.info(f"Expected accuracy: {model_info.get('accuracy', 'Unknown')}")
    
    # Initialize and run subtitle generator
    generator = SubtitleGenerator(
        model_size=args.model,
        max_workers=args.max_workers
    )
    
    success = generator.generate_subtitle(
        video_file=args.input,
        output_file=args.output,
        verbose=args.verbose,
        use_cache=not args.no_cache
    )
    
    if success:
        print(f"\n[+] Subtitle Bahasa Indonesia berhasil dibuat: {args.output}")
        sys.exit(0)
    else:
        print(f"\n[-] Gagal membuat subtitle")
        sys.exit(1)


if __name__ == "__main__":
    main()
