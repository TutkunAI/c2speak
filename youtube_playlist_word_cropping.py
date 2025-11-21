#!/usr/bin/env python3
"""
YouTube playlist → word-level .raw + features DB (NO Codec2 for feature extraction)
Fix: Features are now extracted from .raw files, not .bit compressed Codec2 files.
Codec2 encoding is still saved, but not used for features.
"""

import argparse
import re
import subprocess
from pathlib import Path
from typing import List, Dict
import sqlite3

import whisper
from pydub import AudioSegment
import yt_dlp as ytdl

from phonemizer import phonemize
import numpy as np
import torch
import torchaudio

try:
    import whisperx
    HAVE_WHISPERX = True
except ImportError:
    HAVE_WHISPERX = False

# ---------- helpers ----------
def sanitize_filename(s: str) -> str:
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "", s)
    return s or "unnamed"

def run_ffmpeg_conv(in_path: Path, out_path: Path, sr: int = 8000, ch: int = 1):
    cmd = [
        "ffmpeg", "-y", "-i", str(in_path),
        "-ar", str(sr), "-ac", str(ch),
        "-acodec", "pcm_s16le",
        str(out_path)
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ---------- download ----------
def download_playlist_audio(url: str, outdir: Path) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    # Auto-detect cookies.txt
    possible_cookie_locations = [
        Path("cookies.txt"),
        Path("/content/cookies.txt"),
        Path("cookies/cookies.txt"),
    ]

    cookie_file = None
    for p in possible_cookie_locations:
        if p.exists():
            cookie_file = str(p)
            print(f"Using cookies file: {cookie_file}")
            break

    if cookie_file is None:
        print("WARNING: cookies.txt not found — downloads may fail for restricted videos.\n"
              "Place cookies.txt in the same folder as the script or /content/ in Colab.")

    # yt-dlp options
    opts = {
        "ignoreerrors": True,
        "quiet": False,
        "format": "bestaudio/best",
        "cookiefile": cookie_file,         # <-- ENABLE COOKIES
        "nocheckcertificate": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        },
        "outtmpl": str(outdir / "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "m4a",
            "preferredquality": "192",
        }],
    }

    audio_files = []
    with ytdl.YoutubeDL(opts) as y:
        try:
            info = y.extract_info(url, download=False)
        except Exception as e:
            print(f"Playlist fetch error: {e}")
            return []

        entries = info.get("entries") if info else None

        if entries:
            for e in entries:
                if not e:
                    continue
                vid = e.get("id")
                video_url = f"https://www.youtube.com/watch?v={vid}"
                try:
                    print(f"Downloading {video_url}")
                    y.download([video_url])
                except Exception as ex:
                    print(f"Download failed for {vid}: {ex}")
        else:
            y.download([url])

    # Collect audio files
    for f in outdir.iterdir():
        if f.suffix.lower() in (".m4a", ".mp3", ".aac", ".webm", ".wav"):
            audio_files.append(f)

    return audio_files


# ---------- transcription ----------
def transcribe(audio: Path, model_name="medium", language="en") -> Dict:
    model = whisper.load_model(model_name)
    return model.transcribe(str(audio), language=language, word_timestamps=False)

# ---------- alignment & IPA phonemes ----------
def get_word_phoneme_timestamps(audio_path: Path, result: Dict, language="en", device="cpu") -> List[Dict]:
    entries = []
    if HAVE_WHISPERX:
        print("Running WhisperX forced phoneme alignment")
        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
        aligned = whisperx.align(result["segments"], align_model, metadata, str(audio_path), device, return_char_alignments=True)
        char_segs = aligned.get("char_segments", [])
        word_segs = aligned.get("word_segments", aligned.get("segments", []))

        phoneme_map: Dict[int, List[str]] = {}
        for cs in char_segs:
            widx = cs.get("word-idx")
            phoneme = cs.get("char") or cs.get("text") or ""
            if widx is None: continue
            phoneme_map.setdefault(widx, []).append(phoneme)

        for widx, w in enumerate(word_segs):
            word_text = w.get("text") or w.get("word") or ""
            start = float(w.get("start", 0.0))
            end = float(w.get("end", 0.0))
            phoneme_seq = phoneme_map.get(widx, [])
            phoneme_str = "_".join([p for p in phoneme_seq if p])
            if not phoneme_str:
                safe_text = re.sub(r"[^A-Za-z']+", "", word_text)
                try:
                    phoneme_str = phonemize(
                        safe_text,
                        language="en-us",
                        backend="espeak",
                        strip=True,
                        preserve_punctuation=False,
                        njobs=1,
                    ).replace(" ", "_")
                except Exception:
                    phoneme_str = sanitize_filename(safe_text)
            entries.append({"word": word_text.strip(), "start": start, "end": end, "phonemes": phoneme_str})
        return entries

    # fallback
    for seg in result.get("segments", []):
        seg_text = seg.get("text", "").strip()
        s, e = seg.get("start", 0.0), seg.get("end", 0.0)
        dur = max(1e-3, e - s)
        parts = re.findall(r"\S+", seg_text)
        if not parts: continue
        step = dur / len(parts)
        for i, w in enumerate(parts):
            start = s + i*step
            end = s + (i+1)*step
            safe_text = re.sub(r"[^A-Za-z']+", "", w)
            try:
                phoneme_str = phonemize(
                    safe_text,
                    language="en-us",
                    backend="espeak",
                    strip=True,
                    preserve_punctuation=False,
                    njobs=1
                ).replace(" ", "_")
            except Exception:
                phoneme_str = sanitize_filename(safe_text)
            entries.append({"word": w, "start": start, "end": end, "phonemes": phoneme_str})
    return entries

# ---------- DB ----------
def init_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT,
            phonemes TEXT,
            start REAL,
            end REAL,
            audio_file TEXT,
            codec2_file TEXT,
            features_file TEXT
        )
    ''')
    conn.commit()
    return conn

# ---------- feature extraction ----------
def extract_features(raw_path: Path, sr: int = 8000, n_mels: int = 80):
    y = np.fromfile(raw_path, dtype=np.int16).astype(np.float32) / 32768.0
    y_t = torch.tensor(y).unsqueeze(0)

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=512,
        hop_length=160,
        win_length=512,
        n_mels=n_mels,
        power=2.0
    )(y_t)

    mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec).squeeze(0)

    f0 = torchaudio.functional.detect_pitch_frequency(
        y_t,
        sample_rate=sr,
        frame_time=0.02
    ).squeeze(0).numpy()

    f0 = np.nan_to_num(f0)

    rms = torch.sqrt(torch.mean(y_t.unfold(1, 512, 160)**2, dim=2)).squeeze(0)
    rms = rms.numpy()

    return mel_db.numpy(), f0, rms

# ---------- cropping + Codec2 + features + DB ----------
def crop_save_raw_and_codec2(audio_path: Path, entries: List[Dict], outdir: Path, codec2_bin: Path, codec2_outdir: Path, conn):
    outdir.mkdir(parents=True, exist_ok=True)
    codec2_outdir.mkdir(parents=True, exist_ok=True)
    features_dir = outdir / "features"
    features_dir.mkdir(exist_ok=True)
    audio = AudioSegment.from_file(str(audio_path))
    total_ms = len(audio)
    c = conn.cursor()

    for idx, ent in enumerate(entries):
        try:
            word = ent["word"]
            phonemes = ent["phonemes"]
            start = float(ent["start"])
            end = float(ent["end"])
            dur = max(0.001, end-start)
            pad = dur*0.25
            a = max(0.0, start - pad)
            b = min(total_ms/1000.0, end + pad)
            crop = audio[int(a*1000):int(b*1000)]
            crop = crop.set_frame_rate(8000).set_channels(1).set_sample_width(2)

            raw_path = outdir / f"{idx:05d}_{phonemes}.raw"
            crop.export(raw_path, format="raw")

            c2_out = codec2_outdir / f"{raw_path.stem}.bit"
            bit_rate = "2400"
            subprocess.run([str(codec2_bin), bit_rate, str(raw_path), str(c2_out)], check=True)

            # FIX: Extract features from .raw directly
            mel, f0, rms = extract_features(raw_path)
            feat_path = features_dir / f"{raw_path.stem}.npz"
            np.savez_compressed(feat_path, mel=mel, f0=f0, rms=rms)

            c.execute('''
                INSERT INTO words (word, phonemes, start, end, audio_file, codec2_file, features_file)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (word, phonemes, start, end, str(raw_path), str(c2_out), str(feat_path)))
            conn.commit()

            print(f"Saved {raw_path.name}, Codec2 {c2_out.name}, features {feat_path.name}")
        except Exception as e:
            print(f"Failed word {ent}: {e}")

# ---------- orchestrator ----------
def process_playlist(url: str, outroot: Path, tmproot: Path, codec2_bin: Path, model="medium", language="en", device="cpu"):
    tmproot.mkdir(parents=True, exist_ok=True)
    raw_dir = tmproot / "raw"
    raw_dir.mkdir(exist_ok=True)

    db_path = outroot / "phonemes.db"
    conn = init_db(db_path)

    print("Downloading playlist audio …")
    files = download_playlist_audio(url, raw_dir)
    if not files:
        print("No audio found.")
        return

    trans_dir = tmproot / "transcribe"
    trans_dir.mkdir(exist_ok=True)
    codec2_outdir = outroot / "codec2"

    for f in files:
        base = f.stem
        t_wav = trans_dir / f"{base}.whisper.wav"
        print(f"Preparing {base} (16 kHz mono)")
        run_ffmpeg_conv(f, t_wav, sr=16000, ch=1)

        print(f"Transcribing {base}")
        res = transcribe(t_wav, model_name=model, language=language)

        entries = get_word_phoneme_timestamps(t_wav, res, language=language, device=device)
        crop_save_raw_and_codec2(f, entries, outroot / base, codec2_bin, codec2_outdir, conn)

    conn.close()
    print("All done!")

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube playlist → word-level .raw + IPA + features (NO .bit decoding)")
    parser.add_argument("--playlist", required=True, help="YouTube playlist URL")
    parser.add_argument("--outdir", default="./output_words", help="Output directory")
    parser.add_argument("--tmpdir", default="./tmp_work", help="Temporary directory")
    parser.add_argument("--model", default="tiny", help="Whisper model")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--codec2_bin", required=True, help="Path to c2enc binary")
    args = parser.parse_args()

    process_playlist(args.playlist, Path(args.outdir), Path(args.tmpdir), Path(args.codec2_bin),
                     model=args.model, language=args.language, device=args.device)
