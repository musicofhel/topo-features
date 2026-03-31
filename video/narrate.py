"""
Generate narration audio from ElevenLabs, time-sync to video, and mux.

Usage:
    python video/narrate.py                     # generate + sync + mux
    python video/narrate.py --generate-only     # just generate TTS (no sync/mux)
    python video/narrate.py --sync-only         # just sync + mux (TTS already generated)
    python video/narrate.py --act 3             # regenerate only Act 3

Requires:
    ELEVENLABS_API_KEY env var (or pass --api-key)
    ffmpeg + ffprobe on PATH
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from elevenlabs import ElevenLabs

# ── Config ─────────────────────────────────────────────────────────────
VOICE_ID = "iP95p4xoKVk53GoZ742B"  # Tom — Balanced, Clean and Approachable
MODEL_ID = "eleven_multilingual_v2"
VOICE_SETTINGS = {
    "stability": 0.50,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True,
}

ACTS = {
    1: {"duration": 35.0, "file": "act1.txt"},
    2: {"duration": 30.0, "file": "act2.txt"},
    3: {"duration": 50.0, "file": "act3.txt"},
    4: {"duration": 30.0, "file": "act4.txt"},
    5: {"duration": 23.0, "file": "act5.txt"},
    6: {"duration": 12.0, "file": "act6.txt"},
}

SEGMENTS_DIR = Path("video/narration_segments")
OUTPUT_DIR = Path("video/output")


def generate_tts(client, voice_id, act_num):
    """Generate TTS for a single act, save as MP3."""
    text_file = SEGMENTS_DIR / ACTS[act_num]["file"]
    text = text_file.read_text().strip()

    print(f"  Act {act_num}: generating TTS ({len(text.split())} words)...")

    audio_iter = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=MODEL_ID,
        voice_settings=VOICE_SETTINGS,
        output_format="mp3_44100_128",
    )

    out_path = OUTPUT_DIR / f"act{act_num}.mp3"
    with open(out_path, "wb") as f:
        for chunk in audio_iter:
            f.write(chunk)

    duration = get_duration(out_path)
    target = ACTS[act_num]["duration"]
    ratio = duration / target
    print(f"    → {out_path} ({duration:.1f}s, target {target:.0f}s, ratio {ratio:.2f}x)")
    return out_path


def get_duration(path):
    """Get audio duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def time_stretch(input_path, target_duration, output_path):
    """Time-stretch audio to exact target duration using atempo filter.

    atempo only accepts values in [0.5, 100.0], so we chain multiple
    filters if needed (e.g., 0.4x = atempo=0.5,atempo=0.8).
    """
    actual = get_duration(input_path)
    ratio = actual / target_duration  # >1 means too long → speed up

    if abs(ratio - 1.0) < 0.02:
        # Within 2% — just copy and pad/trim
        print(f"    Ratio {ratio:.3f} — within tolerance, padding/trimming to {target_duration:.1f}s")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", f"apad=whole_dur={target_duration}",
            "-t", str(target_duration),
            "-c:a", "libmp3lame", "-b:a", "128k",
            str(output_path),
        ], capture_output=True, check=True)
        return

    # Build atempo filter chain (each filter limited to [0.5, 100.0])
    filters = []
    remaining = ratio
    while remaining > 100.0:
        filters.append("atempo=100.0")
        remaining /= 100.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")

    filter_str = ",".join(filters)
    print(f"    Stretching {actual:.1f}s → {target_duration:.1f}s (atempo chain: {filter_str})")

    subprocess.run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", f"{filter_str},apad=whole_dur={target_duration}",
        "-t", str(target_duration),
        "-c:a", "libmp3lame", "-b:a", "128k",
        str(output_path),
    ], capture_output=True, check=True)

    final_dur = get_duration(output_path)
    print(f"    Final: {final_dur:.2f}s (target: {target_duration:.1f}s)")


def concat_and_mux():
    """Concat synced segments and mux with silent video."""
    synced_files = [OUTPUT_DIR / f"act{i}_synced.mp3" for i in range(1, 7)]
    missing = [f for f in synced_files if not f.exists()]
    if missing:
        print(f"ERROR: Missing synced files: {missing}")
        sys.exit(1)

    # Concat audio segments
    narration_path = OUTPUT_DIR / "narration_full.mp3"
    list_file = OUTPUT_DIR / "concat_list.txt"
    with open(list_file, "w") as f:
        for sf in synced_files:
            f.write(f"file '{sf.resolve()}'\n")

    print(f"\nConcatenating → {narration_path}")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:a", "libmp3lame", "-b:a", "128k",
        str(narration_path),
    ], capture_output=True, check=True)

    total_dur = get_duration(narration_path)
    print(f"  Total narration: {total_dur:.1f}s (target: 180.0s)")

    # Mux with silent video
    silent_video = OUTPUT_DIR / "topofeatures_silent.mp4"
    final_video = OUTPUT_DIR / "topofeatures_explainer.mp4"

    if not silent_video.exists():
        print(f"\nWARNING: {silent_video} not found — run 'python video/render.py' first")
        print(f"Narration saved to {narration_path}")
        return

    print(f"Muxing → {final_video}")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(silent_video),
        "-i", str(narration_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(final_video),
    ], capture_output=True, check=True)

    final_dur = get_duration(final_video)
    print(f"  Final video: {final_dur:.1f}s")
    print(f"\n  {final_video}")


def main():
    parser = argparse.ArgumentParser(description="Generate + sync narration audio")
    parser.add_argument("--act", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="Generate only this act")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate TTS, skip sync/mux")
    parser.add_argument("--sync-only", action="store_true",
                        help="Only sync + mux (TTS already generated)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="ElevenLabs API key (or set ELEVENLABS_API_KEY)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    acts_to_process = [args.act] if args.act else list(range(1, 7))

    # ── Generate TTS ───────────────────────────────────────────────────
    if not args.sync_only:
        api_key = args.api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            print("ERROR: Set ELEVENLABS_API_KEY or pass --api-key")
            sys.exit(1)

        client = ElevenLabs(api_key=api_key)
        voice_id = VOICE_ID
        print(f"Voice: Tom ({voice_id})\n")

        print("Generating TTS:")
        for act_num in acts_to_process:
            generate_tts(client, voice_id, act_num)

    if args.generate_only:
        print("\nDone (generate-only). Run with --sync-only to sync + mux.")
        return

    # ── Time-sync each segment ─────────────────────────────────────────
    print("\nTime-syncing segments:")
    for act_num in acts_to_process:
        raw = OUTPUT_DIR / f"act{act_num}.mp3"
        synced = OUTPUT_DIR / f"act{act_num}_synced.mp3"
        if not raw.exists():
            print(f"  Act {act_num}: SKIP (no raw file)")
            continue
        target = ACTS[act_num]["duration"]
        print(f"  Act {act_num}:")
        time_stretch(raw, target, synced)

    # ── Concat + mux ──────────────────────────────────────────────────
    concat_and_mux()


if __name__ == "__main__":
    main()
