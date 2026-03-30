# ElevenLabs TTS Guide

## Voice
Tom — Balanced, Clean and Approachable

## Model
eleven_multilingual_v2

## Settings
- Stability: 0.50
- Similarity boost: 0.75
- Style: 0.0
- Speaker boost: true

## Target durations per segment
| File | Duration |
|------|----------|
| act1.txt | 35s |
| act2.txt | 30s |
| act3.txt | 50s |
| act4.txt | 30s |
| act5.txt | 23s |
| act6.txt | 12s |

If a segment comes back too short, increase stability to 0.60.
If too long, decrease to 0.40. Adjust per-segment, not globally.

## Stitch
```bash
ffmpeg -i act1.mp3 -i act2.mp3 -i act3.mp3 -i act4.mp3 -i act5.mp3 -i act6.mp3 \
  -filter_complex "[0][1][2][3][4][5]concat=n=6:v=0:a=1" \
  narration_full.mp3
```

## Combine with video
```bash
python video/render.py

ffmpeg -i video/output/topofeatures_silent.mp4 \
  -i narration_full.mp3 \
  -c:v copy -c:a aac -shortest \
  video/output/topofeatures_explainer.mp4
```
