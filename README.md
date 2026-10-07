# max-verstappen-edit
Max Verstappen Cinematic Tik Tok edit

A scripted editing pipeline for a dark, beat-synced 9:16 F1 edit (see `edit/STYLE.md`).

```
pip install yt-dlp opencv-python-headless numpy numba librosa soundfile
edit/fetch_sources.sh                       # footage + music -> media/ (gitignored)
edit/scan.py media/clips media/scan         # contact sheets + scene cuts for shot picking
edit/analyze_audio.py media/music/x.wav media/beats.json
edit/render.py edit/timeline.json out.mp4 --preview   # half-res check
edit/render.py edit/timeline.json out.mp4             # 1080x1920 final
```

`render.py` decodes each source once, applies time remap (speed ramps, freezes,
reverse, stutter, motion blur), 9:16 reframe + zoom + shake in a single resample,
grade, effects (dip, flash, bloom, whip, zoom transition, directional/radial blur,
RGB split, glitch/tearing, strobe, ghost, invert/threshold impact frames), grain,
and encodes once (x264 High, CRF 15, bt709, AAC 320k).

Timeline times accept seconds, beat refs (`"b12.5"`) or bass-hit refs (`"h7"`).
