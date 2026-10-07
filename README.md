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

## Downloading media locally on Windows

YouTube blocks downloads from cloud IPs, so fetch the media on your own machine.
`fetch_sources.sh` runs unchanged in Git Bash (same `sources.txt`, same quality filters).

```powershell
winget install --id Git.Git -e
winget install --id yt-dlp.yt-dlp -e
winget install --id Gyan.FFmpeg -e
winget install --id OpenJS.NodeJS.LTS -e
# open a NEW PowerShell window so PATH updates, then:
git clone -b main-edit-pipeline https://github.com/leteygety/max-verstappen-edit.git
cd max-verstappen-edit
& "C:\Program Files\Git\bin\bash.exe" edit/fetch_sources.sh
```

Output lands in `media\clips\*.mkv` and `media\music\*.wav`; any `FAILED: <name>` line
means every candidate for that entry was rejected (the rest still download).
