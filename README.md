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

## Still-frame edits (media/selected)

The two TikTok edits are cut from the approved still pool in `media/selected/` (640x360 JPGs,
never modified). Every frame has a burned-in timecode box, so each one gets a non-destructive
9:16 crop in `edit/framing.json` that keeps it (and other broadcast overlays) out of frame.

```
edit/check_framing.py edit/framing.json media/selected media/work/sheets/crops.jpg   # crops clear of text?
edit/upscale.py media/selected media/work/sr/x4 --weights realesr-general-x4v3.pth   # 4x Real-ESRGAN copies
edit/analyze_audio.py "media/music/<track>" media/work/v1/beats.json --fixed-tempo     # constant beat grid
edit/build_timeline.py edit/cuts/v1.json edit/timeline_v1.json                       # cut list -> timeline
edit/render.py edit/timeline_v1.json media/output/verstappen_edit_v1.mp4
edit/verify_render.py edit/timeline_v1.json media/output/verstappen_edit_v1.mp4 media/work/sheets/v1.jpg
```

`edit/cuts/v1.json` (FUNK CRIMINAL 2, slowed) and `edit/cuts/v2.json` (FALL FROM THE SKY PT. 2)
are separate cut lists built on each track's own beat grid and structure; see their `_doc` notes.
Audio is loudness-normalised per segment (`loudness`, default -10 LUFS, -1 dBTP).
