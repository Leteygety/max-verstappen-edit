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

## 16:9 video edits (reference style)

The horizontal edits are cut from the real video (`media/clips`, 1080p50 / 4K sources), not
from stills: 1920x1080, 30 fps, two acts (cars -> Max), one teal insert, black-and-white finale.

```
edit/scan_motion.py media/clips/<clip>.mkv --json media/work/scan/<clip>.json   # cuts + in-shot motion
edit/strips.py media/work/sheets/cand.jpg <clip>:<t0>:<t1> ...                     # timecoded strips
edit/build_video_timeline.py edit/cuts_video/v1.json edit/timeline_v1_16x9.json    # cut list -> timeline
edit/render.py edit/timeline_v1_16x9.json media/work/preview/v1.mp4 --preview
edit/render.py edit/timeline_v1_16x9.json media/output/verstappen_edit_v1_16x9.mp4
edit/verify_render.py edit/timeline_v1_16x9.json media/output/verstappen_edit_v1_16x9.mp4 media/work/v1_qc.jpg
edit/style_metrics.py media/output/verstappen_edit_v1_16x9.mp4 edit/timeline_v1_16x9.json --per-shot
```

A video cut list places source moments on the song's fixed beat grid
(`[beat, source, in-point, options]`, beats are song beats such as `"b15"`); the builder checks
every shot against the source's scene cuts (from `scan_motion.py`) and keeps the reframed window
clear of burned-in broadcast graphics (`keepout` boxes). `render.py --no-fx` renders without the
effect list, which `verify_render.py` uses to confirm every cut sits on its beat frame.

## Video fragments (local)

`edit/fetch_fragments.sh` downloads the 6 source videos behind `media/selected` once (best
quality, up to 4K) and cuts 4-16 s fragments around every approved moment into
`media/fragments/<name>__<start>-<end>.mp4` (<=1080p, CRF 16, original fps, no audio).
Run it locally (Git Bash on Windows): `bash edit/fetch_fragments.sh`.
