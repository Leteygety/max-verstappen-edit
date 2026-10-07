# Style notes from the reference edit

Measured frame by frame from the 11 s reference (1024×576, 29.97 fps, music ≈117 BPM).

| Trait | Measured | Renderer mapping |
|---|---|---|
| Cut rate | ~1 shot per beat (0.4–1.2 s), half-beat cuts in bursts | shots placed on `b<n>` beat refs |
| Signature transition | luma crashes to ~5–15/255 for 2–4 frames between shots, highlights survive | `dip` fx (`env: tri`) across each cut |
| Exposure blooms | whites to 170–200/255 for 0.3–0.9 s, saturation drops | `bloom` fx + `bleach` grade |
| Saturation | mostly low (mean chroma 10–30), Red Bull livery keeps colour | hue-selective grade (`base`/`cold`) |
| Colour pop | one teal/pink section late in the edit (chroma ~60) | `teal` grade, used once |
| Ending | near-monochrome, high-contrast close-ups of Max | `bw` grade |
| Strobe/glitch bursts | 0.4–0.8 s of per-frame changes (diff 60–120) on 1/8 notes | `glitch` + `strobe` + `rgb jitter` |
| Motion | heavy directional blur on passes, radial blur on punch-ins | `whip`, `zoomtrans`, speed-ramp motion blur |

Pacing rules used when cutting:
- Open on one held, dark car shot with slow push-in; first hard cut on first bass drop.
- Alternate car → Max → onboard → trackside → crowd → detail; never three of one kind in a row.
- Ramp speed up into an overtake, drop to 0.3–0.5× right before a big hit, cut on the hit.
- Flashes only on the biggest hits (drops / section changes), not every beat.
- Shake intensity follows bass-hit strength; light shake everywhere else.
