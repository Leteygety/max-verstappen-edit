#!/usr/bin/env bash
# Cut short, high-quality video fragments around the approved moments (edit/fragments.txt).
# Each source video is downloaded once in the best quality available (<=2160p, >=720p), or
# reused from media/clips or media/src; every range is then cut and re-encoded to <=1080p
# H.264 (CRF 16, original frame rate, no audio) into media/fragments/<name>__<start>-<end>.mp4,
# so a fragment's t=0 is <start> seconds into its source. Runs locally (Git Bash on Windows).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/.."
src="$root/media/src"; out="$root/media/fragments"
mkdir -p "$src" "$out"
yt=(yt-dlp --no-playlist)
if ! command -v deno >/dev/null && command -v node >/dev/null; then
  yt+=(--js-runtimes node)
fi
log="$out/sources.txt"
: > "$log"
tr -d '\r' < "$here/fragments.txt" | grep -v '^\s*#' | grep -v '^\s*$' | while IFS='|' read -r name id ranges; do
  name="$(echo "$name" | xargs)"; id="$(echo "$id" | xargs)"
  f="$(ls "$root/media/clips/${name}__${id}".* "$src/${name}__${id}".* 2>/dev/null | grep -v '\.part$' | head -1 || true)"
  if [ -z "$f" ]; then
    echo "== downloading $name ($id)"
    "${yt[@]}" -f "bv*[height<=2160][height>=720]+ba/b[height>=720]" -S "res:2160,fps" \
      --merge-output-format mkv -o "$src/${name}__${id}.%(ext)s" \
      "https://www.youtube.com/watch?v=$id" </dev/null || true
    f="$(ls "$src/${name}__${id}".mkv 2>/dev/null || true)"
    if [ -z "$f" ]; then echo "FAILED: $name ($id)"; echo "$name $id FAILED" >> "$log"; continue; fi
  fi
  info="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate \
          -of csv=p=0 "$f" </dev/null)"
  echo "== $name: source $info"
  echo "$name $id $info" >> "$log"
  for r in $ranges; do
    a="${r%-*}"; b="${r#*-}"
    o="$out/${name}__${a}-${b}.mp4"
    [ -s "$o" ] && continue
    d="$(awk -v a="$a" -v b="$b" 'BEGIN{printf "%.3f", b-a}')"
    ffmpeg -v error -y -ss "$a" -i "$f" -t "$d" -an \
      -vf "scale=-2:'min(1080,ih)':flags=lanczos" \
      -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p "$o" </dev/null || true
    got="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$o" </dev/null 2>/dev/null || true)"
    if awk -v g="${got:-0}" 'BEGIN{exit !(g+0 > 0.5)}'; then
      echo "   $name $a-$b ok (${got}s)"
    else
      rm -f "$o"; echo "   FAILED cut: $name $a-$b (outside the video?)"
    fi
  done
done
echo; cat "$log"; du -sh "$out"
