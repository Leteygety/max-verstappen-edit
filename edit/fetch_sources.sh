#!/usr/bin/env bash
# Download source footage + music listed in sources.txt into media/ (not committed).
# Video: best stream up to 2160p (skips anything under 720p); music: best audio as WAV.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$here/../media}"
mkdir -p "$out/clips" "$out/music"
grep -v '^\s*#' "$here/sources.txt" | grep -v '^\s*$' | while IFS='|' read -r name target; do
  name="$(echo "$name" | xargs)"; target="$(echo "$target" | xargs)"
  if [ "$name" = music ]; then
    yt-dlp -x --audio-format wav -o "$out/music/%(title).80s.%(ext)s" "$target" || echo "FAILED: $name"
  else
    yt-dlp -f "bv*[height<=2160][height>=720]+ba/b[height>=720]" --merge-output-format mkv \
      -o "$out/clips/${name}__%(id)s.%(ext)s" "$target" || echo "FAILED: $name"
  fi
done
ls -la "$out/clips" "$out/music"
