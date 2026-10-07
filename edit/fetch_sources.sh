#!/usr/bin/env bash
# Download source footage + music listed in sources.txt into media/ (not committed).
# Video: best stream up to 2160p (skips anything under 720p); music: best audio as WAV.
#
# Each entry may list alternates separated by "||". Every ytsearchN target is tried
# candidate by candidate until one passes the quality/length filters, then the next
# alternate is only used if the whole search produced nothing.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$here/../media}"
mkdir -p "$out/clips" "$out/music"

# YouTube extraction needs a JS runtime; use node when there is no deno.
yt=(yt-dlp --no-playlist --ignore-errors --no-overwrites --max-downloads 1)
if ! command -v deno >/dev/null && command -v node >/dev/null; then
  yt+=(--js-runtimes "node:$(command -v node)")
fi

fetch() {  # name target
  local name="$1" target="$2" rc=0
  if [ "$name" = music ]; then
    "${yt[@]}" -x --audio-format wav --match-filter "duration < 600" \
      -o "$out/music/%(title).80s.%(ext)s" "$target" || rc=$?
    ls "$out/music/"*.wav >/dev/null 2>&1 || return 1
  else
    "${yt[@]}" -f "bv*[height<=2160][height>=720]+ba/b[height>=720]" \
      -S "res:2160,fps,vbr" --merge-output-format mkv \
      --match-filter "duration > 20 & duration < 1500" \
      -o "$out/clips/${name}__%(id)s.%(ext)s" "$target" || rc=$?
    ls "$out/clips/${name}__"*.mkv >/dev/null 2>&1 || return 1
  fi
  # 101 = stopped after --max-downloads, i.e. success
  [ "$rc" = 0 ] || [ "$rc" = 101 ] || [ "$rc" = 1 ]
}

grep -v '^\s*#' "$here/sources.txt" | grep -v '^\s*$' | while IFS='|' read -r name rest; do
  name="$(echo "$name" | xargs)"
  ok=0
  IFS=$'\n' read -r -d '' -a alts < <(echo "$rest" | sed 's/^|//; s/||/\n/g'; printf '\0') || true
  for target in "${alts[@]}"; do
    target="$(echo "$target" | xargs)"
    [ -n "$target" ] || continue
    echo "== $name <- $target"
    if fetch "$name" "$target" </dev/null; then ok=1; break; fi
    echo "   no usable result, trying next alternate"
  done
  [ "$ok" = 1 ] || echo "FAILED: $name"
done
ls -la "$out/clips" "$out/music"
