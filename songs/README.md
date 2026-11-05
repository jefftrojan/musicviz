# Study audio and licensing

The primary experiment uses three instrumental tracks by Jason Shaw
(Audionautix), each released under Creative Commons Attribution 3.0. Required
credit: **“music by audionautix.com.”** The repository retains the downloaded
source files and produces fixed, peak-normalized 8-second PCM excerpts with
`python prepare_songs.py`.

| Study ID | Source title | Excerpt | Intended profile | Wikimedia Commons source |
|---|---|---:|---|---|
| `calm_blue_lake` | Calm Blue Lake | 48–56 s | low envelope variation | <https://commons.wikimedia.org/wiki/File:Audionautix-com-ccby-calmbluelake.mp3> |
| `acoustic_shuffle` | Acoustic Shuffle | 50–58 s | medium envelope variation | <https://commons.wikimedia.org/wiki/File:Audionautix-com-ccby-acousticshuffle.mp3> |
| `heavy_drums_bass` | Heavy Drums Bass | 38–46 s | high onset variation | <https://commons.wikimedia.org/wiki/File:Audionautix-com-ccby-heavydrumsbass.mp3> |

All three Commons records identify Jason Shaw/Audionautix as creator, specify
CC BY 3.0, and give the same attribution requirement. Source SHA-256 checksums
are frozen in `prepare_songs.py`. Files under `songs/candidates/` were screened
but are not part of the study.

The excerpt windows were selected using audio-only RMS and onset-envelope
variation before any generator outcomes were inspected. Peak normalization to
-1 dBFS equalizes playback headroom without changing within-excerpt dynamics.
