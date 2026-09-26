# The Buried Vault — channel trailer

A 20-second, 1920x1080 paper-cut diorama trailer styled after the channel logo and banner (`assets/`), with synthesized audio. It is rendered entirely in code with numpy, Pillow, scipy and ffmpeg.

The camera ends by pulling back until the rendered bronze vault door lines up with the door in `assets/logo.jpg`, then dissolves into the logo.

```
pip install numpy pillow scipy imageio-ffmpeg
python render.py the_buried_vault.mp4      # full render (~4 min on 4 cores)
python render.py --preview                 # contact sheet + stills in preview/
```

## 35-second channel-page trailer

`trailer35.py` renders `the_buried_vault_trailer_35s.mp4` (1920x1080, 35 s, with audio). It is one continuous dolly:

1. The vault door grinds open.
2. The camera moves down a corridor past four alcoves, each labelled with carved text: LOST TREASURE, ANCIENT RUINS, UNSOLVED MYSTERIES and LEGENDARY ARTIFACTS.
3. It arrives in a torchlit chamber. There, the logo lettering carves into the wall above an open chest, then "New episodes every week" and a SUBSCRIBE sigil that pulses once.

The labels use Cinzel (`assets/fonts/Cinzel.ttf`, SIL Open Font License, see `assets/fonts/OFL.txt`).

```
python trailer35.py the_buried_vault_trailer_35s.mp4     # ~7 min on 4 cores
python trailer35.py --preview [times...]                 # contact sheet in preview/
```
