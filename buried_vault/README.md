# The Buried Vault — channel trailer

A 20-second, 1920x1080 (16:9) paper-cut diorama trailer with synthesized ambient audio. It is rendered entirely in code with numpy, Pillow and ffmpeg.

```
pip install numpy pillow scipy imageio-ffmpeg
python render.py the_buried_vault.mp4      # full render (~3 min on 4 cores)
python render.py --preview                 # contact sheet + stills in preview/
```

The font is Cinzel (SIL Open Font License), from google/fonts.
