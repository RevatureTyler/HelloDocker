# The Buried Vault — channel trailer

A 20-second, 1920x1080 paper-cut diorama trailer styled after the channel logo and banner (`assets/`), with synthesized audio. It is rendered entirely in code with numpy, Pillow, scipy and ffmpeg.

The camera ends by pulling back until the rendered bronze vault door lines up with the door in `assets/logo.jpg`, then dissolves into the logo.

```
pip install numpy pillow scipy imageio-ffmpeg
python render.py the_buried_vault.mp4      # full render (~4 min on 4 cores)
python render.py --preview                 # contact sheet + stills in preview/
```
