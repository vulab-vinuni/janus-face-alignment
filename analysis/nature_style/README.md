# Nature figure style hook

`sitecustomize.py` restyles every matplotlib figure at save time to the Nature Communications
figure specification: Arial-metric sans-serif (Liberation Sans), lowercase bold panel labels at 8 pt,
all other text 5.5–7 pt non-bold (sizes referenced to the 183 mm double-column width, with the
tight bounding box taken into account), and 0.6–0.75 pt axes and ticks. Data and layout are untouched.

Usage: `PYTHONPATH=analysis/nature_style python3 <figure_script>.py`

`meas.py <fig.pdf>` reports the font sizes (scaled to 183 mm), font families and bold glyphs of a PDF.
