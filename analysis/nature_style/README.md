# Nature figure style hook

`sitecustomize.py` restyles every matplotlib figure at save time to the Nature Communications
figure specification: Arial-metric sans-serif (Liberation Sans), lowercase bold panel labels at 8 pt,
all other text 5.5–7 pt non-bold (sizes referenced to the 183 mm double-column width, with the
tight bounding box taken into account), closed frame with outward major ticks on the left and bottom axes only (no minor ticks), 0.5 pt axes and ticks, and thinner data strokes (lines ≤1 pt, marker and bar edges ≤0.6 pt). Data and layout are untouched.

Usage: `PYTHONPATH=analysis/nature_style python3 <figure_script>.py`

`meas.py <fig.pdf>` reports the font sizes (scaled to 183 mm), font families and bold glyphs of a PDF.
