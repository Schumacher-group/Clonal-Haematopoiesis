import math
import random
import os
import numpy as np
from PIL import Image

# --- Parameters ---
SLIDE_W, SLIDE_H = 1920, 1080   # output resolution (16:9)
GRID_COLS = 600                  # pixels wide
GRID_ROWS = 450                  # pixels tall — 270,000 cells total
S = 0.55                         # fitness coefficient
T_MAX = 24                     # number of slides (t=0 to t=T_MAX)

# Colourblind-safe Wong palette
WT_COLOUR  = (86,   180, 233)    # teal      — wild-type
MUT_COLOUR = (213,  94,   0)    # vermillion — mutant

OUT_DIR = "pixel_backgrounds"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Setup ---
total_cells = GRID_COLS * GRID_ROWS

# Shuffle pixel indices once — defines the order pixels flip from WT to mutant
rng = random.Random(99)
flip_order = list(range(total_cells))
rng.shuffle(flip_order)

# --- Generate ---
state    = np.zeros(total_cells, dtype=np.uint8)  # 0 = WT, 1 = mutant
prev_mut = 0

for t in range(T_MAX + 1):
    n_mut = min(round(math.exp(S * t)), total_cells)

    # Flip any new pixels to mutant
    for i in range(prev_mut, n_mut):
        state[flip_order[i]] = 1
    prev_mut = n_mut

    # Build colour array at grid resolution
    grid = state.reshape(GRID_ROWS, GRID_COLS)
    rgb  = np.where(grid[..., None] == 0, WT_COLOUR, MUT_COLOUR).astype(np.uint8)

    # Stretch to fill full slide using nearest-neighbour (keeps crisp pixel edges)
    img = Image.fromarray(rgb, "RGB").resize((SLIDE_W, SLIDE_H), Image.NEAREST)

    img.save(
        os.path.join(OUT_DIR, f"slide_t{t:02d}.png"), "PNG", optimize=True
    )
    print(f"t={t:2d}  mutant={n_mut:>8,} ({n_mut/total_cells*100:5.1f}%)  wt={total_cells-n_mut:>8,}")

print(f"\nDone — {T_MAX + 1} images saved to '{OUT_DIR}/'")