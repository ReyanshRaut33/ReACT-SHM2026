import streamlit as st
# Step-by-step ReACT-SHM concept simulation for Google Colab
# Cleaned version:
# - only ONE top surface line
# - AE sensors fully above the concrete
# - FBG sensors on the surface
# - nanobots vertical with nozzle entering concrete
# - more space below title
# - step 3 sends acoustic waves to all AE sensors

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.font_manager as fm
import requests

# -----------------------------
# Load Bricolage Grotesque font
# -----------------------------
font_url = "https://raw.githubusercontent.com/google/fonts/main/ofl/bricolagegrotesque/BricolageGrotesque%5Bopsz%2Cwdth%2Cwght%5D.ttf"
font_path = "BricolageGrotesque.ttf"

r = requests.get(font_url)
r.raise_for_status()

with open(font_path, "wb") as f:
    f.write(r.content)

fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

# -----------------------------
# Global scene settings
# -----------------------------
W, H = 16, 10

# Concrete block bounds
CONCRETE_X0 = 1.0
CONCRETE_Y0 = 2.0   # moved down to create more space under title
CONCRETE_W = 14.0
CONCRETE_H = 6.8

# TRUE top surface = top edge of concrete
surface_y = CONCRETE_Y0

# Sensors on surface
ae_positions = [(3.0, surface_y), (8.0, surface_y), (13.0, surface_y)]
fbg_positions = [(5.3, surface_y), (10.7, surface_y)]

# Nanobots above surface
nanobot_positions = [(4.5, surface_y - 0.95), (11.5, surface_y - 0.95)]

# Crack location
crack_anchor = (4.2, 7.2)

selected_ae = ae_positions[0]
selected_nanobot = nanobot_positions[0]

STEP_NAMES = [
    "1. Initial setup",
    "2. Crack forms",
    "3. Acoustic wave reaches AE sensors",
    "4. AE sensor signals nanobot",
    "5. Nanobot releases nanoparticles",
    "6. Ultrasonic guidance to crack",
    "7. Crack heals"
]

current_step = 0

# -----------------------------
# Helpers
# -----------------------------
def ax_text(ax, *args, **kwargs):
    kwargs.setdefault("fontproperties", font_prop)
    return ax.text(*args, **kwargs)

def draw_concrete(ax):
    # Draw concrete block ONCE. Its top border is the only surface line.
    concrete = Rectangle(
        (CONCRETE_X0, CONCRETE_Y0),
        CONCRETE_W,
        CONCRETE_H,
        facecolor="#d9d9d9",
        edgecolor="black",
        linewidth=1.6
    )
    ax.add_patch(concrete)

    # light texture
    rng = np.random.default_rng(4)
    for _ in range(75):
        x = rng.uniform(CONCRETE_X0 + 0.2, CONCRETE_X0 + CONCRETE_W - 0.2)
        y = rng.uniform(CONCRETE_Y0 + 0.2, CONCRETE_Y0 + CONCRETE_H - 0.2)
        r = rng.uniform(0.03, 0.07)
        ax.add_patch(Circle((x, y), r, facecolor="#bfbfbf", edgecolor="none", alpha=0.22))

    ax_text(
        ax,
        CONCRETE_X0 + CONCRETE_W / 2,
        CONCRETE_Y0 + CONCRETE_H / 2,
        "Concrete",
        ha="center",
        va="center",
        fontsize=14,
        alpha=0.28
    )

def draw_ae_sensor(ax, x, y, active=False):
    color = "#ff8c00" if active else "#666666"

    # Entire AE body ABOVE the concrete surface
    body_w, body_h = 0.56, 0.22
    body_bottom = y - body_h
    body = Rectangle(
        (x - body_w / 2, body_bottom),
        body_w, body_h,
        facecolor=color,
        edgecolor="black",
        linewidth=1.2
    )

    # tiny foot just touching the surface from above
    foot = Rectangle(
        (x - 0.045, y),
        0.09, 0.07,
        facecolor=color,
        edgecolor="black",
        linewidth=1.0
    )

    ax.add_patch(body)
    ax.add_patch(foot)
    ax_text(ax, x, body_bottom - 0.12, "AE", ha="center", va="center", fontsize=9, weight="bold")

def draw_fbg_sensor(ax, x, y):
    # Draw exactly on the surface
    ax.plot([x - 0.72, x + 0.72], [y, y], color="#1f77b4", linewidth=3)
    ax_text(ax, x, y + 0.38, "FBG", ha="center", va="center", fontsize=9, weight="bold")

def draw_nanobot(ax, x, y, active=False):
    body_color = "#7a3db8" if active else "#444444"
    nozzle_color = "#a64dff" if active else "#666666"

    body = Rectangle(
        (x - 0.22, y), 0.44, 0.9,
        facecolor=body_color, edgecolor="black", linewidth=1.2
    )
    cap = Rectangle(
        (x - 0.26, y - 0.08), 0.52, 0.08,
        facecolor=body_color, edgecolor="black", linewidth=1.0
    )
    nozzle = Rectangle(
        (x - 0.06, y + 0.9), 0.12, 0.42,
        facecolor=nozzle_color, edgecolor="black", linewidth=1.0
    )

    ax.add_patch(body)
    ax.add_patch(cap)
    ax.add_patch(nozzle)
    ax_text(ax, x, y - 0.24, "Nanobot", ha="center", va="center", fontsize=8.5, weight="bold")

def crack_points(scale=1.0):
    x0, y0 = crack_anchor
    return np.array([
        [x0, y0],
        [x0-0.18*scale, y0-0.38*scale],
        [x0-0.34*scale, y0-0.75*scale],
        [x0-0.22*scale, y0-1.10*scale],
        [x0-0.48*scale, y0-1.48*scale],
        [x0-0.22*scale, y0-1.85*scale],
        [x0-0.56*scale, y0-2.25*scale],
    ])

def draw_crack(ax, alpha=1.0, heal_fraction=0.0):
    pts = crack_points()
    n = max(2, int(len(pts) * (1 - 0.75 * heal_fraction)))
    pts2 = pts[:n]

    ax.plot(
        pts2[:, 0], pts2[:, 1],
        color="#3b1f0b",
        linewidth=3.0 * (1 - heal_fraction * 0.5),
        alpha=alpha
    )

    for i, (x, y) in enumerate(pts2[1:-1], start=1):
        side = -1 if i % 2 == 0 else 1
        ax.plot(
            [x, x + 0.18 * side * (1 - heal_fraction)],
            [y, y - 0.18],
            color="#3b1f0b",
            linewidth=1.5,
            alpha=alpha
        )

def draw_acoustic_wave(ax, start, end, progress=1.0, color="#ff8c00", label=None):
    sx, sy = start
    ex, ey = end

    for k in range(3):
        t = max(0.0, min(1.0, progress - k * 0.12))
        if t <= 0:
            continue
        x = sx + (ex - sx) * t
        y = sy + (ey - sy) * t
        r = 0.18 + 0.22 * t + 0.05 * k
        ax.add_patch(Circle((x, y), r, fill=False, edgecolor=color, linewidth=1.6, alpha=0.9 - 0.22 * k))

    if label and progress > 0.95:
        ax_text(ax, (sx + ex) / 2 - 0.2, (sy + ey) / 2 - 0.5, label, fontsize=10, color=color)

def draw_signal_arrow(ax, start, end, color="red", label=None):
    arr = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=16, linewidth=2.0, color=color)
    ax.add_patch(arr)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax_text(ax, mx, my - 0.32, label, color=color, fontsize=10, ha="center")

def draw_particles(ax, source, target=None, count=16, progress=0.0, curved=False):
    rng = np.random.default_rng(10)
    sx, sy = source

    if not curved or target is None:
        for _ in range(count):
            dx = rng.uniform(-0.32, 0.32)
            dy = rng.uniform(0.0, 1.0 * progress)
            ax.add_patch(Circle((sx + dx, sy + dy), 0.04, facecolor="gold", edgecolor="black", linewidth=0.3))
        return

    tx, ty = target
    c1 = (sx + 0.15, sy + 2.2)
    c2 = (tx + 1.05, ty - 1.8)
    path = Path([source, c1, c2, target], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])

    verts = path.interpolated(60).vertices
    ax.plot(verts[:, 0], verts[:, 1], linestyle="--", color="#7a3db8", linewidth=2.0, alpha=0.8)
    ax_text(ax, (sx + tx) / 2 + 0.35, (sy + ty) / 2 - 0.25, "Ultrasonic guidance", color="#7a3db8", fontsize=10)

    for i in range(count):
        t = max(0.0, min(1.0, progress - i * 0.03))
        if t <= 0:
            continue
        x = ((1 - t) ** 3) * source[0] + 3 * ((1 - t) ** 2) * t * c1[0] + 3 * (1 - t) * (t ** 2) * c2[0] + (t ** 3) * target[0]
        y = ((1 - t) ** 3) * source[1] + 3 * ((1 - t) ** 2) * t * c1[1] + 3 * (1 - t) * (t ** 2) * c2[1] + (t ** 3) * target[1]
        ax.add_patch(Circle((x, y), 0.045, facecolor="gold", edgecolor="black", linewidth=0.3))

def draw_healing_cloud(ax, center, alpha=1.0):
    cx, cy = center
    rng = np.random.default_rng(22)
    for _ in range(18):
        dx = rng.uniform(-0.35, 0.25)
        dy = rng.uniform(-0.15, 0.25)
        r = rng.uniform(0.05, 0.11)
        ax.add_patch(Circle((cx + dx, cy + dy), r, facecolor="#5dade2", edgecolor="none", alpha=0.25 * alpha))
    ax_text(ax, cx + 0.95, cy - 0.22, "C-S-H forms", color="#1f77b4", fontsize=10)

def setup_ax():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax




def render_scene(step_idx, ax):
    draw_concrete(ax)

    for pos in ae_positions:
        draw_ae_sensor(ax, *pos, active=(step_idx >= 2))
    for pos in fbg_positions:
        draw_fbg_sensor(ax, *pos)
    for pos in nanobot_positions:
        draw_nanobot(ax, *pos, active=(step_idx >= 4 and pos == selected_nanobot))

    if step_idx >= 1:
        if step_idx < 6:
            draw_crack(ax, alpha=1.0, heal_fraction=0.0)
        else:
            draw_healing_cloud(ax, crack_anchor, alpha=1.0)
            draw_crack(ax, alpha=0.45, heal_fraction=0.85)

    if step_idx == 2:
        for i, ae in enumerate(ae_positions):
            draw_acoustic_wave(ax, crack_anchor, ae, progress=1.0, label="Acoustic wave" if i == 0 else None)
            
    elif step_idx == 3:
        for ae in ae_positions:
            draw_acoustic_wave(ax, crack_anchor, ae, progress=1.0)

        signal_start = (selected_ae[0], selected_ae[1] - 0.33)
        signal_end = (selected_nanobot[0], selected_nanobot[1] + 0.28)
        draw_signal_arrow(ax, signal_start, signal_end, color="red", label="Detection signal")

    elif step_idx == 4:
        signal_start = (selected_ae[0], selected_ae[1] - 0.33)
        signal_end = (selected_nanobot[0], selected_nanobot[1] + 0.28)
        draw_signal_arrow(ax, signal_start, signal_end, color="red")

        nozzle_source = (selected_nanobot[0], selected_nanobot[1] + 1.3)
        draw_particles(ax, nozzle_source, progress=0.95, curved=False)
        ax_text(ax, nozzle_source[0] + 1.0, nozzle_source[1] + 0.72, "Nanoparticles released", fontsize=10)

    elif step_idx == 5:
        nozzle_source = (selected_nanobot[0], selected_nanobot[1] + 1.3)
        draw_particles(ax, nozzle_source, target=(crack_anchor[0] - 0.1, crack_anchor[1] - 0.25), progress=1.0, curved=True)

    elif step_idx == 6:
        nozzle_source = (selected_nanobot[0], selected_nanobot[1] + 1.3)
        draw_particles(ax, nozzle_source, target=(crack_anchor[0] - 0.1, crack_anchor[1] - 0.25), progress=0.82, curved=True)
        draw_healing_cloud(ax, crack_anchor, alpha=1.0)



import streamlit as st

st.title("ReACT-SHM Simulation")

step = st.slider("Select Step", 0, len(STEP_NAMES) - 1, 0)

fig, ax = setup_ax()
render_scene(step, ax)

st.pyplot(fig)
