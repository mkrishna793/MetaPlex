"""
Generate publication-quality SVG visualizations for MetaPlex research.
"""
import json
import csv
import os
import math

RESULTS = r"D:\neurocartography_data\results"
OUTPUT = r"D:\MetaPlex\visuals"
os.makedirs(OUTPUT, exist_ok=True)

# Colors
C_BG = "#0d1117"
C_CARD = "#161b22"
C_TEXT = "#e6edf3"
C_ACCENT = "#58a6ff"
C_GREEN = "#3fb950"
C_ORANGE = "#d29922"
C_RED = "#f85149"
C_PURPLE = "#bc8cff"
C_CYAN = "#39d2c0"
C_PINK = "#f778ba"
PHASE_COLORS = ["#f85149","#d29922","#58a6ff","#bc8cff","#3fb950","#39d2c0"]
BENCH_COLORS = ["#f85149","#d29922","#58a6ff","#bc8cff","#3fb950","#39d2c0","#f778ba"]

# Load data
with open(os.path.join(RESULTS, "phase_validation.json")) as f:
    pv = json.load(f)
stats = pv["stats"]
phases = pv["phases"]

# ==============================================================
# 1. Hub Stability SVG
# ==============================================================
def make_hub_stability_svg():
    W, H = 900, 400
    pad_l, pad_r, pad_t, pad_b = 60, 30, 40, 50
    gw = W - pad_l - pad_r
    gh = H - pad_t - pad_b

    vals = [s["hub_stability"] for s in stats]
    max_v = max(vals) * 1.2 if max(vals) > 0 else 0.05
    n = len(vals)

    def px(i): return pad_l + (i / (n - 1)) * gw
    def py(v): return pad_t + gh - (v / max_v) * gh

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    lines.append(f'<rect width="{W}" height="{H}" fill="{C_BG}" rx="12"/>')
    lines.append(f'<text x="{W//2}" y="28" fill="{C_TEXT}" font-size="16" font-weight="700" text-anchor="middle">Hub Stability Across Layers (Jaccard Top-100)</text>')

    # Phase bands
    for p in phases:
        x1 = px(p["start"])
        x2 = px(p["end"])
        lines.append(f'<rect x="{x1}" y="{pad_t}" width="{x2-x1}" height="{gh}" fill="{PHASE_COLORS[p["phase"]-1]}" opacity="0.08"/>')

    # Grid
    for i in range(5):
        v = max_v * i / 4
        y = py(v)
        lines.append(f'<line x1="{pad_l}" y1="{y}" x2="{W-pad_r}" y2="{y}" stroke="{C_TEXT}" stroke-opacity="0.1" stroke-dasharray="4,4"/>')
        lines.append(f'<text x="{pad_l-8}" y="{y+4}" fill="{C_TEXT}" font-size="10" text-anchor="end" opacity="0.6">{v:.3f}</text>')

    # X labels
    for i in range(0, n, 5):
        lines.append(f'<text x="{px(i)}" y="{H-10}" fill="{C_TEXT}" font-size="10" text-anchor="middle" opacity="0.6">L{i}</text>')

    # Line
    pts = " ".join(f"{px(i)},{py(v)}" for i, v in enumerate(vals))
    lines.append(f'<polyline points="{pts}" fill="none" stroke="{C_PURPLE}" stroke-width="2.5" stroke-linejoin="round"/>')
    for i, v in enumerate(vals):
        lines.append(f'<circle cx="{px(i)}" cy="{py(v)}" r="3" fill="{C_PURPLE}"/>')

    lines.append('</svg>')
    with open(os.path.join(OUTPUT, "hub_stability.svg"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  ✓ hub_stability.svg")

# ==============================================================
# 2. Mean Abs Z Across Layers SVG
# ==============================================================
def make_mean_z_svg():
    W, H = 900, 400
    pad_l, pad_r, pad_t, pad_b = 60, 30, 40, 50
    gw = W - pad_l - pad_r
    gh = H - pad_t - pad_b

    vals = [s["mean_abs_z"] for s in stats]
    max_v = max(vals) * 1.2
    n = len(vals)

    def px(i): return pad_l + (i / (n - 1)) * gw
    def py(v): return pad_t + gh - (v / max_v) * gh

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    lines.append(f'<rect width="{W}" height="{H}" fill="{C_BG}" rx="12"/>')
    lines.append(f'<text x="{W//2}" y="28" fill="{C_TEXT}" font-size="16" font-weight="700" text-anchor="middle">Cognitive Stress (Mean |Z|) Across Layers</text>')

    for p in phases:
        x1 = px(p["start"])
        x2 = px(p["end"])
        lines.append(f'<rect x="{x1}" y="{pad_t}" width="{x2-x1}" height="{gh}" fill="{PHASE_COLORS[p["phase"]-1]}" opacity="0.08"/>')

    pts = " ".join(f"{px(i)},{py(v)}" for i, v in enumerate(vals))
    lines.append(f'<polyline points="{pts}" fill="none" stroke="{C_CYAN}" stroke-width="2.5" stroke-linejoin="round"/>')
    for i, v in enumerate(vals):
        lines.append(f'<circle cx="{px(i)}" cy="{py(v)}" r="3" fill="{C_CYAN}"/>')

    for i in range(0, n, 5):
        lines.append(f'<text x="{px(i)}" y="{H-10}" fill="{C_TEXT}" font-size="10" text-anchor="middle" opacity="0.6">L{i}</text>')

    lines.append('</svg>')
    with open(os.path.join(OUTPUT, "cognitive_stress.svg"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  ✓ cognitive_stress.svg")

# ==============================================================
# 3. Six-Phase Lifecycle Diagram SVG
# ==============================================================
def make_lifecycle_svg():
    W, H = 900, 320
    phase_names = [
        "Sensory\nScanner", "Universal\nTranslator", "Logic\nEngine",
        "Deep\nReasoning", "Knowledge\nSynthesis", "Output\nPreparation"
    ]
    layer_ranges = ["L0-3", "L4-7", "L8-15", "L16-25", "L26-35", "L36-41"]
    phase_icons = ["P1","P2","P3","P4","P5","P6"]

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    lines.append(f'<rect width="{W}" height="{H}" fill="{C_BG}" rx="12"/>')
    lines.append(f'<text x="{W//2}" y="30" fill="{C_TEXT}" font-size="18" font-weight="700" text-anchor="middle">The Lifecycle of a Thought - Six Cognitive Phases</text>')

    bw = 120
    bh = 160
    gap = 12
    start_x = (W - (6 * bw + 5 * gap)) / 2
    start_y = 55

    for i in range(6):
        x = start_x + i * (bw + gap)
        y = start_y
        col = PHASE_COLORS[i]
        lines.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="10" fill="{col}" fill-opacity="0.15" stroke="{col}" stroke-width="2"/>')
        lines.append(f'<text x="{x+bw/2}" y="{y+30}" fill="{col}" font-size="18" font-weight="900" text-anchor="middle">{phase_icons[i]}</text>')
        for j, part in enumerate(phase_names[i].split("\n")):
            lines.append(f'<text x="{x+bw/2}" y="{y+55+j*18}" fill="{C_TEXT}" font-size="13" font-weight="600" text-anchor="middle">{part}</text>')
        lines.append(f'<text x="{x+bw/2}" y="{y+100}" fill="{col}" font-size="12" text-anchor="middle" font-weight="700">{layer_ranges[i]}</text>')
        lines.append(f'<text x="{x+bw/2}" y="{y+120}" fill="{C_TEXT}" font-size="10" text-anchor="middle" opacity="0.6">Phase {i+1}</text>')

        if i < 5:
            ax = x + bw + 2
            ay = y + bh / 2
            lines.append(f'<line x1="{ax}" y1="{ay}" x2="{ax+gap-4}" y2="{ay}" stroke="{C_TEXT}" stroke-width="2" stroke-opacity="0.5" marker-end="url(#arrowhead)"/>')

    lines.append('<defs><marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">')
    lines.append(f'<polygon points="0 0, 8 3, 0 6" fill="{C_TEXT}" fill-opacity="0.5"/></marker></defs>')

    # Bottom stats
    cross_lingual = [1018, 1150, 2322, 2865, 2692, 1618]
    for i in range(6):
        x = start_x + i * (bw + gap)
        y = start_y + bh + 15
        lines.append(f'<text x="{x+bw/2}" y="{y+10}" fill="{C_TEXT}" font-size="9" text-anchor="middle" opacity="0.5">Cross-Lingual</text>')
        lines.append(f'<text x="{x+bw/2}" y="{y+24}" fill="{PHASE_COLORS[i]}" font-size="14" font-weight="700" text-anchor="middle">{cross_lingual[i]:,}</text>')

    lines.append('</svg>')
    with open(os.path.join(OUTPUT, "lifecycle_phases.svg"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  ✓ lifecycle_phases.svg")

# ==============================================================
# 4. Neuron Hierarchy Pyramid SVG
# ==============================================================
def make_pyramid_svg():
    W, H = 600, 380
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    lines.append(f'<rect width="{W}" height="{H}" fill="{C_BG}" rx="12"/>')
    lines.append(f'<text x="{W//2}" y="30" fill="{C_TEXT}" font-size="16" font-weight="700" text-anchor="middle">The Sparse Elite Pyramid — 430,080 Neurons</text>')

    cx = W / 2
    tiers = [
        {"label": "L3 Hub/Goal", "pct": "~3%", "count": "~12,900", "color": C_RED, "w": 160, "y": 60, "h": 70},
        {"label": "L2 Specialist", "pct": "~12%", "count": "~51,600", "color": C_ORANGE, "w": 320, "y": 145, "h": 70},
        {"label": "L1 Foundation", "pct": "~85%", "count": "~365,568", "color": C_ACCENT, "w": 500, "y": 230, "h": 70},
    ]
    for t in tiers:
        x = cx - t["w"] / 2
        lines.append(f'<rect x="{x}" y="{t["y"]}" width="{t["w"]}" height="{t["h"]}" rx="8" fill="{t["color"]}" fill-opacity="0.2" stroke="{t["color"]}" stroke-width="2"/>')
        lines.append(f'<text x="{cx}" y="{t["y"]+25}" fill="{t["color"]}" font-size="14" font-weight="700" text-anchor="middle">{t["label"]}</text>')
        lines.append(f'<text x="{cx}" y="{t["y"]+45}" fill="{C_TEXT}" font-size="12" text-anchor="middle">{t["count"]} neurons ({t["pct"]})</text>')
        lines.append(f'<text x="{cx}" y="{t["y"]+60}" fill="{C_TEXT}" font-size="10" text-anchor="middle" opacity="0.5">{"Cross-domain integrators" if "Hub" in t["label"] else "Domain experts" if "Spec" in t["label"] else "General-purpose backbone"}</text>')

    lines.append(f'<text x="{cx}" y="{H-20}" fill="{C_TEXT}" font-size="11" text-anchor="middle" opacity="0.4">Top 100 neurons (0.02%) have z-scores up to 86 sigma - extreme power-law</text>')
    lines.append('</svg>')
    with open(os.path.join(OUTPUT, "neuron_pyramid.svg"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  ✓ neuron_pyramid.svg")

# ==============================================================
# 5. Benchmark Fingerprint Heatmap SVG
# ==============================================================
def make_benchmark_heatmap_svg():
    benchmarks = ["GSM8K", "HellaSwag", "HumanEval", "MMLU_Pro", "MMMLU", "RedTeaming", "TruthfulQA"]
    W, H = 900, 350
    pad_l, pad_t = 100, 50
    cell_w = 18
    cell_h = 32
    gw = cell_w * 42
    gh = cell_h * 7

    # Load benchmark stress from the raw CSVs
    import pandas as pd
    bench_dir = r"D:\neurocartography_data\benchmarks"
    stress = {}
    for b in benchmarks:
        s = []
        for l in range(42):
            path = os.path.join(bench_dir, b, f"layer_{l:02d}_neuron_zscores.csv")
            df = pd.read_csv(path)
            s.append(float(df['z_score'].abs().mean()))
        stress[b] = s

    # Normalize each benchmark by its own max
    norm = {}
    for b in benchmarks:
        mx = max(stress[b])
        norm[b] = [v / mx for v in stress[b]]

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    lines.append(f'<rect width="{W}" height="{H}" fill="{C_BG}" rx="12"/>')
    lines.append(f'<text x="{W//2}" y="30" fill="{C_TEXT}" font-size="16" font-weight="700" text-anchor="middle">Benchmark Stress Fingerprints (7 Tasks × 42 Layers)</text>')

    for bi, b in enumerate(benchmarks):
        y = pad_t + bi * cell_h
        lines.append(f'<text x="{pad_l-8}" y="{y+cell_h/2+4}" fill="{BENCH_COLORS[bi]}" font-size="10" text-anchor="end" font-weight="600">{b}</text>')
        for li in range(42):
            x = pad_l + li * cell_w
            v = norm[b][li]
            # Map to color intensity
            r = int(13 + v * 230)
            g = int(17 + (1-v) * 20)
            b_c = int(34 + (1-v) * 20)
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w-1}" height="{cell_h-1}" rx="2" fill="rgb({r},{g},{b_c})" opacity="{0.3+v*0.7}"/>')

    for i in range(0, 42, 5):
        x = pad_l + i * cell_w
        lines.append(f'<text x="{x+cell_w/2}" y="{pad_t+gh+15}" fill="{C_TEXT}" font-size="9" text-anchor="middle" opacity="0.5">L{i}</text>')

    lines.append('</svg>')
    with open(os.path.join(OUTPUT, "benchmark_heatmap.svg"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  ✓ benchmark_heatmap.svg")

# ==============================================================
# 6. Polysemantic Collision Count SVG
# ==============================================================
def make_collision_svg():
    W, H = 900, 400
    pad_l, pad_r, pad_t, pad_b = 60, 30, 40, 50
    gw = W - pad_l - pad_r
    gh = H - pad_t - pad_b

    # Load collision data
    collisions = {}
    with open(os.path.join(RESULTS, "polysemantic_sites.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            collisions[int(row["layer"])] = int(row["collision_count"])

    vals = [collisions.get(i, 0) for i in range(42)]
    max_v = max(vals) * 1.15
    n = 42

    def px(i): return pad_l + (i / (n - 1)) * gw
    def py(v): return pad_t + gh - (v / max_v) * gh

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    lines.append(f'<rect width="{W}" height="{H}" fill="{C_BG}" rx="12"/>')
    lines.append(f'<text x="{W//2}" y="28" fill="{C_TEXT}" font-size="16" font-weight="700" text-anchor="middle">Safety vs Reasoning Polysemantic Collisions per Layer</text>')

    for p in phases:
        x1 = px(p["start"])
        x2 = px(p["end"])
        lines.append(f'<rect x="{x1}" y="{pad_t}" width="{x2-x1}" height="{gh}" fill="{PHASE_COLORS[p["phase"]-1]}" opacity="0.08"/>')

    # Threshold line at 20
    y20 = py(20)
    lines.append(f'<line x1="{pad_l}" y1="{y20}" x2="{W-pad_r}" y2="{y20}" stroke="{C_RED}" stroke-width="1" stroke-dasharray="6,4" stroke-opacity="0.6"/>')
    lines.append(f'<text x="{W-pad_r+5}" y="{y20+4}" fill="{C_RED}" font-size="9" opacity="0.8">threshold=20</text>')

    # Bars
    bar_w = gw / n * 0.7
    for i, v in enumerate(vals):
        x = px(i) - bar_w / 2
        y = py(v)
        h = py(0) - y
        col = C_RED if v > 80 else C_ORANGE if v > 50 else C_ACCENT
        lines.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" rx="2" fill="{col}" opacity="0.8"/>')

    for i in range(0, n, 5):
        lines.append(f'<text x="{px(i)}" y="{H-10}" fill="{C_TEXT}" font-size="10" text-anchor="middle" opacity="0.6">L{i}</text>')

    lines.append('</svg>')
    with open(os.path.join(OUTPUT, "polysemantic_collisions.svg"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  ✓ polysemantic_collisions.svg")


# ==============================================================
# 7. Singular Value Heartbeat SVG
# ==============================================================
def make_sv_heartbeat_svg():
    W, H = 900, 350
    pad_l, pad_r, pad_t, pad_b = 60, 30, 40, 50
    gw = W - pad_l - pad_r
    gh = H - pad_t - pad_b

    sv_data = [8.387,9.386,11.586,8.647,9.240,10.056,10.189,9.538,11.721,10.329,7.611,8.720,9.968,9.640,7.551,9.254,7.202,8.519,8.082,7.820,6.977,6.665,8.664,9.896,9.575,7.599,7.114,8.669,9.099,8.548,8.794,7.762,8.673,8.009,8.212,8.460,6.198,5.292,5.065,5.749,6.084,8.530]
    max_v = max(sv_data) * 1.1
    min_v = min(sv_data) * 0.9
    n = len(sv_data)

    def px(i): return pad_l + (i / (n - 1)) * gw
    def py(v): return pad_t + gh - ((v - min_v) / (max_v - min_v)) * gh

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    lines.append(f'<rect width="{W}" height="{H}" fill="{C_BG}" rx="12"/>')
    lines.append(f'<text x="{W//2}" y="28" fill="{C_TEXT}" font-size="16" font-weight="700" text-anchor="middle">Singular Value Heartbeat - Knowledge Concentration per Layer</text>')

    for p in phases:
        x1 = px(p["start"])
        x2 = px(p["end"])
        lines.append(f'<rect x="{x1}" y="{pad_t}" width="{x2-x1}" height="{gh}" fill="{PHASE_COLORS[p["phase"]-1]}" opacity="0.08"/>')

    # Area fill
    area_pts = f"{px(0)},{pad_t+gh} "
    area_pts += " ".join(f"{px(i)},{py(v)}" for i, v in enumerate(sv_data))
    area_pts += f" {px(n-1)},{pad_t+gh}"
    lines.append(f'<polygon points="{area_pts}" fill="{C_GREEN}" fill-opacity="0.1"/>')

    pts = " ".join(f"{px(i)},{py(v)}" for i, v in enumerate(sv_data))
    lines.append(f'<polyline points="{pts}" fill="none" stroke="{C_GREEN}" stroke-width="2.5" stroke-linejoin="round"/>')

    # Mark peaks
    for i, v in enumerate(sv_data):
        if v > 10.5:
            lines.append(f'<circle cx="{px(i)}" cy="{py(v)}" r="5" fill="{C_RED}" stroke="{C_BG}" stroke-width="2"/>')
            lines.append(f'<text x="{px(i)}" y="{py(v)-10}" fill="{C_RED}" font-size="10" text-anchor="middle" font-weight="700">{v:.1f}</text>')

    for i in range(0, n, 5):
        lines.append(f'<text x="{px(i)}" y="{H-10}" fill="{C_TEXT}" font-size="10" text-anchor="middle" opacity="0.6">L{i}</text>')

    lines.append('</svg>')
    with open(os.path.join(OUTPUT, "sv_heartbeat.svg"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  ✓ sv_heartbeat.svg")


print("Generating SVG visualizations...")
make_hub_stability_svg()
make_mean_z_svg()
make_lifecycle_svg()
make_pyramid_svg()
make_collision_svg()
make_sv_heartbeat_svg()
make_benchmark_heatmap_svg()
print("All SVGs generated!")
