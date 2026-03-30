"""
TopoFeatures Explainer — 3:00 Manim animation
Render preview:  manim render -pql video/explainer.py TopoFeaturesExplainer
Render final:    manim render -qh video/explainer.py TopoFeaturesExplainer
"""

from manim import *
import numpy as np
from scipy.spatial.distance import pdist, squareform

# ── Global config ──────────────────────────────────────────────────────────
config.background_color = "#1a1a2e"

# ── Color constants ────────────────────────────────────────────────────────
TEAL = "#58c4dd"
YELLOW_C = "#ffff00"
CORAL = "#ff6b6b"
SOFT_WHITE = "#e0e0e0"
DIM_GRAY = "#666666"
CODE_BG = "#0d1117"


class TopoFeaturesExplainer(Scene):
    def construct(self):
        self.act1_problem()       # 00:00 - 00:35
        self._log_time("Act 1", 35)
        self.act2_point_cloud()   # 00:35 - 01:05
        self._log_time("Act 2", 65)
        self.act3_filtration()    # 01:05 - 01:55
        self._log_time("Act 3", 115)
        self.act4_diagram()       # 01:55 - 02:25
        self._log_time("Act 4", 145)
        self.act5_results()       # 02:25 - 02:48
        self._log_time("Act 5", 168)
        self.act6_usage()         # 02:48 - 03:00
        self._log_time("Act 6", 180)

    def _log_time(self, act_name, target_s):
        elapsed = self.renderer.time
        status = "OK" if abs(elapsed - target_s) <= 2 else "DRIFT"
        print(f"{act_name} ends at: {elapsed:.1f}s (target: {target_s}s) {status}")

    # ══════════════════════════════════════════════════════════════════════
    # ACT 1: The Problem  (00:00 – 00:35, 35s)
    # ══════════════════════════════════════════════════════════════════════
    def act1_problem(self):
        # 00:00 – Title card
        title = Text("What shape is your time series?", font_size=48, color=SOFT_WHITE)
        self.play(Write(title, run_time=2))  # 00:00-00:02
        self.wait(2)                          # 00:02-00:04
        self.play(FadeOut(title, run_time=0.5))  # 00:04-00:04.5

        # 00:04 – Time series + feature names
        axes = Axes(
            x_range=[0, 10, 1], y_range=[-2, 2, 1],
            x_length=7, y_length=3,
            axis_config={"color": DIM_GRAY, "include_ticks": True},
        ).shift(LEFT * 1.5)

        t = np.linspace(0, 10, 300)
        line1 = axes.plot(lambda x: np.sin(x), color=TEAL)
        line2 = axes.plot(lambda x: np.sin(1.6 * x + 0.5), color=YELLOW_C)
        line3 = axes.plot(lambda x: np.sin(2.3 * x + 1.0), color=CORAL)

        self.play(FadeIn(axes, run_time=0.5))  # 00:04.5-00:05
        self.play(
            Create(line1, run_time=1.5),
            Create(line2, run_time=1.5),
            Create(line3, run_time=1.5),
        )  # 00:05-00:06.5

        feat_names = VGroup(
            Text("mean", font_size=22, color=SOFT_WHITE),
            Text("variance", font_size=22, color=SOFT_WHITE),
            Text("spectral power", font_size=22, color=SOFT_WHITE),
            Text("entropy", font_size=22, color=SOFT_WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(RIGHT, buff=1.0)

        self.play(
            LaggedStart(
                *[FadeIn(f, run_time=0.3) for f in feat_names],
                lag_ratio=0.4,
            )
        )  # ~00:06.5-00:08.5
        self.wait(4)  # 00:08.5-00:12.5  narration: "statistics… miss something"

        # 00:15 – Morph to attractor
        self.play(
            FadeOut(axes, run_time=1),
            FadeOut(feat_names, run_time=1),
        )  # 00:12.5-00:13.5
        self.play(
            FadeOut(line1, run_time=0.5),
            FadeOut(line2, run_time=0.5),
            FadeOut(line3, run_time=0.5),
        )  # 00:13.5-00:14

        # Lissajous attractor
        lissajous = ParametricFunction(
            lambda t: np.array([
                2 * np.sin(t),
                2 * np.sin(1.6 * t + 0.5),
                0,
            ]),
            t_range=[0, 4 * np.pi, 0.02],
            color=TEAL,
            stroke_width=2,
        )
        self.play(FadeIn(lissajous, run_time=1))  # 00:14-00:15

        # Dots along curve
        dot_ts = np.linspace(0, 4 * np.pi, 40, endpoint=False)
        dots_on_curve = VGroup(*[
            Dot(
                point=[2 * np.sin(t_val), 2 * np.sin(1.6 * t_val + 0.5), 0],
                radius=0.04, color=TEAL,
            )
            for t_val in dot_ts
        ])
        self.play(FadeIn(dots_on_curve, run_time=0.5))  # 00:15-00:15.5

        attractor_label = Text("attractor", font_size=24, color=DIM_GRAY).next_to(
            lissajous, DOWN, buff=0.3
        )
        self.play(FadeIn(attractor_label, run_time=0.5))  # 00:15.5-00:16

        # 00:25-00:35 – Hold with attractor text
        self.wait(8)  # 00:16-00:24  narration catch-up
        attractor_text = Text(
            "Its structure tells you things\nno statistic will reveal.",
            font_size=28, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(attractor_text, run_time=1))  # 00:24-00:25
        self.wait(8)  # 00:25-00:33

        # Transition out
        self.play(
            FadeOut(lissajous), FadeOut(dots_on_curve),
            FadeOut(attractor_label), FadeOut(attractor_text),
            run_time=1,
        )  # 00:33-00:34
        self.wait(1)  # 00:34-00:35

    # ══════════════════════════════════════════════════════════════════════
    # ACT 2: From Time Series to Point Cloud  (00:35 – 01:05, 30s)
    # ══════════════════════════════════════════════════════════════════════
    def act2_point_cloud(self):
        # ── PCA path (00:35-00:46) ──
        matrix_rect = Rectangle(
            width=3.5, height=1.8, fill_color=CODE_BG, fill_opacity=0.9,
            stroke_color=DIM_GRAY,
        ).shift(LEFT * 3.5)
        matrix_label = Text(
            "50 sensors × 5000 steps", font_size=16, color=SOFT_WHITE,
        ).move_to(matrix_rect)

        arrow = Arrow(
            matrix_rect.get_right(), matrix_rect.get_right() + RIGHT * 2,
            color=SOFT_WHITE, buff=0.2,
        )
        pca_label = Text("PCA(3)", font_size=18, color=YELLOW_C).next_to(arrow, UP, buff=0.1)

        cloud_rect = Rectangle(
            width=2, height=1.4, fill_color=CODE_BG, fill_opacity=0.9,
            stroke_color=DIM_GRAY,
        ).next_to(arrow, RIGHT, buff=0.2)
        cloud_label = Text("3D cloud", font_size=16, color=SOFT_WHITE).move_to(cloud_rect)

        self.play(FadeIn(matrix_rect), FadeIn(matrix_label), run_time=0.5)  # 00:35-00:35.5
        self.play(GrowArrow(arrow, run_time=0.5), FadeIn(pca_label, run_time=0.5))  # 00:35.5-00:36
        self.play(FadeIn(cloud_rect), FadeIn(cloud_label), run_time=0.5)  # 00:36-00:36.5
        self.wait(3)  # 00:36.5-00:39.5  narration

        # Transform cloud label into actual dots (noisy ring)
        rng = np.random.default_rng(99)
        theta_pca = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        pca_dots = VGroup(*[
            Dot(
                point=[
                    1.5 * np.cos(th) + 0.15 * rng.standard_normal() + 2.5,
                    1.5 * np.sin(th) + 0.15 * rng.standard_normal(),
                    0,
                ],
                radius=0.04, color=TEAL,
            )
            for th in theta_pca
        ])
        self.play(
            FadeOut(cloud_rect), FadeOut(cloud_label),
            FadeIn(pca_dots, run_time=1.5),
        )  # 00:39.5-00:41
        self.wait(3)  # 00:41-00:44

        # Clear PCA visual
        self.play(
            FadeOut(matrix_rect), FadeOut(matrix_label),
            FadeOut(arrow), FadeOut(pca_label), FadeOut(pca_dots),
            run_time=0.5,
        )  # 00:44-00:44.5

        # ── Delay embedding path (00:46-00:58) ──
        embed_axes = Axes(
            x_range=[0, 8, 1], y_range=[-1.5, 1.5, 1],
            x_length=8, y_length=2.5,
            axis_config={"color": DIM_GRAY},
        ).shift(UP * 1)

        signal = embed_axes.plot(
            lambda x: np.sin(x) + 0.3 * np.sin(3 * x),
            color=TEAL, x_range=[0, 8],
        )
        self.play(FadeIn(embed_axes, run_time=0.3), Create(signal, run_time=1))  # 00:44.5-00:45.8

        # Highlight 3 points at t, t-tau, t-2tau
        tau = 1.5
        t_val = 5.0
        sample_xs = [t_val, t_val - tau, t_val - 2 * tau]
        sample_labels = ["x(t)", "x(t−τ)", "x(t−2τ)"]
        sample_colors = [CORAL, YELLOW_C, TEAL]

        marks = VGroup()
        labels = VGroup()
        for x_s, lab, col in zip(sample_xs, sample_labels, sample_colors):
            y_s = np.sin(x_s) + 0.3 * np.sin(3 * x_s)
            pt = embed_axes.c2p(x_s, y_s)
            dot = Dot(pt, radius=0.06, color=col)
            vline = DashedLine(
                pt, embed_axes.c2p(x_s, 0),
                color=col, stroke_width=1.5,
            )
            txt = Text(lab, font_size=16, color=col).next_to(dot, UP, buff=0.15)
            marks.add(dot, vline)
            labels.add(txt)

        self.play(FadeIn(marks, run_time=1), FadeIn(labels, run_time=1))  # 00:45.8-00:46.8

        # Arrow to a point in 2D space below
        embed_dot_target = Dot([2, -2, 0], radius=0.06, color=SOFT_WHITE)
        arrow_embed = Arrow(
            embed_axes.c2p(t_val, 0) + DOWN * 0.3,
            embed_dot_target.get_center(),
            color=SOFT_WHITE, stroke_width=2, buff=0.15,
        )
        one_pt_label = Text("→ one point", font_size=18, color=SOFT_WHITE).next_to(
            embed_dot_target, RIGHT, buff=0.2,
        )
        self.play(
            GrowArrow(arrow_embed, run_time=0.5),
            FadeIn(embed_dot_target, run_time=0.5),
            FadeIn(one_pt_label, run_time=0.5),
        )  # 00:46.8-00:47.3
        self.wait(2)  # 00:47.3-00:49.3

        # Flash many dots to fill cloud
        rng2 = np.random.default_rng(42)
        theta_embed = np.linspace(0, 2 * np.pi, 35, endpoint=False)
        embed_cloud = VGroup(*[
            Dot(
                point=[
                    1.2 * np.cos(th) + 0.12 * rng2.standard_normal(),
                    1.2 * np.sin(th) + 0.12 * rng2.standard_normal() - 2,
                    0,
                ],
                radius=0.04, color=TEAL,
            )
            for th in theta_embed
        ])
        self.play(FadeIn(embed_cloud, run_time=1))  # 00:49.3-00:50.3
        self.wait(3)  # 00:50.3-00:53.3

        # ── Converge (00:58-01:05) ──
        self.play(
            FadeOut(embed_axes), FadeOut(signal), FadeOut(marks), FadeOut(labels),
            FadeOut(arrow_embed), FadeOut(embed_dot_target), FadeOut(one_pt_label),
            run_time=0.5,
        )  # 00:53.3-00:53.8

        # Center the cloud
        self.play(embed_cloud.animate.move_to(ORIGIN), run_time=0.5)  # 00:53.8-00:54.3

        cloud_text = Text(
            "A point cloud with a shape", font_size=28, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(cloud_text, run_time=0.5))  # 00:54.3-00:54.8
        self.wait(5)  # 00:54.8-00:59.8

        # Keep dots reference for Act 3 by storing on self
        self._cloud_dots = embed_cloud

        self.play(FadeOut(cloud_text, run_time=0.5))  # 00:59.8-01:00.3
        self.play(FadeOut(embed_cloud, run_time=0.5))  # 01:00.3-01:00.8
        self.wait(4.2)  # pad to 01:05

    # ══════════════════════════════════════════════════════════════════════
    # ACT 3: Growing Balls / Filtration  (01:05 – 01:55, 50s)
    # ══════════════════════════════════════════════════════════════════════
    def act3_filtration(self):
        # ── Pre-compute point cloud & edges ──
        rng = np.random.default_rng(42)
        n_pts = 60
        theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        pts = np.column_stack([
            2 * np.cos(theta) + 0.25 * rng.standard_normal(n_pts),
            2 * np.sin(theta) + 0.25 * rng.standard_normal(n_pts),
        ])

        dist_matrix = squareform(pdist(pts))

        # Filtration radii
        r0 = 0.05
        r1 = 0.30
        r2 = 0.55
        r3 = 0.85
        r4 = 1.50

        def edges_at_radius(r):
            """Return list of (i,j) where dist < 2*r."""
            result = []
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    if dist_matrix[i, j] < 2 * r:
                        result.append((i, j))
            return result

        edges_r1 = edges_at_radius(r1)
        edges_r2 = edges_at_radius(r2)
        edges_r3 = edges_at_radius(r3)

        # New edges at each step (for incremental reveal)
        edges_r1_set = set(edges_r1)
        edges_r2_new = [e for e in edges_r2 if e not in edges_r1_set]
        edges_r2_set = set(edges_r2)
        edges_r3_new = [e for e in edges_r3 if e not in edges_r2_set]

        # ── Count connected components at each radius ──
        def count_components(n, edge_list):
            parent = list(range(n))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
            for i, j in edge_list:
                union(i, j)
            return len(set(find(x) for x in range(n)))

        cc_r1 = count_components(n_pts, edges_r1)
        cc_r2 = count_components(n_pts, edges_r2)

        # ── Create visual objects ──
        point_dots = VGroup(*[
            Dot(point=[pts[i, 0], pts[i, 1], 0], radius=0.04, color=TEAL)
            for i in range(n_pts)
        ])

        # Growing circles around each point
        circles = VGroup(*[
            Circle(
                radius=r0, stroke_color=TEAL, stroke_opacity=0.3,
                fill_color=TEAL, fill_opacity=0.08,
            ).move_to([pts[i, 0], pts[i, 1], 0])
            for i in range(n_pts)
        ])

        # Pre-create ALL edge lines (invisible initially)
        all_edges_r3 = edges_at_radius(r3)
        edge_lines = {}
        for i, j in all_edges_r3:
            line = Line(
                [pts[i, 0], pts[i, 1], 0],
                [pts[j, 0], pts[j, 1], 0],
                color=TEAL, stroke_width=1.5, stroke_opacity=0,
            )
            edge_lines[(i, j)] = line

        edge_group = VGroup(*edge_lines.values())

        # β counters
        beta0_text = Text(f"β₀ = {n_pts}", font_size=24, color=SOFT_WHITE)
        beta1_text = Text("β₁ = 0", font_size=24, color=SOFT_WHITE)
        counter = VGroup(beta0_text, beta1_text).arrange(
            DOWN, aligned_edge=LEFT, buff=0.15,
        ).to_corner(UR, buff=0.5)

        # Show initial state
        self.play(FadeIn(point_dots, run_time=0.5))  # 01:05-01:05.5
        self.play(FadeIn(circles, run_time=0.5))       # 01:05.5-01:06
        self.play(FadeIn(counter, run_time=0.5))        # 01:06-01:06.5
        self.add(edge_group)  # add all edges (invisible)

        # ── Step 1 (01:05-01:12): r0 → r1, small clusters ──
        scale_factor_1 = r1 / r0
        self.play(
            *[c.animate.scale(scale_factor_1) for c in circles],
            run_time=3,
        )  # 01:06.5-01:09.5

        # Reveal r1 edges
        r1_anims = [edge_lines[e].animate.set_stroke(opacity=1) for e in edges_r1 if e in edge_lines]
        if r1_anims:
            self.play(*r1_anims, run_time=0.5)  # 01:09.5-01:10

        # Update β₀
        new_b0 = Text(f"β₀ = {cc_r1}", font_size=24, color=SOFT_WHITE).move_to(beta0_text)
        self.play(Transform(beta0_text, new_b0, run_time=0.3))  # 01:10-01:10.3
        self.wait(1.7)  # 01:10.3-01:12  narration: "many isolated clusters"

        # ── Step 2 (01:12-01:20): r1 → r2, clusters merge, loop forms ──
        scale_factor_2 = r2 / r1
        self.play(
            *[c.animate.scale(scale_factor_2) for c in circles],
            run_time=3,
        )  # 01:12-01:15

        # Reveal r2 new edges
        r2_anims = [edge_lines[e].animate.set_stroke(opacity=1) for e in edges_r2_new if e in edge_lines]
        if r2_anims:
            self.play(*r2_anims, run_time=0.5)  # 01:15-01:15.5

        new_b0_2 = Text(f"β₀ = {cc_r2}", font_size=24, color=SOFT_WHITE).move_to(beta0_text)
        self.play(Transform(beta0_text, new_b0_2, run_time=0.3))  # 01:15.5-01:15.8
        self.wait(4.2)  # 01:15.8-01:20  narration catch-up

        # ── Step 3 (01:20-01:33): Loop highlight, β₁ = 1 ──
        # Identify ring edges: consecutive points in theta ordering form the outer ring
        ring_edges = []
        for k in range(n_pts):
            i, j = k, (k + 1) % n_pts
            pair = (min(i, j), max(i, j))
            if pair in edge_lines:
                ring_edges.append(pair)

        # Highlight ring edges in CORAL
        ring_highlight_anims = []
        for pair in ring_edges:
            line = edge_lines[pair]
            ring_highlight_anims.append(
                line.animate.set_color(CORAL).set_stroke(width=3, opacity=1)
            )
        if ring_highlight_anims:
            self.play(
                LaggedStart(*ring_highlight_anims, lag_ratio=0.02),
                run_time=1.5,
            )  # 01:20-01:21.5

        # β₁ → 1
        new_b1 = Text("β₁ = 1", font_size=24, color=CORAL).move_to(beta1_text)
        self.play(Transform(beta1_text, new_b1, run_time=0.3))  # 01:21.5-01:21.8
        self.play(Indicate(beta1_text, color=CORAL, run_time=0.5))  # 01:21.8-01:22.3
        self.wait(10.7)  # 01:22.3-01:33  narration: "topological feature is born"

        # ── Step 4 (01:33-01:43): Loop fills & dies ──
        scale_factor_3 = r3 / r2
        self.play(
            *[c.animate.scale(scale_factor_3) for c in circles],
            run_time=3,
        )  # 01:33-01:36

        # Reveal r3 new edges (interior cross-connections)
        r3_anims = [edge_lines[e].animate.set_stroke(opacity=1) for e in edges_r3_new if e in edge_lines]
        if r3_anims:
            self.play(*r3_anims, run_time=0.5)  # 01:36-01:36.5

        # Ring edges revert to normal
        revert_anims = []
        for pair in ring_edges:
            line = edge_lines[pair]
            revert_anims.append(
                line.animate.set_color(TEAL).set_stroke(width=1.5)
            )
        if revert_anims:
            self.play(*revert_anims, run_time=0.5)  # 01:36.5-01:37

        # β₁ → 0
        new_b1_0 = Text("β₁ = 0", font_size=24, color=SOFT_WHITE).move_to(beta1_text)
        self.play(Transform(beta1_text, new_b1_0, run_time=0.3))  # 01:37-01:37.3

        new_b0_1 = Text("β₀ = 1", font_size=24, color=SOFT_WHITE).move_to(beta0_text)
        self.play(Transform(beta0_text, new_b0_1, run_time=0.3))  # 01:37.3-01:37.6
        self.wait(5.4)  # 01:37.6-01:43  narration: "born at one scale, died at another"

        # ── Step 5 (01:43-01:55): Persistence = signal ──
        all_act3 = VGroup(point_dots, circles, edge_group, counter)
        self.play(all_act3.animate.set_opacity(0.3), run_time=1)  # 01:43-01:44

        persist_text = VGroup(
            Text("Long-lived features → real structure", font_size=28, color=CORAL),
            Text("Short-lived features → noise", font_size=28, color=DIM_GRAY),
        ).arrange(DOWN, buff=0.4).move_to(ORIGIN)

        self.play(Write(persist_text[0], run_time=1.5))  # 01:44-01:45.5
        self.play(Write(persist_text[1], run_time=1.5))  # 01:45.5-01:47
        self.wait(6)  # 01:47-01:53  narration finish

        self.play(
            FadeOut(all_act3), FadeOut(persist_text),
            run_time=1,
        )  # 01:53-01:54
        self.wait(1)  # 01:54-01:55

    # ══════════════════════════════════════════════════════════════════════
    # ACT 4: The Persistence Diagram  (01:55 – 02:25, 30s)
    # ══════════════════════════════════════════════════════════════════════
    def act4_diagram(self):
        # Axes
        pd_axes = Axes(
            x_range=[0, 1.5, 0.5], y_range=[0, 1.5, 0.5],
            x_length=5, y_length=5,
            axis_config={"color": SOFT_WHITE, "include_ticks": True},
        )
        x_label = Text("birth", font_size=20, color=SOFT_WHITE).next_to(pd_axes.x_axis, DOWN, buff=0.2)
        y_label = Text("death", font_size=20, color=SOFT_WHITE).next_to(pd_axes.y_axis, LEFT, buff=0.2).rotate(PI / 2)
        diag_line = DashedLine(
            pd_axes.c2p(0, 0), pd_axes.c2p(1.5, 1.5),
            color=DIM_GRAY, stroke_width=1.5,
        )

        self.play(
            FadeIn(pd_axes, run_time=0.5),
            FadeIn(x_label, run_time=0.5),
            FadeIn(y_label, run_time=0.5),
            Create(diag_line, run_time=0.5),
        )  # 01:55-01:55.5

        # Persistence diagram points
        # H0 features (near diagonal, short-lived)
        h0_pts = [(0.10, 0.18), (0.15, 0.25), (0.22, 0.30), (0.08, 0.14), (0.30, 0.38)]
        # H1 features (one short, one long)
        h1_pts_short = [(0.50, 0.58)]
        h1_pts_long = [(0.50, 1.15)]  # THE loop

        pd_dots = VGroup()
        for bx, dy in h0_pts:
            d = Dot(pd_axes.c2p(bx, dy), radius=0.05, color=TEAL)
            pd_dots.add(d)
        for bx, dy in h1_pts_short:
            d = Dot(pd_axes.c2p(bx, dy), radius=0.05, color=CORAL)
            pd_dots.add(d)
        for bx, dy in h1_pts_long:
            d = Dot(pd_axes.c2p(bx, dy), radius=0.08, color=CORAL)
            pd_dots.add(d)

        self.play(
            LaggedStart(*[FadeIn(d, run_time=0.3) for d in pd_dots], lag_ratio=0.3),
        )  # 01:55.5-01:58
        self.wait(6.5)  # 01:58-02:04.5  narration: "near diagonal → noise"

        # 02:08 – Five feature annotations (cycle through)
        the_loop_dot = pd_dots[-1]  # the long-lived H1 dot

        def make_annotation(text_str, target_mob, direction=RIGHT, color=SOFT_WHITE):
            txt = Text(text_str, font_size=18, color=color)
            arr = Arrow(
                txt.get_edge_center(-direction),
                target_mob.get_center(),
                color=color, stroke_width=2, buff=0.1, max_tip_length_to_length_ratio=0.15,
            )
            txt.next_to(arr, direction, buff=0.1)
            return VGroup(txt, arr)

        # β₀ → cluster of H0 dots
        ann_b0 = make_annotation("β₀", pd_dots[2], RIGHT, TEAL)
        self.play(FadeIn(ann_b0, run_time=0.8))  # 02:03-02:03.8
        self.wait(1.2)  # 02:03.8-02:05

        # β₁ → CORAL dot
        ann_b1 = make_annotation("β₁", the_loop_dot, LEFT, CORAL)
        self.play(FadeOut(ann_b0, run_time=0.3), FadeIn(ann_b1, run_time=0.8))  # 02:05-02:06.1
        self.wait(0.9)  # 02:06.1-02:07

        # Persistence entropy
        ann_pe = Text("persistence entropy", font_size=18, color=YELLOW_C).to_edge(
            DOWN, buff=1.5,
        )
        self.play(FadeOut(ann_b1, run_time=0.3), FadeIn(ann_pe, run_time=0.8))  # 02:07-02:08.1
        self.wait(1.4)  # 02:08.1-02:09.5

        # Max persistence — line from CORAL dot to diagonal
        loop_birth, loop_death = 0.50, 1.15
        max_pers_line = Line(
            pd_axes.c2p(loop_birth, loop_death),
            pd_axes.c2p(loop_birth, loop_birth),
            color=CORAL, stroke_width=2,
        )
        max_label = Text("max persistence", font_size=18, color=CORAL).next_to(
            max_pers_line, RIGHT, buff=0.15,
        )
        self.play(
            FadeOut(ann_pe, run_time=0.3),
            Create(max_pers_line, run_time=0.5),
            FadeIn(max_label, run_time=0.5),
        )  # 02:09.5-02:10.3
        self.wait(1.7)  # 02:10.3-02:12

        # Total persistence
        total_label = Text("total persistence", font_size=18, color=YELLOW_C).next_to(
            max_label, DOWN, buff=0.3,
        )
        self.play(
            FadeIn(total_label, run_time=0.5),
        )  # 02:12-02:12.5
        self.wait(5)  # 02:12.5-02:17.5

        # Clean annotations
        self.play(
            FadeOut(max_pers_line), FadeOut(max_label), FadeOut(total_label),
            run_time=0.5,
        )  # 02:16-02:16.5

        # 02:18 – Code snippet
        diagram_group = VGroup(pd_axes, x_label, y_label, diag_line, pd_dots)
        self.play(diagram_group.animate.set_opacity(0.4), run_time=0.5)  # 02:16.5-02:17

        code_bg = Rectangle(
            width=6, height=1.8, fill_color=CODE_BG, fill_opacity=0.95,
            stroke_color=DIM_GRAY, stroke_width=1,
        )
        code_line1 = Text(
            "from topofeatures import extract", font_size=20,
            color=TEAL, font="Monospace",
        )
        code_line2 = Text(
            "features = extract(X)", font_size=20,
            color=SOFT_WHITE, font="Monospace",
        )
        code_text = VGroup(code_line1, code_line2).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        code_block = VGroup(code_bg, code_text)

        self.play(FadeIn(code_block, run_time=1))  # 02:17-02:18
        self.wait(5)  # 02:18-02:23  narration: "five numbers, one function call"

        self.play(FadeOut(code_block), FadeOut(diagram_group), run_time=1)  # 02:23-02:24
        self.wait(1)  # 02:24-02:25

    # ══════════════════════════════════════════════════════════════════════
    # ACT 5: What It Predicts  (02:25 – 02:48, 23s)
    # ══════════════════════════════════════════════════════════════════════
    def act5_results(self):
        # 02:25 – Title flash
        title5 = Text("What it predicts", font_size=40, color=SOFT_WHITE)
        self.play(FadeIn(title5, run_time=0.5))  # 02:25-02:25.5
        self.wait(2)  # 02:25.5-02:27.5
        self.play(FadeOut(title5, run_time=0.5))  # 02:27-02:27.5

        # 02:31 – Results montage
        results_data = [
            ("Oscillator synchronization", "ρ = 0.88"),
            ("Neural network memory", "ρ = 0.83"),
            ("Power grid stability", "ρ = 0.69"),
            ("Real EEG sleep depth", "ρ = 0.83"),
        ]

        rows = VGroup()
        for domain, rho in results_data:
            left = Text(domain, font_size=24, color=SOFT_WHITE)
            right = Text(rho, font_size=24, color=YELLOW_C)
            row = VGroup(left, right).arrange(RIGHT, buff=1.5)
            rows.add(row)
        rows.arrange(DOWN, buff=0.4)

        self.play(FadeIn(rows, run_time=1))  # 02:27.5-02:28.5
        self.wait(7)  # 02:28.5-02:35.5  narration reads results

        # 02:40 – Summary
        summary = VGroup(
            Text("18/20 dynamical systems", font_size=30, color=CORAL),
            Text("From 6% of observed variables", font_size=26, color=SOFT_WHITE),
            Text("On real data", font_size=26, color=SOFT_WHITE),
        ).arrange(DOWN, buff=0.3)

        self.play(FadeOut(rows, run_time=0.5), FadeIn(summary, run_time=1))  # 02:35.5-02:37
        self.wait(8)  # 02:37-02:45  narration: "just the geometry"

        self.play(FadeOut(summary, run_time=1))  # 02:45-02:46
        self.wait(2)  # 02:46-02:48

    # ══════════════════════════════════════════════════════════════════════
    # ACT 6: Use It  (02:48 – 03:00, 12s)
    # ══════════════════════════════════════════════════════════════════════
    def act6_usage(self):
        # Two side-by-side code blocks
        left_bg = Rectangle(
            width=5, height=1.6, fill_color=CODE_BG, fill_opacity=0.95,
            stroke_color=DIM_GRAY, stroke_width=1,
        )
        left_code = VGroup(
            Text("from topofeatures import extract", font_size=18, color=TEAL, font="Monospace"),
            Text("features = extract(X)", font_size=18, color=SOFT_WHITE, font="Monospace"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        left_block = VGroup(left_bg, left_code).shift(LEFT * 3.2)

        right_bg = Rectangle(
            width=5, height=1.6, fill_color=CODE_BG, fill_opacity=0.95,
            stroke_color=DIM_GRAY, stroke_width=1,
        )
        right_code = VGroup(
            Text("# plug into tsfresh", font_size=18, color=DIM_GRAY, font="Monospace"),
            Text("settings.update(TOPO_SETTINGS)", font_size=18, color=SOFT_WHITE, font="Monospace"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        right_block = VGroup(right_bg, right_code).shift(RIGHT * 3.2)

        self.play(
            FadeIn(left_block, run_time=1),
            FadeIn(right_block, run_time=1),
        )  # 02:48-02:49
        self.wait(4)  # 02:49-02:53

        # 02:56 – pip install + final card
        self.play(FadeOut(left_block), FadeOut(right_block), run_time=0.5)  # 02:53-02:53.5

        pip_text = Text("pip install topo-features", font_size=40, color=TEAL)
        repo_text = Text(
            "github.com/musicofhel/topo-features", font_size=20, color=DIM_GRAY,
        ).next_to(pip_text, DOWN, buff=0.4)

        self.play(FadeIn(pip_text, run_time=0.5))  # 02:53.5-02:54
        self.play(FadeIn(repo_text, run_time=0.5))  # 02:54-02:54.5
        self.wait(5.5)  # 02:54.5-03:00  hold to end
