"""
Gorsellestirme:
Renkler: Siyah(engel) | Koyu gri(U-trap) | Kirmizi(open) | Mavi(closed) | 
Yesil(A* yolu) | Siyah cizgi(Robot yolu) | Siyah kesikli cizgi(A* gercek yolu)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Ortak alt-fonksiyonlar
def _build_display(grid_original, grid_used, open_log=None, closed_log=None):
    rows = len(grid_used)
    cols = len(grid_used[0])
    display = np.ones((rows, cols, 3))

    for y in range(rows):
        for x in range(cols):
            if grid_original[y][x] == 1:
                display[y, x] = [0.10, 0.10, 0.10]           # siyah engel

    for y in range(rows):
        for x in range(cols):
            if grid_original[y][x] == 0 and grid_used[y][x] == 1:
                display[y, x] = [0.48, 0.48, 0.48]           # koyu gri U-trap

    if open_log:
        for (x, y) in open_log:
            if 0 <= y < rows and 0 <= x < cols and grid_used[y][x] == 0:
                current = display[y, x]
                red_color = np.array([0.88, 0.18, 0.18])
                display[y, x] = red_color * 0.7 + current * 0.3

    if closed_log:
        for (x, y) in closed_log:
            if 0 <= y < rows and 0 <= x < cols and grid_used[y][x] == 0:
                current = display[y, x]
                blue_color = np.array([0.22, 0.46, 0.90])
                display[y, x] = blue_color * 0.7 + current * 0.3

    return display


def _draw_display(ax, display, rows, cols):
    ax.imshow(display, origin='upper', interpolation='nearest',
              extent=[-0.5, cols - 0.5, rows - 0.5, -0.5])
    for i in range(cols + 1):
        ax.axvline(i - 0.5, color='#aaaaaa', lw=0.5)
    for j in range(rows + 1):
        ax.axhline(j - 0.5, color='#aaaaaa', lw=0.5)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels(range(cols), fontsize=7)
    ax.set_yticklabels(range(rows), fontsize=7)
    ax.set_xlabel('x (col)', fontsize=9)
    ax.set_ylabel('y (row)', fontsize=9)


def _draw_astar_path(ax, raw_path, final_path, color_final='#111111',
                     color_raw='#999999', linestyle_final='-',
                     linestyle_raw='--'):
    if raw_path and len(raw_path) > 1:
        xs = [p[0] for p in raw_path]
        ys = [p[1] for p in raw_path]
        ax.plot(xs, ys, linestyle_raw, color=color_raw, lw=1.1, zorder=5)

    if final_path and len(final_path) > 1:
        xs = [p[0] for p in final_path]
        ys = [p[1] for p in final_path]
        ax.plot(xs, ys, linestyle_final, color=color_final, lw=2.4, zorder=6)

        for i in range(1, len(final_path) - 1):
            dx1 = final_path[i][0] - final_path[i - 1][0]
            dy1 = final_path[i][1] - final_path[i - 1][1]
            dx2 = final_path[i + 1][0] - final_path[i][0]
            dy2 = final_path[i + 1][1] - final_path[i][1]
            if (dx1, dy1) != (dx2, dy2):
                ax.plot(final_path[i][0], final_path[i][1], 's',
                        color='#22cc22', ms=10, zorder=8,
                        markeredgecolor='white', markeredgewidth=0.8)


def _draw_endpoints(ax, start, goal):
    ax.plot(start[0], start[1], 'o',
            color='#11cc11', ms=12, zorder=9,
            markeredgecolor='white', markeredgewidth=1.2)
    ax.plot(goal[0], goal[1], 's',
            color='#ffcc00', ms=12, zorder=9,
            markeredgecolor='#888800', markeredgewidth=1.2)


# Improved A* statik figuru
def draw_static_result(ax, grid_original, grid_used,
                       raw_path, final_path,
                       open_log, closed_log,
                       start, goal):
    rows = len(grid_used)
    cols = len(grid_used[0])
    display = _build_display(grid_original, grid_used, open_log, closed_log)
    _draw_display(ax, display, rows, cols)
    _draw_astar_path(ax, raw_path, final_path)
    _draw_endpoints(ax, start, goal)


def static_legend_handles():
    return [
        mpatches.Patch(color='#101010', label='Obstacle'),
        mpatches.Patch(color='#787878', label='U-trap filled'),
        mpatches.Patch(color='#3a75e6', label='Closed list'),
        mpatches.Patch(color='#e02e2e', label='Open list'),
        plt.Line2D([0], [0], color='#999999', lw=1.2,
                   linestyle='--', label='Raw path'),
        plt.Line2D([0], [0], color='#111111', lw=2.0,
                   label='Final path (key nodes)'),
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='#22cc22', ms=8,
                   label='Inflection point'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#11cc11', ms=9,
                   label='Start'),
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='#ffcc00',
                   markeredgecolor='#888800', ms=9,
                   label='Goal'),
    ]

# Hybrid Q-A* animasyonlu GIF ureticisi
class HybridFrameRenderer:
    def __init__(self, grid_original, grid_used, raw_path, final_path,
                 open_log, closed_log, start, goal, env_name=""):
        self.grid_original = [row[:] for row in grid_original]
        self.grid_used = [row[:] for row in grid_used]
        self.raw_path   = list(raw_path)   if raw_path   else []
        self.final_path = list(final_path) if final_path else []
        self.open_log   = set(open_log)    if open_log   else set()
        self.closed_log = set(closed_log)  if closed_log else set()
        self.start = start
        self.goal  = goal
        self.env_name = env_name
        self.frames = []   # RGB array listesi

    def capture(self, robot_pos, dyn_obstacles, ray_endpoints,
                robot_trail, local_target,
                is_q_mode=False, step=0, total_reward=0):
        rows = len(self.grid_used)
        cols = len(self.grid_used[0])

        # Figur boyutunu harita boyutuna gore orantila (legend icin sagda +3 inc)
        cell_px = 0.38 if max(rows, cols) <= 10 else 0.24
        fig_w = max(7.5, cols * cell_px + 3.2)
        fig_h = max(5.5, rows * cell_px + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # Temel harita + open/closed katmanlari
        display = _build_display(self.grid_original, self.grid_used,
                                 self.open_log, self.closed_log)
        _draw_display(ax, display, rows, cols)

        # A* planlanan kuresel rota
        if self.raw_path and len(self.raw_path) > 1:
            rx = [p[0] for p in self.raw_path]
            ry = [p[1] for p in self.raw_path]
            ax.plot(rx, ry, '--', color='#111111', lw=0.9,
                    alpha=0.5, zorder=5, label='_nolegend_')

        if self.final_path and len(self.final_path) > 1:
            fx = [p[0] for p in self.final_path]
            fy = [p[1] for p in self.final_path]
            ax.plot(fx, fy, '--', color='#22aa22', lw=2.0,
                    zorder=6, label='_nolegend_')

        # Start ve goal
        _draw_endpoints(ax, self.start, self.goal)

        # Dinamik engeller — siyah dikdortgen 
        sq = 0.72  # kare yari-boyutu (hucre birimiyle)
        for obs in dyn_obstacles:
            ox, oy = obs['pos'][0], obs['pos'][1]
            rect = plt.Rectangle(
                (ox - sq / 2, oy - sq / 2), sq, sq,
                facecolor='#111111', edgecolor='#555555',
                linewidth=0.8, zorder=11
            )
            ax.add_patch(rect)

        # Robot — buyuk kirmizi daire 
        ax.plot(robot_pos[0], robot_pos[1], 'o',
                color='#e02222', ms=14, zorder=14,
                markeredgecolor='white', markeredgewidth=1.5)

        # Raycast isinlari — siyah yari-saydam, 12 adet 
        if ray_endpoints:
            for (ex, ey) in ray_endpoints:
                ax.plot([robot_pos[0], ex], [robot_pos[1], ey],
                        '-', color='#111111', lw=0.7,
                        alpha=0.35, zorder=10)

        # Yerel hedef — mavi ucgen 
        if local_target and local_target != self.goal:
            ltx, lty = local_target
            ax.plot(ltx, lty, '^',
                    color='#1a6fcc', ms=12, zorder=13,
                    markeredgecolor='white', markeredgewidth=1.0)

        # Q-Learning yorungesi
        if robot_trail and len(robot_trail) > 1:
            tx = [p[0] for p in robot_trail]
            ty = [p[1] for p in robot_trail]
            ax.plot(tx, ty, '-', color='#111111', lw=2.2,
                    alpha=0.80, zorder=12)

        # Mod basligi
        if is_q_mode:
            mode_text  = 'Q-Learning Mode: Evading Dynamic Obstacle!'
            mode_color = '#cc2222'  # kirmizi
        else:
            mode_text  = 'A-Star Mode: Following Global Route'
            mode_color = '#228822'  # yesil

        ax.set_title(
            f'Hybrid Q-A* — {self.env_name}\n'
            f'Step {step}  |  Reward {total_reward:+d}',
            fontsize=9, pad=4
        )
        # Mod etiketi: figurun ustunde renkli bir bant
        fig.text(0.50, 0.975, mode_text,
                 ha='center', va='top',
                 fontsize=8, fontweight='bold',
                 color='white',
                 bbox=dict(facecolor=mode_color, edgecolor='none',
                           boxstyle='round,pad=0.25', alpha=0.88))

        legend_handles = _hybrid_legend_handles()
        ax.legend(handles=legend_handles,
                  loc='upper left',
                  bbox_to_anchor=(1.01, 1.0),
                  fontsize=7.5,
                  framealpha=0.95,
                  edgecolor='#cccccc',
                  borderpad=0.6,
                  labelspacing=0.45)

        plt.tight_layout(rect=[0, 0, 0.78, 0.96])

        # RGB array'e cevir
        fig.canvas.draw()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgba = rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        self.frames.append(rgba[..., :3].copy())
        plt.close(fig)

    # GIF kaydet
    def save_gif(self, path, fps=6):
        if not self.frames:
            return
        try:
            import imageio
        except ImportError:
            print(f'[uyari] imageio bulunamadi; gif kaydedilemedi: {path}')
            return
        imageio.mimsave(path, self.frames, duration=1.0 / fps, loop=0)

# Legend tanimlari
def _hybrid_legend_handles():
    return [
        # Harita katmanlari
        mpatches.Patch(facecolor='#101010', label='Obstacle'),
        mpatches.Patch(facecolor='#787878', label='U-trap filled'),
        mpatches.Patch(facecolor='#3a75e6', label='Closed list'),
        mpatches.Patch(facecolor='#e02e2e', label='Open list'),
        # Yol
        plt.Line2D([0], [0], color='#22aa22', lw=1.8,
                   linestyle='--', label='A* planned route'),
        plt.Line2D([0], [0], color='#ff8000', lw=2.0,
                   label='Q-Learning trajectory'),
        # Robot ve hedefler
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#e02222', ms=9,
                   markeredgecolor='white',
                   label='Robot'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#11cc11', ms=8,
                   markeredgecolor='white',
                   label='Start'),
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='#ffcc00',
                   markeredgecolor='#888800', ms=8,
                   label='Goal'),
        plt.Line2D([0], [0], marker='^', color='w',
                   markerfacecolor='#1a6fcc', ms=9,
                   markeredgecolor='white',
                   label='Local target'),
        # Dinamik engel
        mpatches.Patch(facecolor='#111111', edgecolor='#555555',
                       label='Dynamic obstacle'),
        # Raycast
        plt.Line2D([0], [0], color='#111111', lw=0.8,
                   alpha=0.45, label='Raycast'),
    ]