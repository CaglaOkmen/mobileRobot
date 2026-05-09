"""
Improved A* Algorithm (Zhang et al., 2024)
1. Adaptif Maliyet Fonksiyonu | 2. U-Tuzak Doldurma | 3. Yon Kazanci (K) | 4. Anahtar Dugum Secimi
Çikti: raw_path, final_path, open_set, closed_set, grid_used
"""

import math
import numpy as np
from heapq import heappush, heappop
from dataclasses import dataclass, field

# Dugum yapisi
@dataclass(order=True)
class Node:
    f: float
    g: float = field(compare=False)
    h: float = field(compare=False)
    x: int   = field(compare=False)
    y: int   = field(compare=False)
    parent: 'Node' = field(compare=False, default=None)

# ---------------------------------------------------------------------------
# Yardimci Araclar ve Adaptif Formuller
# ---------------------------------------------------------------------------
# Global ve Lokal Engel Oranlari
def init_memory_matrix(grid):
    r, c = len(grid), len(grid[0])
    g_arr = np.asarray(grid, dtype=np.int32)
    p_glob = float(g_arr.sum()) / (r * c)                      # 5
    a = max(1, int((r + c) * (p_glob if p_glob < 0.1 else 0.1)))  # 12

    # Integral image: ip[y+1, x+1] = g_arr[:y+1, :x+1].sum()
    ip = np.zeros((r + 1, c + 1), dtype=np.int64)
    ip[1:, 1:] = g_arr.cumsum(axis=0).cumsum(axis=1)

    ys = np.arange(r)
    xs = np.arange(c)
    Y, X = np.meshgrid(ys, xs, indexing='ij')

    y0 = np.clip(Y - a, 0, r)
    y1 = np.clip(Y + a + 1, 0, r)
    x0 = np.clip(X - a, 0, c)
    x1 = np.clip(X + a + 1, 0, c)

    win_sum  = ip[y1, x1] - ip[y0, x1] - ip[y1, x0] + ip[y0, x0]
    win_size = (y1 - y0) * (x1 - x0)
    mem = win_sum / np.maximum(win_size, 1) # 6

    return mem.astype(float), p_glob, a

# Kosegen (Diagonal) Mesafe
def heuristic(x, y, gx, gy):
    dx, dy = abs(gx - x), abs(gy - y)
    return (math.sqrt(2) - 1) * min(dx, dy) + max(dx, dy) 

# Yon Kazanci (K) 
def get_k_gain(nx, ny, cx, cy, gx, gy):
    ax, ay, bx, by = gx - cx, gy - cy, nx - cx, ny - cy
    ma, mb = math.hypot(ax, ay), math.hypot(bx, by)
    if ma < 1e-9 or mb < 1e-9: return 1.0
    return 2.0 - max(-1.0, min(1.0, (ax*bx + ay*by) / (ma*mb))) 

# ---------------------------------------------------------------------------
# Bolum 3.2: U-sekilli Tuzak Doldurma 
# ---------------------------------------------------------------------------
class VirtualGrid:
    def __init__(self, grid, rot): 
        self.grid = grid
        self.rot = rot
        self.R = len(grid)
        self.C = len(grid[0])
        
    # Rotasyon matrisi: Orijinal gridi dondurur
    def from_rot(self, x, y): 
        if self.rot == 0: return x, y
        if self.rot == 1: return y, self.R - 1 - x
        if self.rot == 2: return self.C - 1 - x, self.R - 1 - y
        if self.rot == 3: return self.C - 1 - y, x
    
    # Rotasyon matrisi: Dondurulmus gridi orijinal grid formatina donusturur      
    def to_rot(self, tx, ty):
        if self.rot == 0: return tx, ty 
        if self.rot == 1: return self.R - 1 - ty, tx
        if self.rot == 2: return self.C - 1 - tx, self.R - 1 - ty
        if self.rot == 3: return ty, self.C - 1 - tx

    # Dondurulmus grid uzerinden okuma islemi        
    def get(self, x, y):
        tx, ty = self.from_rot(x, y)
        if 0 <= ty < self.R and 0 <= tx < self.C:
            return self.grid[ty][tx]
        return 1
    
    # Dondurulmus grid uzerinden yazma islemi    
    def set(self, x, y, val):
        tx, ty = self.from_rot(x, y)
        if 0 <= ty < self.R and 0 <= tx < self.C:
            self.grid[ty][tx] = val
    
    # Boyut hesaplama
    def dims(self):
        if self.rot % 2 == 0: return self.C, self.R
        else: return self.R, self.C

#----------------------------------------------------------------------------
# Bolum 3.2: U-sekilli Tuzak Doldurma Fonksiyonlari
#----------------------------------------------------------------------------
def fill_u_traps(grid, ox, oy, start, goal, curr):
    any_filled = False

    for rot in range(4):
        vgrid = VirtualGrid(grid, rot)
        c, r = vgrid.dims()
        # Optimal node, start, goal rotasyon uzayinda.
        rx,     ry     = vgrid.to_rot(ox,       oy)
        s_rx,   s_ry   = vgrid.to_rot(start[0], start[1])
        g_rx,   g_ry   = vgrid.to_rot(goal[0],  goal[1])
        sg_set = {(s_rx, s_ry), (g_rx, g_ry)}

        if not (0 <= ry < r):
            continue

        max_d = max(r, c) // 2 + 1
        py_candidates = [ry]
        for d in range(1, max_d):
            if ry + d < r: py_candidates.append(ry + d)
            if ry - d >= 0: py_candidates.append(ry - d)

        rotation_filled = False
        for py in py_candidates:
            if rotation_filled:
                break

            px = rx
            while px >= 0:
                if vgrid.get(px, py) == 1:
                    # Engele çarpıldı; bu satırda U-trap yok.
                    break

                # ------ Step 2: P1 kosulu (sol+alt komsu duvar ve sol-alt kose) ------
                if vgrid.get(px - 1, py) == 1 and vgrid.get(px, py + 1) == 1 and vgrid.get(px - 1, py + 1) == 1:
                    # Filtre 1 - Sınır duvar:
                    lx, ly = vgrid.from_rot(px - 1, py)
                    bx, by = vgrid.from_rot(px,     py + 1)
                    if (lx == 0 or lx == vgrid.C - 1 or
                            ly == 0 or ly == vgrid.R - 1 or
                            bx == 0 or bx == vgrid.C - 1 or
                            by == 0 or by == vgrid.R - 1):
                        px -= 1; continue

                    p1 = (px, py)

                    # ------ Step 2 devam: P2 ara ------
                    p2 = None
                    for y in range(py - 1, -1, -1):
                        if vgrid.get(px - 1, y) != 1: break
                        if y - 1 < 0: break
                        if vgrid.get(px, y - 1) == 1:
                            if vgrid.get(px - 1, y - 1) != 1:
                                break  # Kose boslugu varsa tuzak tam kapali degildir
                            ux, uy = vgrid.from_rot(px, y - 1)
                            if (ux == 0 or ux == vgrid.C - 1 or
                                    uy == 0 or uy == vgrid.R - 1):
                                break
                            p2 = (px, y); break

                    if not p2:
                        px -= 1; continue

                    # ------ Step 3: P3 ve P4 ------
                    p3 = None
                    for cx in range(p2[0] + 1, c):
                        wall_up    = vgrid.get(cx,     p2[1] - 1) == 1
                        wall_right = vgrid.get(cx + 1, p2[1])     == 1
                        if wall_up and not wall_right: continue
                        if wall_up and wall_right: p3 = (cx, p2[1]); break
                        if not wall_up: p3 = (cx - 1, p2[1]); break

                    p4 = None
                    for cx in range(p1[0] + 1, c):
                        wall_down  = vgrid.get(cx,     p1[1] + 1) == 1
                        wall_right = vgrid.get(cx + 1, p1[1])     == 1
                        if wall_down and not wall_right: continue
                        if wall_down and wall_right: p4 = (cx, p1[1]); break
                        if not wall_down: p4 = (cx - 1, p1[1]); break

                    p34_set = {p3, p4} - {None}
                    y_min = min(p1[1], p2[1])
                    y_max = max(p1[1], p2[1])

                    # Filtre 4 - Kenar: tuzak tüm grid yüksekliğini kaplıyorsa
                    # gerçek tuzak değildir.
                    if y_min <= 1 and y_max >= r - 2:
                        break  # bu satırı bırak, dış for'da bir sonraki py

                    # ------ Step 4: Pd katmanlarını doldur ------
                    offset = 0
                    while px + offset < c:
                        pd  = [(px + offset,     y) for y in range(y_min, y_max + 1)]
                        pd1 = [(px + offset + 1, y) for y in range(y_min, y_max + 1)
                               if px + offset + 1 < c]
                        if not pd: break

                        # Fig.5e: Pd'de P3/P4 → Pd doldur ve dur
                        if set(pd) & p34_set:
                            for cx, cy in pd:
                                if vgrid.get(cx, cy) == 0:
                                    vgrid.set(cx, cy, 1); any_filled = True
                            break

                        # Fig.5g: Pd'de start/goal → dur
                        if set(pd) & sg_set:
                            break

                        # Fig.5f: Pd1'de engel → Pd doldur ve dur
                        if any(vgrid.get(cx, cy) == 1 for cx, cy in pd1):
                            for cx, cy in pd:
                                if vgrid.get(cx, cy) == 0:
                                    vgrid.set(cx, cy, 1); any_filled = True
                            break

                        # Fig.5g: Pd1'de start/goal → Pd doldur ve dur
                        if set(pd1) & sg_set:
                            for cx, cy in pd:
                                if vgrid.get(cx, cy) == 0:
                                    vgrid.set(cx, cy, 1); any_filled = True
                            break

                        # Fig.5e: Pd1'de P3/P4 → Pd+Pd1 doldur ve dur
                        if set(pd1) & p34_set:
                            for cx, cy in pd + pd1:
                                if vgrid.get(cx, cy) == 0:
                                    vgrid.set(cx, cy, 1); any_filled = True
                            break

                        # Fig.5d: Hiçbiri yok → Pd doldur, ilerle
                        for cx, cy in pd:
                            if vgrid.get(cx, cy) == 0:
                                vgrid.set(cx, cy, 1); any_filled = True
                        offset += 1

                    # Bu rotasyon için P1 işlendi; aynı rotasyonda başka
                    # py denemiyoruz (zaten dolgu yapıldı veya bilinçli durduk).
                    rotation_filled = True
                    break

                px -= 1  # P1 yok, sola

    return any_filled

# U-sekilli tuzak doldurma sonucu olusan engelleri listeler
def report_filled_cells(grid_orig, grid_used):
    filled = []
    rows, cols = len(grid_orig), len(grid_orig[0])
    for y in range(rows):
        for x in range(cols):
            if grid_orig[y][x] == 0 and grid_used[y][x] == 1:
                filled.append((x, y))
    return filled

# ---------------------------------------------------------------------------
# Bolum 3.4: Algoritma 1 - Anahtar Dugum Secimi (Yol Duzlestirme)
# ---------------------------------------------------------------------------
def _segment_blocked_by_grid(ax, ay, bx, by, grid):
    """A→B doğru parçası bir engel hücresinin İÇİNDEN geçiyor mu?
    Supercover (Amanatides–Woo) hat algoritması: segmentin değdiği
    her hücreyi sırayla ziyaret eder. Hücre kareleri 1×1 ve merkezleri
    tamsayı koordinatlardadır.
    """
    rows, cols = len(grid), len(grid[0])
    # Segmenti birim adımlarda örneklemek yerine matematiksel hat yürüyüşü:
    # (ax,ay) ve (bx,by) tamsayı düğüm merkezleridir; aralarında segmentin
    # geçtiği hücreleri belirleriz.
    x0, y0 = ax + 0.5, ay + 0.5    # hücre içi parametrik konum
    x1, y1 = bx + 0.5, by + 0.5
    dx, dy = x1 - x0, y1 - y0
    n = max(abs(bx - ax), abs(by - ay))
    if n == 0:
        return False
    # n+1 örnek noktada ve aralarındaki yarı-noktalarda hücre kontrolü:
    # yeterince yoğun örnekleme yapalım (her birim için 4 örnek).
    samples = max(4 * n, 8)
    for k in range(samples + 1):
        t = k / samples
        sx = x0 + dx * t
        sy = y0 + dy * t
        cx, cy = int(sx), int(sy)
        if 0 <= cy < rows and 0 <= cx < cols and grid[cy][cx] == 1:
            if (cx, cy) == (ax, ay) or (cx, cy) == (bx, by):
                continue
            return True
    return False

# Anahtar Dugum Secimi
def key_node_selection(path, grid, obs_r=0.5, rob_r=0.5):
    if len(path) < 3: return list(path)
    pl = list(path)
    i = 0

    while i < len(pl) - 2:
        p1, p2, p3 = pl[i], pl[i+1], pl[i+2]
        # Doğrusallık (Collinearity) Kontrolü
        det = p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])
        if abs(det) < 1e-6:
            pl.pop(i + 1); continue

        # P1 → P3 SEGMENT'i bir engel hücresinden geçiyor mu?
        blocked = _segment_blocked_by_grid(p1[0], p1[1], p3[0], p3[1], grid)

        if blocked: i += 1
        else: pl.pop(i + 1)
    return pl

# ---------------------------------------------------------------------------
# Ana Algoritma: Improved A*
# ---------------------------------------------------------------------------
def improved_astar(grid_input, start, goal, obs_r=0.5, rob_r=0.5, use_trap=True, use_key=True):
    grid = [row[:] for row in grid_input]
    grid_orig = [row[:] for row in grid_input]   # U-fill ÖNCESİ saf harita
    rows, cols = len(grid), len(grid[0])
    sx, sy, gx, gy = start[0], start[1], goal[0], goal[1]

    # Hafiza matrisi ve oranlar
    mem, p_glob, a = init_memory_matrix(grid)
    
    open_heap, open_dict, closed_set = [], {}, set()
    closed_log = set()

    # Baslangic dugumu 
    s_node = Node(f=0, g=0, h=heuristic(sx, sy, gx, gy), x=sx, y=sy)
    s_node.f = s_node.g + math.exp(p_glob - mem[sy][sx]) * 1.0 * s_node.h
    heappush(open_heap, s_node); open_dict[(sx, sy)] = s_node

    # Sekiz yon hereketi 
    moves = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]

    while open_heap:
        curr = heappop(open_heap)
        cx, cy = curr.x, curr.y

        if (cx, cy) in closed_set: continue

        if (cx, cy) == (gx, gy): # Hedefe ulasildi
            path = []; n = curr
            while n: path.append((n.x, n.y)); n = n.parent
            raw_path = path[::-1]
            final_path = key_node_selection(raw_path, grid_orig, obs_r, rob_r) if use_key else raw_path
            return raw_path, final_path, set(open_dict.keys()), closed_log, grid

        closed_set.add((cx, cy)); closed_log.add((cx, cy))
        open_dict.pop((cx, cy), None)

        if use_trap: # U-Tuzaklari doldur ve matrisi guncelle
            is_grid_changed = fill_u_traps(grid, cx, cy, start, goal, (cx, cy))
            if is_grid_changed:
                mem, _, _ = init_memory_matrix(grid)

        # Yerel engel oranini al ve global oran ile agirlikli bir sekilde carpan olarak kullan
        p_local_curr = mem[cy][cx]
        weight_curr = math.exp(p_glob - p_local_curr)

        # Sekiz yone hareket ile komsulari genislet
        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < cols and 0 <= ny < rows) or grid[ny][nx] == 1 or (nx, ny) in closed_set: continue 
            
            # Yeni dugumun maliyetini hesapla (11)
            g_new = curr.g + (math.sqrt(2) if dx and dy else 1.0)
            h_new = heuristic(nx, ny, gx, gy)
            k_gain = get_k_gain(nx, ny, cx, cy, gx, gy)
            f_new = g_new + weight_curr * k_gain * h_new

            if (nx, ny) not in open_dict or f_new < open_dict[(nx, ny)].f:
                n_node = Node(f=f_new, g=g_new, h=h_new, x=nx, y=ny, parent=curr)
                open_dict[(nx, ny)] = n_node
                heappush(open_heap, n_node)

    return [], [], set(open_dict.keys()), closed_log, grid

# ---------------------------------------------------------------------------
# Performans Metrikleri Hesaplayicilari
# ---------------------------------------------------------------------------

# Yol uzunlugu
def path_length(p): return sum(math.hypot(p[i+1][0]-p[i][0], p[i+1][1]-p[i][1]) for i in range(len(p)-1))

# Bukum dugumleri
def count_inflection_nodes(p): return sum(1 for i in range(1, len(p)-1) if (p[i][0]-p[i-1][0], p[i][1]-p[i-1][1]) != (p[i+1][0]-p[i][0], p[i+1][1]-p[i][1]))

# Donme acisi
def total_turning_angle(p):
    tot = 0.0
    for i in range(1, len(p)-1):
        dx1, dy1 = p[i][0]-p[i-1][0], p[i][1]-p[i-1][1]
        dx2, dy2 = p[i+1][0]-p[i][0], p[i+1][1]-p[i][1]
        cos_a = max(-1.0, min(1.0, (dx1*dx2 + dy1*dy2) / (math.hypot(dx1,dy1)*math.hypot(dx2,dy2) + 1e-12)))
        tot += math.degrees(math.acos(cos_a)) if math.degrees(math.acos(cos_a)) > 1e-6 else 0
    return tot