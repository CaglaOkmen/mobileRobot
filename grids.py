"""
Farkli test ortamlari ve grid haritalari (Tamamen Rastgele ve Dinamik Versiyon)

Bu dosya artik sabit start/goal pozisyonlari tutmaz. 
Bunun yerine her episode icin rastgele, guvenli ve ulasilabilir 
yeni bir konfigurasyon ureten get_episode_config() fonksiyonunu barindirir.
"""
import random
import math
from collections import deque

# ---------------------------------------------------------------------------
# 1. GRID TANIMLAMALARI
# ---------------------------------------------------------------------------

# Sadece 1 adet 10x10 Harita birakildi
GRID_10X10_U_DOWN = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# İki adet 30x30 Harita
def create_30x30_grid():
    grid = [[0 for _ in range(30)] for _ in range(30)]
    for i in range(30):
        grid[0][i] = 1
        grid[29][i] = 1
        grid[i][0] = 1
        grid[i][29] = 1
    return grid

GRID_30X30_U_DOWN = create_30x30_grid()
# Merkezde buyuk U-Tuzak (asagiya acik)
for c in range(10, 21):
    GRID_30X30_U_DOWN[10][c] = 1
for r in range(10, 20):
    GRID_30X30_U_DOWN[r][10] = 1
    GRID_30X30_U_DOWN[r][20] = 1

GRID_30X30_NO_U_DOWN = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Sadece grid matrislerini tutan kucuk bir sozluk
RAW_GRIDS = {
    "10x10_U_DOWN": GRID_10X10_U_DOWN,
    "30x30_U_DOWN": GRID_30X30_U_DOWN,
    "30x30_NO_U_DOWN": GRID_30X30_NO_U_DOWN
}

# ---------------------------------------------------------------------------
# 2. ALGORİTMALAR (Boşluk Bulma ve Ulaşılabilirlik Testi)
# ---------------------------------------------------------------------------
def _get_empty_cells(grid):
    cells = []
    height = len(grid)
    width = len(grid[0])
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 0:
                cells.append((x, y))
    return cells

def _is_path_reachable(grid, start, goal):
    height = len(grid)
    width = len(grid[0])
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == goal:
            return True
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == 0:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return False

# ---------------------------------------------------------------------------
# 3. DİNAMİK BÖLÜM (EPISODE) ÜRETİCİSİ
# ---------------------------------------------------------------------------
def get_episode_config(env_name, episode, is_test=False):
    """
    Belirli bir episode icin seed'i degistirerek rastgele start, goal ve 
    dinamik engeller uretir.
    
    - is_test = False: Egitim (Train) icin uretim (base_seed=42)
    - is_test = True: Test icin uretim (base_seed=9999)
    """
    
    # Train ve Test asamalari ayni ezberlenmis haritaya dusmesin diye farkli base_seed
    base_seed = 9 if is_test else 999
    current_seed = base_seed + episode
    
    rng = random.Random(current_seed)
    grid = RAW_GRIDS[env_name]
    
    # Haritaya bagli kurallari belirle
    if "10x10" in env_name:
        min_dist = 3.0
        num_dyn_obs = 2
    else: # 30x30
        min_dist = 15.0
        num_dyn_obs = 12
        
    empty_cells = _get_empty_cells(grid)
    
    # 1. Sartlari Saglayan Baslangic ve Hedef Secimi
    while True:
        start = rng.choice(empty_cells)
        goal = rng.choice(empty_cells)
        
        if start == goal:
            continue
            
        # Uzaklik sarti
        dist = math.hypot(start[0] - goal[0], start[1] - goal[1])
        if dist >= min_dist:
            # Ulasilabilirlik (BFS) sarti
            if _is_path_reachable(grid, start, goal):
                break
                
    # 2. Dinamik Engellerin Uretilmesi
    available_for_obs = [cell for cell in empty_cells if cell != start and cell != goal]
    actual_num_obs = min(num_dyn_obs, len(available_for_obs))
    
    # Kesisim olmamasi icin rastgele essiz yer secimi
    chosen_obs_pos = rng.sample(available_for_obs, actual_num_obs)
    
    dyn_obs_list = []
    modes = ['horizontal', 'vertical', 'random']
    
    for pos in chosen_obs_pos:
        dyn_obs_list.append({
            'pos': pos,
            'mode': rng.choice(modes),
            'seed': rng.randint(0, 100000) # Dinamik engel yonu/tercihi icin ic seed
        })
        
    return {
        "grid": grid,
        "start": start,
        "goal": goal,
        "dynamic_obstacles": dyn_obs_list
    }