"""
Rastgele, Dinamik ve Prosedurel uretilmis Harita/Senaryo Olusturucu

ozellikler:
- Basit (Hucresel/Bloklu) ve Karmasik (Gurultulu) harita modlari.
- Boyutlar: 10x10, 30x30, 50x50.
- BFS ile yolun kesinlikle ulasilabilir oldugunun garantisi.
- Dinamik engellerin guvenli/gecerli hucrelerde baslatilmasi.
"""
import random
import math
from collections import deque

# UyumluHARITA isimleri
RAW_GRIDS = {
    "10x10_basit": [],
    "10x10_karmasik": [],
    "30x30_basit": [],
    "30x30_karmasik": [],
    "50x50_basit": [],
    "50x50_karmasik": []
}

# Karmasik haritalar icin gurultulu harita algoritmasi
def _generate_complex_grid(size, density, rng):
    grid = [[0 for _ in range(size)] for _ in range(size)]
    
    for i in range(size):
        grid[0][i] = 1
        grid[size - 1][i] = 1
        grid[i][0] = 1
        grid[i][size - 1] = 1
        
    for y in range(1, size - 1):
        for x in range(1, size - 1):
            if rng.random() < density:
                grid[y][x] = 1
                
    return grid

# Basit haritalar icin hucresel otomat algoritmasi
def _generate_simple_grid(size, rng):
    grid = [[1 if rng.random() < 0.40 else 0 for _ in range(size)] for _ in range(size)]

    for _ in range(2):
        new_grid = [[0 for _ in range(size)] for _ in range(size)]
        for y in range(size):
            for x in range(size):
                walls = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= nx < size and 0 <= ny < size:
                            walls += grid[ny][nx]
                        else:
                            walls += 1 
                
                if walls >= 5:
                    new_grid[y][x] = 1
                else:
                    new_grid[y][x] = 0
        grid = new_grid

    for i in range(size):
        grid[0][i] = 1
        grid[size - 1][i] = 1
        grid[i][0] = 1
        grid[i][size - 1] = 1

    return grid

# Ortak kontrol fonksiyonlari
def _get_empty_cells(grid):
    cells = []
    height, width = len(grid), len(grid[0])
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 0:
                cells.append((x, y))
    return cells

def _is_path_reachable(grid, start, goal):
    height, width = len(grid), len(grid[0])
    visited = set([start])
    queue = deque([start])
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

# Bu fonksiyon dikey veya yatayda serbest hareket edebilecek engelleri belirler.
def _get_valid_modes(grid, cx, cy):
    modes = []
    height, width = len(grid), len(grid[0])
    
    # Yatay
    if (cx > 0 and grid[cy][cx - 1] == 0) or (cx < width - 1 and grid[cy][cx + 1] == 0):
        modes.append('horizontal')
    # Dikey
    if (cy > 0 and grid[cy - 1][cx] == 0) or (cy < height - 1 and grid[cy + 1][cx] == 0):
        modes.append('vertical')
    # Rastgele
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
        if 0 <= cx + dx < width and 0 <= cy + dy < height and grid[cy + dy][cx + dx] == 0:
            modes.append('random')
            break
            
    return modes

# Ana senaryo uretici
def get_episode_config(env_name, episode, is_test=False):
    base_seed = 20 if is_test else 2001
    current_seed = base_seed + episode
    rng = random.Random(current_seed)
    
    # env_name Ayristirma (orn: "10x10_basit" -> size: 10, complexity: "basit")
    parts = env_name.split('_')
    size = int(parts[0].split('x')[0])
    complexity = parts[1]
    
    if size == 10:
        min_dist = 4.0
        num_dyn_obs = 2
        density = 0.15 
    elif size == 30:
        min_dist = 15.0
        num_dyn_obs = 12
        density = 0.20 
    else: # 50
        min_dist = 25.0
        num_dyn_obs = 20
        density = 0.20 

    # Gecerli harita bulunana kadar dongu
    while True:
        if complexity == "basit":
            grid = _generate_simple_grid(size, rng)
        else:
            grid = _generate_complex_grid(size, density, rng)
            
        empty_cells = _get_empty_cells(grid)
        
        if len(empty_cells) < 2:
            continue
            
        start = rng.choice(empty_cells)
        goal = rng.choice(empty_cells)
        
        if start == goal: continue
            
        dist = math.hypot(start[0] - goal[0], start[1] - goal[1])
        if dist >= min_dist and _is_path_reachable(grid, start, goal):
            break 
                
    # Dinamik Engelleri Yerlestir
    available_for_obs = []
    valid_modes_dict = {}
    
    for cell in empty_cells:
        if cell == goal: continue
        # Guvenli alan korumasi (Start ve 1 birim cevresi yasak)
        if abs(cell[0] - start[0]) <= 1 and abs(cell[1] - start[1]) <= 1: continue
            
        valid_modes = _get_valid_modes(grid, cell[0], cell[1])
        if not valid_modes: continue
            
        available_for_obs.append(cell)
        valid_modes_dict[cell] = valid_modes
        
    actual_num_obs = min(num_dyn_obs, len(available_for_obs))
    chosen_obs_pos = rng.sample(available_for_obs, actual_num_obs)
    
    dyn_obs_list = []
    for pos in chosen_obs_pos:
        allowed_modes = valid_modes_dict[pos]
        dyn_obs_list.append({
            'pos': list(pos),
            'mode': rng.choice(allowed_modes),
            'seed': rng.randint(0, 100000)
        })
        
    return {
        "grid": grid,
        "start": list(start),
        "goal": list(goal),
        "dynamic_obstacles": dyn_obs_list
    }

# karistirilmis egitim zamanlayicisi
def get_training_schedule(total_episodes, schedule_seed=42):
    env_names = list(RAW_GRIDS.keys())
    n_envs = len(env_names)

    # Her haritaya esit paylastir, kalani ilk haritalara dagit
    base_count = total_episodes // n_envs
    remainder = total_episodes % n_envs

    schedule = []
    for i, env_name in enumerate(env_names):
        count = base_count + (1 if i < remainder else 0)
        for ep in range(count):
            schedule.append((env_name, ep))

    # Seed'li karistirma — tekrarlanabilir ve harita-bagimsiz sira
    random.Random(schedule_seed).shuffle(schedule)

    return schedule