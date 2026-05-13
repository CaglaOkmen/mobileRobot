import math
import random
import numpy as np

# 8 yon Hareket
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
N_ACTIONS = len(ACTIONS)

# State boyutu: 12 statik ray + 12 dinamik ray + 1 target_bin + 1 distance = 26
STATE_DIM = 26

# ---------------------------------------------------------------------------
# Hybrid D3QN-A* Ortami
# ---------------------------------------------------------------------------
class RaycastAgentEnv:
    def __init__(self, grid_static, start, goal, astar_data, dyn_specs):
        self.grid = np.asarray(grid_static, dtype=np.int8)
        self.height = self.grid.shape[0]
        self.width = self.grid.shape[1]

        self.start = tuple(start)
        self.goal = tuple(goal)

        self.final_path = astar_data['final_path']

        if self.final_path:
            self.waypoints = list(self.final_path)
        else:
            self.waypoints = [self.goal]

        self.dyn_specs = []
        for d in dyn_specs:
            self.dyn_specs.append(dict(d))
        self.wp_threshold = 1.5  # Waypoint radius 
        self.reset()

    def reset(self):
        self.robot_pos = list(self.start)
        self.robot_trail = [tuple(self.start)]

        if len(self.waypoints) > 1:
            self.waypoint_idx = 1
        else:
            self.waypoint_idx = 0

        # Ulasilan en ileri nokta
        self.max_waypoint_reached = self.waypoint_idx

        self.dyn_state = []
        for d in self.dyn_specs:
            mode = d['mode']

            if mode == 'horizontal':
                vec = (1, 0)
            elif mode == 'vertical':
                vec = (0, 1)
            else:
                vec = (0, 0)

            seed = d.get('seed', 0)
            rng = random.Random(seed) if mode == 'random' else None

            self.dyn_state.append({
                'pos': list(d['pos']),
                'mode': mode,
                'vec': vec,
                'rng': rng
            })

        return self._get_state()

    @property
    def local_target(self):
        target_index = min(self.waypoint_idx, len(self.waypoints) - 1)
        return self.waypoints[target_index]

    def _step_dyn_obstacles(self):
        for obs in self.dyn_state:
            if obs['mode'] == 'random':
                dx, dy = obs['rng'].choice(ACTIONS)
            else:
                dx, dy = obs['vec']

            nx = obs['pos'][0] + dx
            ny = obs['pos'][1] + dy

            is_valid_x = 0 <= nx < self.width
            is_valid_y = 0 <= ny < self.height

            if is_valid_x and is_valid_y and self.grid[ny, nx] == 0:
                obs['pos'] = [nx, ny]
            elif obs['mode'] != 'random':
                obs['vec'] = (-dx, -dy)

    def _get_state(self):
        static_grid = self.grid
        dynamic_grid = np.zeros_like(self.grid)

        for obs in self.dyn_state:
            ox, oy = obs['pos']
            if 0 <= ox < self.width and 0 <= oy < self.height:
                dynamic_grid[oy, ox] = 1

        rx = self.robot_pos[0]
        ry = self.robot_pos[1]

        static_ray_bins = []
        dynamic_ray_bins = []
        self.ray_endpoints = []

        # Her 30 derecede 1 ray = 12 ray (toplam)
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            cx = math.cos(rad)
            cy = math.sin(rad)

            hit_type = "none"
            hit_dist = 3.0

            dist = 0.5
            while dist <= 3.0:
                tx = int(round(rx + cx * dist))
                ty = int(round(ry + cy * dist))

                if not (0 <= tx < self.width and 0 <= ty < self.height):
                    hit_type = "static"
                    hit_dist = dist
                    break

                if dynamic_grid[ty, tx] == 1:
                    hit_type = "dynamic"
                    hit_dist = dist
                    break

                if static_grid[ty, tx] == 1:
                    hit_type = "static"
                    hit_dist = dist
                    break

                dist += 0.5

            end_x = rx + cx * (hit_dist - 0.5)
            end_y = ry + cy * (hit_dist - 0.5)
            self.ray_endpoints.append((end_x, end_y))

            if hit_type == "static":
                if hit_dist <= 1.0: static_ray_bins.append(0)
                elif hit_dist <= 2.0: static_ray_bins.append(1)
                else: static_ray_bins.append(2)
                dynamic_ray_bins.append(2)

            elif hit_type == "dynamic":
                if hit_dist <= 1.0: dynamic_ray_bins.append(0)
                elif hit_dist <= 2.0: dynamic_ray_bins.append(1)
                else: dynamic_ray_bins.append(2)
                static_ray_bins.append(2)

            else:
                static_ray_bins.append(2)
                dynamic_ray_bins.append(2)

        tx = self.local_target[0]
        ty = self.local_target[1]

        angle_to_target = math.degrees(math.atan2(ty - ry, tx - rx))
        normalized_angle = (angle_to_target + 360) % 360
        target_bin = int((normalized_angle + 22.5) // 45) % 8

        dist_t = math.hypot(tx - rx, ty - ry)
        max_diag = math.hypot(self.width, self.height)
        dist_norm = dist_t / max_diag
        return tuple(static_ray_bins + dynamic_ray_bins + [target_bin, dist_norm])

    def step(self, action_idx):
        dx, dy = ACTIONS[action_idx]
        nx = self.robot_pos[0] + dx
        ny = self.robot_pos[1] + dy
        reward = -1
        done = False

        is_out_of_bounds = not (0 <= nx < self.width and 0 <= ny < self.height)

        if is_out_of_bounds or self.grid[ny, nx] == 1:
            reward = -300
            done = True

            if not is_out_of_bounds:
                self.robot_pos = [nx, ny]
                self.robot_trail.append((nx, ny))

            self._step_dyn_obstacles()
            return self._get_state(), reward, done

        old_robot = tuple(self.robot_pos)
        old_dyn = [tuple(obs['pos']) for obs in self.dyn_state]

        old_dist = math.hypot(self.local_target[0] - self.robot_pos[0],
                              self.local_target[1] - self.robot_pos[1])
        self.robot_pos = [nx, ny]
        self.robot_trail.append((nx, ny))
        new_dist = math.hypot(self.local_target[0] - nx, self.local_target[1] - ny)

        # Yaklasma miktarını bul
        dist_diff = old_dist - new_dist
        
        # Miktarı bir katsayı ile çarparak odule ekle
        reward += dist_diff * 20.0

        if (nx, ny) == self.goal:
            reward = 500
            done = True
        else:
            # Esnek waypoint kontrolu
            dist_to_wp = math.hypot(self.local_target[0] - nx, self.local_target[1] - ny)
            
            if dist_to_wp < self.wp_threshold:
                # Waypoint indeksi dizinin boyutunu asmasın
                if self.waypoint_idx < len(self.waypoints) - 1:
                    # Sadece yeni bir hedefe ilk defa giriyorsa puan ver
                    if self.waypoint_idx >= self.max_waypoint_reached:
                        reward += 50
                        self.max_waypoint_reached = self.waypoint_idx + 1
                    
                    self.waypoint_idx += 1

            else:
                if dist_to_wp > 3.0 and self.waypoint_idx > 1:
                    closest_idx = self.waypoint_idx
                    min_d = dist_to_wp
                    
                    # Indeksin diziyi asma ihtimalini kesin olarak engelle
                    safe_end_idx = min(self.waypoint_idx, len(self.waypoints))
                    search_start = max(0, safe_end_idx - 10)
                    
                    # Guvenli sınırlar içinde tarama yap
                    for i in range(search_start, safe_end_idx):
                        d = math.hypot(self.waypoints[i][0] - nx, self.waypoints[i][1] - ny)
                        if d < min_d:
                            min_d = d
                            closest_idx = i
                    
                    # Hedefi geriye çek
                    if closest_idx < self.waypoint_idx:
                        self.waypoint_idx = min(closest_idx + 1, len(self.waypoints) - 1)

        self._step_dyn_obstacles()

        if not done:
            new_robot = tuple(self.robot_pos)
            for i, obs in enumerate(self.dyn_state):
                new_obs = tuple(obs['pos'])
                old_obs = old_dyn[i]

                if new_robot == new_obs:
                    reward = -500
                    done = True
                    break

                if new_robot == old_obs and new_obs == old_robot:
                    reward = -500
                    done = True
                    break

        return self._get_state(), reward, done

# State Encoder
def encode_state(state_tuple):
    arr = np.zeros(STATE_DIM, dtype=np.float32)

    # 12 statik ray bins (0,1,2) -> normalize
    for i in range(12):
        arr[i] = state_tuple[i] / 2.0

    # 12 dinamik ray bins (0,1,2) -> normalize
    for i in range(12):
        arr[12 + i] = state_tuple[12 + i] / 2.0

    # Target bin (0..7) -> normalize
    arr[24] = state_tuple[24] / 7.0

    # Distance (already normalized in _get_state)
    arr[25] = state_tuple[25]

    return arr