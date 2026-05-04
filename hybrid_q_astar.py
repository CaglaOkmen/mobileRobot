"""
Hybrid Q-A* Algoritmasi:

1. Hibrit Mimari: Q-Tablosu (Tehlike Algilama) + A* (Hizli Yol Planlama)
2. Dinamik Engeller: Simule Edilmis Hareketli Engeller
3. Ray-Casting: Cevre Algilama (5 Metre Mesafe)
4. Optimizasyon: Hata Orani (Error Rate) ile Egitim Kontrolu
"""
import math
import os
import pickle
import random
import numpy as np
from collections import defaultdict

# 8 yon Hareket
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

# ---------------------------------------------------------------------------
# Hybrid Q-A* Ortami
# ---------------------------------------------------------------------------
class RaycastAgentEnv:
    #Ortam Sinifini Tanimla 
    def __init__(self, grid_static, start, goal, astar_data, dyn_specs):
        self.grid = np.asarray(grid_static, dtype=np.int8)
        self.height = self.grid.shape[0]
        self.width = self.grid.shape[1]
        
        self.start = tuple(start)
        self.goal = tuple(goal)
        
        self.final_path = astar_data['final_path']
        
        # Yol Varsa Waypoint'leri A* Yolundan Al
        if self.final_path: 
            self.waypoints = list(self.final_path) 
        else:
            self.waypoints = [self.goal] 
            
        self.dyn_specs = []
        for d in dyn_specs:
            self.dyn_specs.append(dict(d)) 
            
        self.reset() # Ortami Resetle 

    # Ortami Resetleme Fonksiyonu
    def reset(self):
        self.robot_pos = list(self.start)
        self.robot_trail = [tuple(self.start)]
        
        # Waypoint index 
        if len(self.waypoints) > 1:
            self.waypoint_idx = 1
        else:
            self.waypoint_idx = 0
            
        # Dinamik Engelleri Baslat
        self.dyn_state = []
        for d in self.dyn_specs:
            mode = d['mode']
            
            if mode == 'horizontal':
                vec = (1, 0)
            elif mode == 'vertical':
                vec = (0, 1)
            else:
                vec = (0, 0)
                
            # Rastgele engeller icin seed 
            seed = d.get('seed', 0)
            rng = random.Random(seed) if mode == 'random' else None 
            
            self.dyn_state.append({
                'pos': list(d['pos']),
                'mode': mode,
                'vec': vec,
                'rng': rng
            })
            
        return self._get_state()

    # Yerel Hedefi Belirle
    @property
    def local_target(self):
        target_index = min(self.waypoint_idx, len(self.waypoints) - 1)
        return self.waypoints[target_index]

    # Dinamik Engelleri Hareket Ettir 
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

    # Aktif Haritayi Olustur (Dinamik Engellerle) 
    def _active_grid(self):
        g = self.grid.copy() # Static haritayi kopyala
        for obs in self.dyn_state: # Dinamik engelleri isaretle
            ox = obs['pos'][0]
            oy = obs['pos'][1]
            if 0 <= ox < self.width and 0 <= oy < self.height:
                g[oy, ox] = 1
        return g 
    
    # Durum Uzayi: 12 Ray Mesafesi + 8 Yon Hedef Acisi
    def _get_state(self):
        active_grid = self._active_grid()
        rx = self.robot_pos[0]
        ry = self.robot_pos[1]
        
        ray_dists = []
        self.ray_endpoints = []

        for angle in range(0, 360, 30): # 30 derecelik araliklarla 12 isin
            rad = math.radians(angle)
            cx = math.cos(rad)
            cy = math.sin(rad)
            dist = 0.5
            
            while dist <= 3.0: # 3 adim uzaklik
                tx = int(round(rx + cx * dist))
                ty = int(round(ry + cy * dist))
                
                # Sinir kontrolu 
                is_out_of_bounds = not (0 <= tx < self.width and 0 <= ty < self.height)
                if is_out_of_bounds or active_grid[ty, tx] == 1:
                    break
                dist += 0.5
                
            end_x = rx + cx * (dist - 0.5) 
            end_y = ry + cy * (dist - 0.5) 
            self.ray_endpoints.append((end_x, end_y))
            
            if dist <= 1.0:
                ray_dists.append(0)
            elif dist <= 2.0:
                ray_dists.append(1)
            else:
                ray_dists.append(2)

        tx = self.local_target[0]
        ty = self.local_target[1]
        
        angle_to_target = math.degrees(math.atan2(ty - ry, tx - rx))
        normalized_angle = (angle_to_target + 360) % 360
        target_bin = int((normalized_angle + 22.5) // 45) % 8
        
        return tuple(ray_dists + [target_bin])

    # Step Fonksiyonu ve Odul Sistemi 
    def step(self, action_idx):
        dx, dy = ACTIONS[action_idx]
        nx = self.robot_pos[0] + dx
        ny = self.robot_pos[1] + dy
        
        active_grid = self._active_grid()
        reward = -1 # Adim cezasi
        done = False
        
        is_out_of_bounds = not (0 <= nx < self.width and 0 <= ny < self.height)

        # Yol disina cikar veya engele carparsa
        if is_out_of_bounds or active_grid[ny, nx] == 1: 
            reward = -100
            done = True
        else:
            # Hedefe uzaklik hesaplama
            old_dist = math.hypot(self.local_target[0] - self.robot_pos[0], self.local_target[1] - self.robot_pos[1]) 
            
            self.robot_pos = [nx, ny]
            self.robot_trail.append((nx, ny))
            
            new_dist = math.hypot(self.local_target[0] - nx, self.local_target[1] - ny)
            
            if new_dist < old_dist:
                reward += 10 # Hedefe yakinlasinca odul
            else:
                reward -= 10 # Hedeften uzaklasinca ceza

            if (nx, ny) == self.goal:
                reward = 500 # Hedefe varinca buyuk odul
                done = True
            elif (nx, ny) == self.local_target:
                if self.waypoint_idx < len(self.waypoints) - 1:
                    reward += 50 # Waypoint'i gectiyse odul
                    self.waypoint_idx += 1  # Bir sonraki waypoint'e gec

        self._step_dyn_obstacles() # Dinamik engelleri hareket ettir
        
        if not done:
            for obs in self.dyn_state:
                if obs['pos'] == self.robot_pos:
                    reward = -500 # Dinamik engele carparsa ceza 
                    done = True
                    break

        return self._get_state(), reward, done

# ---------------------------------------------------------------------------
# Q-Learning Ajan Sinifi
# ---------------------------------------------------------------------------
class QLearningAgent:
    def __init__(self, alpha=0.01, gamma=0.9, epsilon=0.99): 
        self.q_table = defaultdict(lambda: [0.0] * len(ACTIONS))
        self.alpha = alpha # Ogrenme orani
        self.gamma = gamma # Indirim orani
        self.epsilon = epsilon # Rastgele hareket orani

    # En uygun hareketi sec
    def choose_action(self, state, explore=True):
        if explore and random.random() < self.epsilon: # Rastgele hareket 
            return random.randint(0, len(ACTIONS) - 1)
            
        return int(np.argmax(self.q_table[state])) # En uygun hareketi sec

    # Q-tablosunu guncelle
    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state][action]
        target = reward + self.gamma * float(np.max(self.q_table[next_state]))
        self.q_table[state][action] += self.alpha * (target - predict)

    # Q-tablosunu kaydet
    def save(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    # Q-tablosunu yukle
    def load(self, path="q_table.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                loaded_table = pickle.load(f)
                self.q_table.update(loaded_table)
            return True
        return False

# Yerel hedefe en kisa hamleyi sec
def _greedy_astar_action(env):
    valid_moves = []
    
    for i, (dx, dy) in enumerate(ACTIONS):
        nx = env.robot_pos[0] + dx
        ny = env.robot_pos[1] + dy
        
        if 0 <= nx < env.width and 0 <= ny < env.height:
            if env.grid[ny, nx] == 0:
                dist = math.hypot(nx - env.local_target[0], ny - env.local_target[1])
                valid_moves.append((i, dist))
                
    if valid_moves:
        best_move = min(valid_moves, key=lambda x: x[1])
        return best_move[0]
        
    return 0

# Dinamik Engel Kontrolu
def _has_dynamic_threat(env, threshold=2.8):
    for obs in env.dyn_state:
        dist = math.hypot(obs['pos'][0] - env.robot_pos[0], obs['pos'][1] - env.robot_pos[1])
        if dist < threshold:
            return True
    return False

# ---------------------------------------------------------------------------
# Q-Learning Egitimi
# ---------------------------------------------------------------------------
def train_on_env(agent, env, episodes, max_steps=300, verbose_every=500):
    decay_rate = 0.00035
    
    for ep in range(episodes):
        state = env.reset()
        steps = 0
        total_reward = 0
        done = False
        
        while not done and steps < max_steps:
            # Dinamik engel varsa Q-Learning, yoksa A*
            if _has_dynamic_threat(env):
                action = agent.choose_action(state, explore=True)
            else:
                action = _greedy_astar_action(env)
                
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
        agent.epsilon = max(0.01, agent.epsilon - decay_rate)
        
        if verbose_every and (ep + 1) % verbose_every == 0:
            print(f"    ep {ep+1}/{episodes}  steps={steps}  reward={total_reward:+d}  eps={agent.epsilon:.3f}  |Q|={len(agent.q_table)}")

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def run_episode(agent, env, max_steps=400, frame_renderer=None, episode_label=""):
    state = env.reset()
    steps = 0
    total_reward = 0
    done = False
    
    if frame_renderer:
        frame_renderer.capture(
            robot_pos=env.robot_pos,
            dyn_obstacles=env.dyn_state,
            ray_endpoints=env.ray_endpoints,
            robot_trail=env.robot_trail,
            local_target=env.local_target,
            is_q_mode=False,
            step=0,
            total_reward=0,
        )
    while not done and steps < max_steps:
        is_q = _has_dynamic_threat(env)
        if is_q:
            action = agent.choose_action(state, explore=False)
        else:
            action = _greedy_astar_action(env)
            
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        
        if frame_renderer:
            frame_renderer.capture(
                robot_pos=env.robot_pos,
                dyn_obstacles=env.dyn_state,
                ray_endpoints=env.ray_endpoints,
                robot_trail=env.robot_trail,
                local_target=env.local_target,
                is_q_mode=is_q,
                step=steps,
                total_reward=total_reward,
            )

    success = done and tuple(env.robot_pos) == env.goal
    
    return {
        'steps': steps,
        'reward': total_reward,
        'success': success,
        'final_pos': tuple(env.robot_pos)
    }