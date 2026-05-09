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
    
    # Durum Uzayi: 12 Statik + 12 Dinamik + 1 Yon Hedef + 1 Mesafe
    def _get_state(self):
        # 1. Statik ve Dinamik Haritaları Ayır
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
        self.ray_endpoints = [] # Görselleştirme için

        # 2. Her 30 derecede bir TEK BİR IŞIN at (Gerçekçi Occlusion Fiziği)
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
                
                # Sınır dışına çıkarsa (Statik duvar kabul et)
                if not (0 <= tx < self.width and 0 <= ty < self.height):
                    hit_type = "static"
                    hit_dist = dist
                    break
                
                # İLK ÖNCE DİNAMİK ENGELE ÇARPARSA
                if dynamic_grid[ty, tx] == 1:
                    hit_type = "dynamic"
                    hit_dist = dist
                    break
                    
                # İLK ÖNCE STATİK ENGELE ÇARPARSA
                if static_grid[ty, tx] == 1:
                    hit_type = "static"
                    hit_dist = dist
                    break
                    
                dist += 0.5
            
            # Görselleştirme için son noktayı kaydet
            end_x = rx + cx * (hit_dist - 0.5)
            end_y = ry + cy * (hit_dist - 0.5)
            self.ray_endpoints.append((end_x, end_y))
            
            # 3. KATEGORİLEME (Binning)
            if hit_type == "static":
                if hit_dist <= 1.0: static_ray_bins.append(0)
                elif hit_dist <= 2.0: static_ray_bins.append(1)
                else: static_ray_bins.append(2)
                dynamic_ray_bins.append(2) # Arkasını göremediği için güvenli
                
            elif hit_type == "dynamic":
                if hit_dist <= 1.0: dynamic_ray_bins.append(0)
                elif hit_dist <= 2.0: dynamic_ray_bins.append(1)
                else: dynamic_ray_bins.append(2)
                static_ray_bins.append(2) # Arkasını göremediği için güvenli
                
            else:
                static_ray_bins.append(2)
                dynamic_ray_bins.append(2)

        # 4. Yerel Hedef Açısı (Target Bin)
        tx = self.local_target[0]
        ty = self.local_target[1]
        
        angle_to_target = math.degrees(math.atan2(ty - ry, tx - rx))
        normalized_angle = (angle_to_target + 360) % 360
        target_bin = int((normalized_angle + 22.5) // 45) % 8
        
        # 5. YENİ: Hedefe Uzaklık (Q-Tablosu için Gruplanmış/Ayrıklaştırılmış)
        dist_t = math.hypot(tx - rx, ty - ry)
        if dist_t <= 2.0:
            dist_bin = 0 # Çok Yakın
        elif dist_t <= 5.0:
            dist_bin = 1 # Orta Mesafede
        elif dist_t <= 10.0:
            dist_bin = 2 # Uzak
        else:
            dist_bin = 3 # Çok Uzak
        
        # Toplam 26 Elemanlı State Tuple'ı
        return tuple(static_ray_bins + dynamic_ray_bins + [target_bin, dist_bin])

    # Step Fonksiyonu ve Odul Sistemi 
    # Step Fonksiyonu ve Odul Sistemi 
    def step(self, action_idx):
        dx, dy = ACTIONS[action_idx]
        nx = self.robot_pos[0] + dx
        ny = self.robot_pos[1] + dy
        reward = -1 # Adim cezasi
        done = False
        
        is_out_of_bounds = not (0 <= nx < self.width and 0 <= ny < self.height)

        # Sinir disi veya statik duvara carpma 
        if is_out_of_bounds or self.grid[ny, nx] == 1:
            reward = -100
            done = True
            
            if not is_out_of_bounds:
                self.robot_pos = [nx, ny]  # Carpma ani gozlemleme
                self.robot_trail.append((nx, ny))
                
            self._step_dyn_obstacles()
            return self._get_state(), reward, done

        # Eski pozisyonlari kaydet (Es zamanli kontrol icin)
        old_robot = tuple(self.robot_pos)
        old_dyn = [tuple(obs['pos']) for obs in self.dyn_state]

        # Robot hareketi
        old_dist = math.hypot(self.local_target[0] - self.robot_pos[0],
                              self.local_target[1] - self.robot_pos[1])
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
                reward += 50
                self.waypoint_idx += 1

        self._step_dyn_obstacles() # Dinamik engelleri hareket ettir

        # 4) Esanli carpma kontrolu (sadece done degilse)
        if not done:
            new_robot = tuple(self.robot_pos)
            for i, obs in enumerate(self.dyn_state):
                new_obs = tuple(obs['pos'])
                old_obs = old_dyn[i]

                # Senaryo A: Ayni kareye giris (Simultaneous Arrival)
                if new_robot == new_obs:
                    reward = -500
                    done = True
                    break

                # Senaryo B: Swap / Kafa kafaya (yer degistirme)
                if new_robot == old_obs and new_obs == old_robot:
                    reward = -500
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
def _has_dynamic_threat(state):
    # state[12:24] = Dinamik isinlarin gruplanmis degerleri (0, 1, 2)
    # Eger 2'den kucuk bir deger varsa (0 veya 1), engel gorus alanindadir.
    dynamic_rays = state[12:24]
    return any(r < 2 for r in dynamic_rays)

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
            if _has_dynamic_threat(state):
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
def run_episode(agent, env, max_steps=400, frame_renderer=None, episode_label="", log_path=None):
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

    log_file = None
    if log_path:
        log_file = open(log_path, "w", encoding="utf-8")
        log_file.write(f"=== Episode: {episode_label} ===\n")
        log_file.write(f"Start: {env.start} | Goal: {env.goal}\n")
        log_file.write(f"{'Step':<5} | {'Robot':<10} | {'Action':<10} | {'Mode':<5} | {'Reward':<7} | {'Total':<7} | {'Dyn Obs Positions'}\n")
        log_file.write("-" * 90 + "\n")

    while not done and steps < max_steps:
        is_q = _has_dynamic_threat(state)
        if is_q:
            action = agent.choose_action(state, explore=False)
        else:
            action = _greedy_astar_action(env)
            
        old_pos = tuple(env.robot_pos)
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        
        if log_file:
            obs_str = ", ".join([str(tuple(o['pos'])) for o in env.dyn_state])
            mode_str = "Q" if is_q else "A*"
            action_str = str(ACTIONS[action])
            log_file.write(f"{steps:<5} | {str(tuple(env.robot_pos)):<10} | {action_str:<10} | {mode_str:<5} | {reward:<7} | {total_reward:<7} | {obs_str}\n")
        
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
    
    if log_file:
        log_file.write("-" * 90 + "\n")
        log_file.write(f"Final Status: {'SUCCESS' if success else 'FAILURE'}\n")
        log_file.close()

    return {
        'steps': steps,
        'reward': total_reward,
        'success': success,
        'final_pos': tuple(env.robot_pos)
    }