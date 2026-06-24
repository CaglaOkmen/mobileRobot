"""
Hybrid Q-A* Algoritmasi:

1. Hibrit Mimari: Q-Tablosu (Tehlike Algilama) + A* (Hizli Yol Planlama)

D3QN ile karsılastirma yapabilmek icin duzenlenmistir.
"""
import math
import os
import sys
import pickle
import random
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')

# Improved A* ve Gorsellestirme
from improved_astar import improved_astar, path_length
from grids import RAW_GRIDS, get_training_schedule, get_episode_config
from visualization import HybridFrameRenderer

# 8 yon Hareket
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
N_ACTIONS = len(ACTIONS)

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
            
        self.wp_threshold = 1.5  # esnek waypoint
        self.reset()

    # Ortami Resetleme Fonksiyonu
    def reset(self):
        self.robot_pos = list(self.start)
        self.robot_trail = [tuple(self.start)]

        # Waypoint index 
        if len(self.waypoints) > 1:
            self.waypoint_idx = 1
        else:
            self.waypoint_idx = 0

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

    # Yerel Hedefi Belirle
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

            if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny, nx] == 0:
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

        # 12 Isınli Raycast Fizigi (Her 30 derecede bir)
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

        # Yerel Hedef Acisi (Target Bin)
        tx = self.local_target[0]
        ty = self.local_target[1]
        angle_to_target = math.degrees(math.atan2(ty - ry, tx - rx))
        normalized_angle = (angle_to_target + 360) % 360
        target_bin = int((normalized_angle + 22.5) // 45) % 8

        # Tabular Q-Table için Ayrıklaştırılmış Mesafe (Ayrık Tuple yapısı bozulmamalı)
        dist_t = math.hypot(tx - rx, ty - ry)
        if dist_t <= 2.0:   dist_bin = 0
        elif dist_t <= 5.0: dist_bin = 1
        elif dist_t <= 10.0: dist_bin = 2
        else:                dist_bin = 3

        # Hashlenebilir 26 Elemanlı Ayrık State Tuple'ı
        return tuple(static_ray_bins + dynamic_ray_bins + [target_bin, dist_bin])

    def step(self, action_idx):
        dx, dy = ACTIONS[action_idx]
        nx = self.robot_pos[0] + dx
        ny = self.robot_pos[1] + dy
        reward = -1
        done = False

        is_out_of_bounds = not (0 <= nx < self.width and 0 <= ny < self.height)

        # D3QN ile aynı stabil ceza ağırlıkları
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

        # Sürekli Yaklaşma Ödülü Sinyali (+20 * delta_d)
        dist_diff = old_dist - new_dist
        reward += dist_diff * 20.0

        if (nx, ny) == self.goal:
            reward = 500
            done = True
        else:
            dist_to_wp = math.hypot(self.local_target[0] - nx, self.local_target[1] - ny)
            if dist_to_wp < self.wp_threshold:
                if self.waypoint_idx < len(self.waypoints) - 1:
                    if self.waypoint_idx >= self.max_waypoint_reached:
                        reward += 50
                        self.max_waypoint_reached = self.waypoint_idx + 1
                    self.waypoint_idx += 1
            else:
                # Waypoint Recovery (Geriye düşme kurtarıcısı)
                if dist_to_wp > 3.0 and self.waypoint_idx > 1:
                    closest_idx = self.waypoint_idx
                    min_d = dist_to_wp
                    safe_end_idx = min(self.waypoint_idx, len(self.waypoints))
                    search_start = max(0, safe_end_idx - 10)
                    for i in range(search_start, safe_end_idx):
                        d = math.hypot(self.waypoints[i][0] - nx, self.waypoints[i][1] - ny)
                        if d < min_d:
                            min_d = d
                            closest_idx = i
                    if closest_idx < self.waypoint_idx:
                        self.waypoint_idx = min(closest_idx + 1, len(self.waypoints) - 1)

        self._step_dyn_obstacles()

        if not done:
            new_robot = tuple(self.robot_pos)
            for i, obs in enumerate(self.dyn_state):
                new_obs = tuple(obs['pos'])
                old_obs = old_dyn[i]

                if new_robot == new_obs or (new_robot == old_obs and new_obs == old_robot):
                    reward = -500
                    done = True
                    break

        return self._get_state(), reward, done

# ---------------------------------------------------------------------------
# Q-Learning Ajan Sinifi
# ---------------------------------------------------------------------------
class QLearningAgent:
    def __init__(self, alpha=0.0003, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.004):
        self.q_table = defaultdict(lambda: [0.0] * N_ACTIONS)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state][action]
        target = reward + self.gamma * float(np.max(self.q_table[next_state]))
        self.q_table[state][action] += self.alpha * (target - predict)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def save(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path="q_table.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.q_table.update(pickle.load(f))
            return True
        return False

# ---------------------------------------------------------------------------
# Yardımcı Hibrit Kontrolcü Fonksiyonları
# ---------------------------------------------------------------------------
def _greedy_astar_action(env):
    valid_moves = []
    for i, (dx, dy) in enumerate(ACTIONS):
        nx, ny = env.robot_pos[0] + dx, env.robot_pos[1] + dy
        if 0 <= nx < env.width and 0 <= ny < env.height:
            if env.grid[ny, nx] == 0:
                dist = math.hypot(nx - env.local_target[0], ny - env.local_target[1])
                valid_moves.append((i, dist))
    if valid_moves:
        return min(valid_moves, key=lambda x: x[1])[0]
    return 0

def _has_dynamic_threat(state):
    return any(r < 2 for r in state[12:24])

# ---------------------------------------------------------------------------
# Eğitim ve Test Operasyon Döngüleri
# ---------------------------------------------------------------------------
def run_hybrid_qlearning(total_episodes=12000, eval_max_steps=400, gif_fps=6, test_only=False):
    random.seed(20)
    np.random.seed(20)
    
    output_dir = "output_hybrid_qlearning"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Hybrid Tabular Q-Learning {'TEST' if test_only else 'TRAINING'} ===")
    print(f"Environments           : {len(RAW_GRIDS)}")

    agent = QLearningAgent(alpha=0.01, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.004)

    # --- EĞİTİM MODU ---
    if not test_only:
        print(f"Total episodes (target): {total_episodes}")
        print(f"Training schedule      : Random shuffled mixed schedule\n")
        
        train_schedule = get_training_schedule(total_episodes=total_episodes, schedule_seed=42)
        train_progress_logs = []
        
        for i, (env_name, ep) in enumerate(train_schedule):
            config = get_episode_config(env_name, episode=ep, is_test=False)
            astar_result = improved_astar(config['grid'], config['start'], config['goal'])
            
            ast_dict = {
                'raw_path': astar_result[0], 'final_path': astar_result[1],
                'open_log': astar_result[2], 'closed_log': astar_result[3], 'grid_used': astar_result[4]
            }

            env = RaycastAgentEnv(config['grid'], config['start'], config['goal'], ast_dict, config['dynamic_obstacles'])
            
            state = env.reset()
            done = False
            steps = 0
            
            while not done and steps < 300:
                is_q = _has_dynamic_threat(state)
                if is_q:
                    action = agent.choose_action(state, explore=True)
                else:
                    action = _greedy_astar_action(env)

                next_state, reward, done = env.step(action)
                
                if is_q:  # Sadece Q modundaki adımları masaya yatırıp öğretelim
                    agent.learn(state, action, reward, next_state)

                state = next_state
                steps += 1

            if (i + 1) % 50 == 0:
                agent.decay_epsilon()

            if (i + 1) % 500 == 0:
                log_line = f"    Step {i+1}/{total_episodes} | Env: {env_name:<15} | eps={agent.epsilon:.3f} | Table Size={len(agent.q_table)}"
                print(log_line)
                train_progress_logs.append(log_line)

        agent.save(os.path.join(output_dir, "q_table.pkl"))

    # --- TEST MODU ---
    else:
        table_path = os.path.join(output_dir, "q_table.pkl")
        if agent.load(table_path):
            print(f"Başarılı: Eğitilmiş Q-Table yüklendi -> {table_path} (Tablo Boyutu: {len(agent.q_table)})")
        else:
            print(f"\nHATA: {table_path} bulunamadı! Lütfen önce eğitin.")
            return

    # --- DEĞERLENDİRME (EVAL) ORTAMLARI ---
    print("\n=== Preparing Evaluation Environments ===")
    eval_tasks = []
    for ep in range(5):
        for env_name in RAW_GRIDS.keys():
            eval_config = get_episode_config(env_name, episode=ep, is_test=True)
            eval_astar = improved_astar(eval_config['grid'], eval_config['start'], eval_config['goal'])
            
            eval_ast_dict = {
                'raw_path': eval_astar[0], 'final_path': eval_astar[1],
                'open_log': eval_astar[2], 'closed_log': eval_astar[3], 'grid_used': eval_astar[4]
            }
            
            env = RaycastAgentEnv(eval_config['grid'], eval_config['start'], eval_config['goal'], eval_ast_dict, eval_config['dynamic_obstacles'])
            eval_tasks.append({'env_name': env_name, 'ep_idx': ep, 'env': env, 'ast': eval_ast_dict, 'config': eval_config})

    print(f"\n=== Evaluation ({len(eval_tasks)} total test cases) ===")
    summary_lines = [
        "Hybrid Tabular Q-Learning: Evaluation Summary\n",
        f"Test Mode Active         : {test_only}\n",
        f"Total test environments  : {len(eval_tasks)}\n",
        f"Final Table Size         : {len(agent.q_table)}\n\n",
        "--- Evaluation Results ---\n"
    ]

    success_count = 0
    for task in eval_tasks:
        label = f"{task['env_name']}_ep{task['ep_idx']}"
        env = task['env']
        ast = task['ast']
        config = task['config']

        gif_path = os.path.join(output_dir, f"hybrid_q_{label}.gif")
        
        renderer = HybridFrameRenderer(
            config['grid'], ast['grid_used'], ast['raw_path'], ast['final_path'],
            ast['open_log'], ast['closed_log'], config['start'], config['goal'], f"{label} (Tabular Q)"
        )

        state = env.reset()
        renderer.capture(env.robot_pos, env.dyn_state, env.ray_endpoints, env.robot_trail, env.local_target, False, 0, 0)
        
        steps, total_reward, done = 0, 0, False
        while not done and steps < eval_max_steps:
            is_q = _has_dynamic_threat(state)
            action = agent.choose_action(state, explore=False) if is_q else _greedy_astar_action(env)
            
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            renderer.capture(env.robot_pos, env.dyn_state, env.ray_endpoints, env.robot_trail, env.local_target, is_q, steps, total_reward)

        renderer.save_gif(gif_path, fps=gif_fps)
        success = done and tuple(env.robot_pos) == env.goal
        if success: success_count += 1
        
        line = f"{label:<25} steps={steps:>3}  reward={total_reward:>+7.1f}  success={'YES' if success else 'NO '}\n"
        print(line, end="")
        summary_lines.append(line)

    summary_lines.append(f"\nOverall Success: {success_count}/{len(eval_tasks)}\n")
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'a' if test_only else 'w') as f:
        f.writelines(summary_lines)
    print(f"\nSummary saved -> {summary_path}")

if __name__ == '__main__':
    args = sys.argv[1:]
    is_test_mode = "test" in [arg.lower() for arg in args]
    run_hybrid_qlearning(total_episodes=12000, eval_max_steps=400, gif_fps=6, test_only=is_test_mode)