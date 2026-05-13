"""
Hybrid D3QN-A* Algoritmasi:

1. Hibrit Mimari: D3QN (Dueling Double DQN) + A* (Hizli Yol Planlama)
2. Dinamik Engeller: Simule Edilmis Hareketli Engeller
3. Ray-Casting: Cevre Algilama (12 ray, 30 derecede bir)
4. D3QN Bilesenleri:
   - Dueling Network: V(s) + A(s,a) ayri kollar
   - Double DQN: Online net aksiyon secer, Target net deger bicer
   - Experience Replay: Deneyim havuzundan rastgele ornekleme
   - Target Network: Periyodik soft/hard guncelleme
"""
import math
import os
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent_env import ACTIONS, N_ACTIONS, STATE_DIM, encode_state

# Cihaz secimi (GPU varsa kullan)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=20):
    """Egitimi tekrarlanabilir kilmak icin tum seed'leri sabitle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# ---------------------------------------------------------------------------
# D3QN Sinir Ag Mimarisi (Dueling)
# ---------------------------------------------------------------------------
class DuelingQNetwork(nn.Module):
    """ Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)) """
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=128):
        super().__init__()

        # Ortak Govde (Feature Extractor)
        self.feature = nn.Sequential( nn.Linear(state_dim, hidden),
                                        nn.ReLU(),
                                        nn.Linear(hidden, hidden),
                                        nn.ReLU())

        # Value Stream: Durumun ne kadar iyi oldugu (skaler)
        self.value_stream = nn.Sequential( nn.Linear(hidden, hidden // 2),
                                           nn.ReLU(),
                                           nn.Linear(hidden // 2, 1))

        # Advantage Stream: Her aksiyonun durum ustune sagladigi avantaj
        self.advantage_stream = nn.Sequential( nn.Linear(hidden, hidden // 2),
                                             nn.ReLU(),
                                             nn.Linear(hidden // 2, n_actions))

    # Sinir aginin ileri yayilimi (forward pass)
    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features) # (B, 1)
        advantage = self.advantage_stream(features) # (B, n_actions)

        # Dueling birlestirme: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# ---------------------------------------------------------------------------
# Experience Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity) # eski deneyimleri siler

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done)) # yeni deneyimleri ekler

    def sample(self, batch_size): # rastgele mini batch ornekler
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # numpy array'leri torch tensor'a cevir
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ---------------------------------------------------------------------------
# D3QN Ajan Sinifi
# ---------------------------------------------------------------------------
class D3QNAgent:
    """
    Dueling Double DQN Ajani:
      - Online network: aksiyon secimi + ogrenme
      - Target network: bootstrap hedeflerinin daha kararli olmasi icin
      - Double DQN: argmax aksiyon online'dan, deger target'tan
    """
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS,hidden=128, lr=5e-4, gamma=0.95, 
                 epsilon=0.99, epsilon_min=0.05, epsilon_decay=0.00035, buffer_size=50000, 
                 batch_size=64, target_update_freq=200, tau=0.01, use_soft_update=True):

        self.state_dim = state_dim
        self.n_actions = n_actions

        # Online ve target aglar
        self.online_net = DuelingQNetwork(state_dim, n_actions, hidden).to(DEVICE)
        self.target_net = DuelingQNetwork(state_dim, n_actions, hidden).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer (Adam)
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # parameters
        self.gamma = gamma # Indirgeme katsayisi
        self.epsilon = epsilon # Epsilon-greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.replay = ReplayBuffer(capacity=buffer_size) # Deneyim havuzu
        self.batch_size = batch_size # Batch boyutu

        self.target_update_freq = target_update_freq # Hedef ag guncelleme sikligi
        self.tau = tau # Soft update katsayisi
        self.use_soft_update = use_soft_update # Yumusak guncelleme

        self.learn_step_counter = 0
        self.last_loss = 0.0

    # Epsilon-greedy aksiyon secimi
    def choose_action(self, state_tuple, explore=True):
        if explore and random.random() < self.epsilon: # Rastgele aksiyon secimi
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad(): # Maksimum Q degeri
            state_vec = encode_state(state_tuple)
            state_t = torch.from_numpy(state_vec).unsqueeze(0).to(DEVICE)
            q_vals = self.online_net(state_t)
            return int(torch.argmax(q_vals, dim=1).item())

    # Deneyimi replay buffer'a ekleme
    def store(self, state_tuple, action, reward, next_state_tuple, done):
        s = encode_state(state_tuple)
        s_next = encode_state(next_state_tuple)
        self.replay.push(s, action, reward, s_next, float(done))

    def learn(self):
        """
        Double DQN:
          a* = argmax_a Q_online(s', a)
          y   = r + gamma * Q_target(s', a*) * (1-done)
          loss = MSE(Q_online(s, a), y)
        """
        if len(self.replay) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        # Mevcut Q degerleri
        q_pred = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: aksiyon secimi online net, deger target net
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        # Huber loss daha kararli (outlier'lara dayanikli)
        loss = F.smooth_l1_loss(q_pred, q_target)
        
        self.optimizer.zero_grad() # Gradientleri sifirla
        loss.backward() # Geri yayilim
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step() # Agi guncelle

        self.last_loss = float(loss.item()) # kayip degerini kaydet
        self.learn_step_counter += 1

        # Target network guncelleme
        if self.use_soft_update:
            self._soft_update_target()
        else:
            if self.learn_step_counter % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
    
    # Polyak averaging: target = tau * online + (1-tau) * target
    def _soft_update_target(self):
        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    # epsilon azalimi
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    # Modeli kaydet
    def save(self, path="d3qn.pt"):
        torch.save({
            'online_state_dict': self.online_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step_counter': self.learn_step_counter,
        }, path)

    # Model yukleme
    def load(self, path="d3qn.pt"):
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=DEVICE)
        self.online_net.load_state_dict(ckpt['online_state_dict'])
        self.target_net.load_state_dict(ckpt['target_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.epsilon = ckpt.get('epsilon', self.epsilon)
        self.learn_step_counter = ckpt.get('learn_step_counter', 0)
        return True

# ---------------------------------------------------------------------------
# A* Yardimcilari 
# ---------------------------------------------------------------------------
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

# Dinamik tehdit kontrolu A* ile D3QN ayrimi icin
def _has_dynamic_threat(state):
    dynamic_rays = state[12:24]
    return any(r < 2 for r in dynamic_rays)

# ---------------------------------------------------------------------------
# D3QN Egitimi
# ---------------------------------------------------------------------------
def train_on_env(agent, env, episodes, max_steps=300, verbose_every=500,
                 learn_every=4, warmup_steps=256):
    for ep in range(episodes):
        state = env.reset()
        steps = 0
        total_reward = 0
        done = False

        while not done and steps < max_steps:
            # Tehdit varsa D3QN, yoksa A*
            if _has_dynamic_threat(state):
                action = agent.choose_action(state, explore=True)
                used_q = True
            else:
                action = _greedy_astar_action(env)
                used_q = False

            next_state, reward, done = env.step(action)

            # Sadece D3QN'in aktif oldugu adimlari buffer'a kaydet
            if used_q:
                agent.store(state, action, reward, next_state, done)

            # Yeterince buyuk buffer'a ulastiktan sonra ogren
            if len(agent.replay) >= warmup_steps and (steps % learn_every == 0):
                agent.learn()

            state = next_state
            total_reward += reward
            steps += 1

        if verbose_every and (ep + 1) % verbose_every == 0:
            print(f"    ep {ep+1}/{episodes}  steps={steps}  reward={total_reward:+.3f}  "
                  f"eps={agent.epsilon:.3f}  loss={agent.last_loss:.4f}  "
                  f"buf={len(agent.replay)}")

# ---------------------------------------------------------------------------
# Test 
# ---------------------------------------------------------------------------
def run_episode(agent, env, max_steps=400, frame_renderer=None,
                episode_label="", log_path=None):
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
        log_file.write(f"=== Episode: {episode_label} (D3QN) ===\n")
        log_file.write(f"Start: {env.start} | Goal: {env.goal}\n")
        log_file.write(f"{'Step':<5} | {'Robot':<10} | {'Action':<10} | {'Mode':<7} | "
                       f"{'Reward':<7} | {'Total':<7} | {'Dyn Obs Positions'}\n")
        log_file.write("-" * 95 + "\n")

    while not done and steps < max_steps:
        is_q = _has_dynamic_threat(state)
        if is_q:
            action = agent.choose_action(state, explore=False)
        else:
            action = _greedy_astar_action(env)

        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1

        if log_file:
            obs_str = ", ".join([str(tuple(o['pos'])) for o in env.dyn_state])
            mode_str = "D3QN" if is_q else "A*"
            action_str = str(ACTIONS[action])
            log_file.write(f"{steps:<5} | {str(tuple(env.robot_pos)):<10} | {action_str:<10} | "
                           f"{mode_str:<7} | {reward:<7.2f} | {total_reward:<7.2f} | {obs_str}\n")

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
        log_file.write("-" * 95 + "\n")
        log_file.write(f"Final Status: {'SUCCESS' if success else 'FAILURE'}\n")
        log_file.close()

    return {
        'steps': steps,
        'reward': total_reward,
        'success': success,
        'final_pos': tuple(env.robot_pos)
    }