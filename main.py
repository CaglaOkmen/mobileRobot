"""
D3QN Hybrid A* main dosyasi.
Sonuclar 'output_hybrid_d3qn' klasorune kaydedilir.
Kullanim: 
 - Egitim ve Test icin: python main_d3qn.py
 - Sadece Test icin   : python main_d3qn.py test
"""
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')

from improved_astar import (improved_astar, path_length)
from grids import RAW_GRIDS, get_training_schedule, get_episode_config
from visualization import HybridFrameRenderer

from hybrid_d3qn_astar import (D3QNAgent, RaycastAgentEnv, train_on_env,
                               run_episode, DEVICE, set_seed)

# ---------------------------------------------------------------------------
# Hibrit D3QN + A*
# ---------------------------------------------------------------------------
def run_hybrid_d3qn(total_episodes=6000, eval_max_steps=500, gif_fps=10, test_only=False):
    set_seed(20) # Rastgeleligi sabitle
    output_dir = "output_hybrid_d3qn"
    os.makedirs(output_dir, exist_ok=True)

    n_envs = len(RAW_GRIDS)

    print(f"\n=== Hybrid D3QN-A* {'TEST' if test_only else 'TRAINING'} ===")
    print(f"Compute device         : {DEVICE}")
    print(f"Environments           : {n_envs}")

    # D3QN Ajani: Guncellenmis hiperparametreler
    agent = D3QNAgent(lr=0.0003, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.004,
                      buffer_size=25000, batch_size=64, hidden=128, target_update_freq=200,
                      tau=0.01, use_soft_update=True)

    # ---------------------------------------------------------
    # EGITIM MODU (Eger test_only True degilse calisir)
    # ---------------------------------------------------------
    if not test_only:
        print(f"Total episodes (target): {total_episodes}")
        print(f"Training schedule      : Random shuffled (katastrofik unutmayi onler)\n")
        
        train_schedule = get_training_schedule(total_episodes=total_episodes, schedule_seed=42)
        train_progress_logs = []
        train_t0 = time.time()

        print(f"  Training started with mixed schedule...")
        for i, (env_name, ep) in enumerate(train_schedule):
            config = get_episode_config(env_name, episode=ep, is_test=False)

            astar_result = improved_astar(config['grid'], config['start'], config['goal'])
            ast_dict = {
                'raw_path': astar_result[0], 'final_path': astar_result[1],
                'open_log': astar_result[2], 'closed_log': astar_result[3],
                'grid_used': astar_result[4]
            }

            env = RaycastAgentEnv(
                grid_static=config['grid'], start=config['start'], goal=config['goal'],
                astar_data=ast_dict, dyn_specs=config['dynamic_obstacles']
            )

            train_on_env(agent, env, episodes=1, max_steps=300, verbose_every=0)

            if (i + 1) % 50 == 0:
                agent.decay_epsilon()

            if (i + 1) % 500 == 0:
                log_line = (f"    Step {i+1}/{total_episodes} | Env: {env_name:<15} | "
                            f"eps={agent.epsilon:.3f} | loss={agent.last_loss:.4f} | buf={len(agent.replay)}")
                print(log_line)
                train_progress_logs.append(log_line)

        train_dur = time.time() - train_t0
        print(f"\nTotal training time: {train_dur:.1f}s")
        print(f"Total learn steps  : {agent.learn_step_counter}")

        agent.save(os.path.join(output_dir, "d3qn.pt"))

    # ---------------------------------------------------------
    # TEST MODU (Model yuklenir)
    # ---------------------------------------------------------
    else:
        model_path = os.path.join(output_dir, "d3qn.pt")
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"Basarili: Egitilmis model yuklendi -> {model_path}")
        else:
            print(f"\nHATA: Model dosyasi ({model_path}) bulunamadi!")
            print("Lutfen once test argumani olmadan calistirip ajani egitin.")
            return

    # ---> TEST (EVAL) ORTAMLARINI OLUŞTUR <---
    print("\n=== Preparing Evaluation Environments ===")
    eval_tasks = []
    
    haritalar = list(RAW_GRIDS.keys())
    for ep in range(5):
        for env_name in haritalar:
            eval_config = get_episode_config(env_name, episode=ep, is_test=True)
            eval_astar = improved_astar(eval_config['grid'], eval_config['start'], eval_config['goal'])
            
            eval_ast_dict = {
                'raw_path': eval_astar[0], 'final_path': eval_astar[1],
                'open_log': eval_astar[2], 'closed_log': eval_astar[3],
                'grid_used': eval_astar[4]
            }
            
            env = RaycastAgentEnv(
                grid_static=eval_config['grid'], start=eval_config['start'],
                goal=eval_config['goal'], astar_data=eval_ast_dict,
                dyn_specs=eval_config['dynamic_obstacles']
            )
            
            eval_tasks.append({
                'env_name': env_name, 'ep_idx': ep, 'env': env,
                'ast': eval_ast_dict, 'config': eval_config
            })

    print(f"\n=== Evaluation ({len(eval_tasks)} total test cases) ===")

    summary_lines = [
        "Hybrid D3QN-Astar: Evaluation Summary\n",
        f"Test Mode Active         : {test_only}\n",
        f"Total test environments  : {len(eval_tasks)}\n",
        f"Final epsilon            : {agent.epsilon:.3f}\n\n",
        "--- Evaluation Results ---\n"
    ]

    if not test_only:
        summary_lines.append("--- Training Progress Logs ---\n")
        for log in train_progress_logs:
            summary_lines.append(log + "\n")
        summary_lines.append("\n")

    success_count = 0
    for task in eval_tasks:
        env_name = task['env_name']
        ep_idx = task['ep_idx']
        env = task['env']
        ast = task['ast']
        config = task['config']

        label = f"{env_name}_ep{ep_idx}"
        gif_path = os.path.join(output_dir, f"hybrid_d3qn_{label}.gif")
        log_path = os.path.join(output_dir, f"log_{label}.txt")
        
        renderer = HybridFrameRenderer(
            grid_original=config['grid'], grid_used=ast['grid_used'],
            raw_path=ast['raw_path'], final_path=ast['final_path'],
            open_log=ast['open_log'], closed_log=ast['closed_log'],
            start=config['start'], goal=config['goal'],
            env_name=f"{label} (D3QN)"
        )
        
        res = run_episode(agent, env, max_steps=eval_max_steps,
                          frame_renderer=renderer, episode_label=label, log_path=log_path)
        
        renderer.save_gif(gif_path, fps=gif_fps)
        
        if res['success']: success_count += 1
        status = "YES" if res['success'] else "NO "
        
        line = (f"{label:<25} steps={res['steps']:>3}  reward={res['reward']:>+7.1f}  success={status}\n")
        print(line, end="")
        summary_lines.append(line)

    summary_lines.append(f"\nOverall Success: {success_count}/{len(eval_tasks)}\n")

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'a' if test_only else 'w') as f:
        if test_only:
            f.write("\n\n" + "="*50 + "\n") # Test calistirildiginda dosyanin altina ekler
        f.writelines(summary_lines)

    print(f"\nSummary saved -> {summary_path}")

if __name__ == '__main__':
    # Terminalden girilen argumanlari kontrol et
    args = sys.argv[1:]
    is_test_mode = "test" in [arg.lower() for arg in args]

    if is_test_mode:
        print("\nHibrit (D3QN + A*) SADECE TEST Modu Basliyor...\n")
        run_hybrid_d3qn(total_episodes=12000, eval_max_steps=400, gif_fps=6, test_only=True)
    else:
        print("\nHibrit (D3QN + A*) Egitimi Basliyor...\n")
        run_hybrid_d3qn(total_episodes=12000, eval_max_steps=400, gif_fps=6, test_only=False)