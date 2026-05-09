import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from improved_astar import (improved_astar, path_length, count_inflection_nodes,
                            total_turning_angle, report_filled_cells)
                            
# YENI IMPORT: Artik ENVIRONMENTS yerine dinamik uretici fonksiyonu aliyoruz
from grids import RAW_GRIDS, get_episode_config

from visualization import (draw_static_result, static_legend_handles, HybridFrameRenderer)
from hybrid_q_astar import (QLearningAgent, RaycastAgentEnv, train_on_env, run_episode)

# ---------------------------------------------------------------------------
# Improved A* Statik Harita Testleri
# ---------------------------------------------------------------------------
def run_astar():
    output_dir = "output_improved_astar"
    os.makedirs(output_dir, exist_ok=True)

    for env_name in RAW_GRIDS.keys():
        print(f"\n====================================================")
        print(f"Running Improved A* on {env_name} map…")

        # Test icin rastgele ama TEK ve SABIT bir harita uret (is_test=True)
        env_data = get_episode_config(env_name, episode=0, is_test=True)
        grid = env_data["grid"]
        start = env_data["start"]
        goal = env_data["goal"]

        runs = 50 if "10x10" in env_name else 5
        
        t0 = time.perf_counter()
        for _ in range(runs):
            improved_astar(grid, start, goal)
        avg_ms = (time.perf_counter() - t0) / runs * 1000

        astar_result = improved_astar(grid, start, goal)
        raw_path = astar_result[0]
        final_path = astar_result[1]
        open_log = astar_result[2]
        closed_log = astar_result[3]
        grid_used = astar_result[4]

        grid_copy = [row[:] for row in grid]
        filled = report_filled_cells(grid_copy, grid_used)
        
        print("\nU-trap filled cells (original 0 -> used 1):")
        if filled:
            for x, y in filled:
                print(f"  ({x}, {y})")
        else:
            print("  None")

        print('\n' + '=' * 52)
        print(f'  Improved A* — {env_name} results')
        print('=' * 52)
        
        print(f"  Searched nodes (closed list): {len(closed_log)}")
        print(f"  Open list nodes             : {len(open_log)}")
        print(f"  Raw path nodes              : {len(raw_path)}")
        print(f"  Final path nodes            : {len(final_path)}")
        print(f"  Inflection nodes (final)    : {count_inflection_nodes(final_path)}")
        print(f"  Turning angle (°) (final)   : {total_turning_angle(final_path):.2f}")
        print(f"  Path-finding time (ms)      : {avg_ms:.3f}")
        print(f"  Raw path length             : {path_length(raw_path):.4f}")
        print(f"  Final path length           : {path_length(final_path):.4f}")
        print('=' * 52)

        fig, ax = plt.subplots(figsize=(9, 6.5))
        draw_static_result(ax, grid, grid_used, raw_path, final_path, open_log, closed_log, start, goal)
        
        stats_text = (
            f"Searched nodes : {len(closed_log)}\n"
            f"Open nodes     : {len(open_log)}\n"
            f"Path nodes     : {len(final_path)}\n"
            f"Inflections    : {count_inflection_nodes(final_path)}\n"
            f"Turning angle  : {total_turning_angle(final_path):.1f}°\n"
            f"Path length    : {path_length(final_path):.4f}\n"
            f"Time (avg)     : {avg_ms:.3f} ms"
        )
        
        ax.text(1.05, 0.5, stats_text, transform=ax.transAxes,
                fontsize=9, va='center', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc'),
                family='monospace')
        
        ax.set_title(f'Improved A* — {env_name}', fontsize=11, pad=10)
        ax.legend(handles=static_legend_handles(), loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)
        
        out = os.path.join(output_dir, f"improved_astar_{env_name}.png")
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure saved -> {out}")

# ---------------------------------------------------------------------------
# Hibrit Q-Learning + A*
# ---------------------------------------------------------------------------
def run_hybrid(total_episodes=3000, eval_max_steps=500, gif_fps=6):
    output_dir = "output_hybrid"
    os.makedirs(output_dir, exist_ok=True)

    haritalar = list(RAW_GRIDS.keys())
    n_envs = len(haritalar)
    eps_per_env = max(1, total_episodes // n_envs)

    print(f"\n=== Hybrid Q-A* training ===")
    print(f"Environments: {n_envs}")
    print(f"Total episodes (target): {total_episodes}")
    print(f"Episodes per environment: {eps_per_env}\n")

    agent = QLearningAgent()
    
    # Test (Evaluation) asamasi icin kaydedilecek ortam degiskenleri
    eval_envs = {}
    eval_astar_summaries = {}
    eval_specs = {}

    train_t0 = time.time()
    
    for env_name in haritalar:
        print(f"  Training on {env_name} ({eps_per_env} episodes)…")
        verbose_freq = max(eps_per_env // 4, 10)
        
        for ep in range(eps_per_env):
            # 1. Her bolumde (episode) o harita icin yeni rastgele konfigurasyon uret
            config = get_episode_config(env_name, episode=ep, is_test=False)
            
            # 2. A* ile rotayi yeni pozisyonlara gore bul
            astar_result = improved_astar(config['grid'], config['start'], config['goal'])
            ast_dict = {
                'raw_path': astar_result[0],
                'final_path': astar_result[1],
                'open_log': astar_result[2],
                'closed_log': astar_result[3],
                'grid_used': astar_result[4]
            }
            
            # 3. Ortami guncel harita ve rotayla baslat
            env = RaycastAgentEnv(
                grid_static=config['grid'],
                start=config['start'],
                goal=config['goal'],
                astar_data=ast_dict,
                dyn_specs=config['dynamic_obstacles']
            )
            
            # 4. Ajanı bu bölüm icin eğit
            train_on_env(agent, env, episodes=1, max_steps=300, verbose_every=0)
            
            if (ep + 1) % verbose_freq == 0:
                print(f"    ep {ep+1}/{eps_per_env}  eps={agent.epsilon:.3f}  |Q|={len(agent.q_table)}")

        # Egitim tamamlandi. Test ve GIF uretimi icin eval haritasi olustur
        eval_config = get_episode_config(env_name, episode=0, is_test=True)
        eval_astar = improved_astar(eval_config['grid'], eval_config['start'], eval_config['goal'])
        eval_ast_dict = {
            'raw_path': eval_astar[0],
            'final_path': eval_astar[1],
            'open_log': eval_astar[2],
            'closed_log': eval_astar[3],
            'grid_used': eval_astar[4]
        }
        eval_env = RaycastAgentEnv(
            grid_static=eval_config['grid'],
            start=eval_config['start'],
            goal=eval_config['goal'],
            astar_data=eval_ast_dict,
            dyn_specs=eval_config['dynamic_obstacles']
        )
        
        eval_envs[env_name] = eval_env
        eval_astar_summaries[env_name] = eval_ast_dict
        eval_specs[env_name] = eval_config

    train_dur = time.time() - train_t0
    print(f"\nTotal training time: {train_dur:.1f}s")
    print(f"Q-table size (unique states): {len(agent.q_table)}")
    
    agent.save(os.path.join(output_dir, "q_table.pkl"))

    print("\n=== Evaluation (one test episode per env, gif rendered) ===")
    
    summary_lines = [
        "Hybrid Q-Astar: Evaluation Summary\n",
        f"Total episodes (training): {eps_per_env * n_envs}\n",
        f"Episodes per env         : {eps_per_env}\n",
        f"Training time            : {train_dur:.1f}s\n",
        f"Q-table size             : {len(agent.q_table)}\n\n"
    ]

    for env_name, env in eval_envs.items():
        env_spec = eval_specs[env_name]
        ast = eval_astar_summaries[env_name]
        
        env.reset()
        
        renderer = HybridFrameRenderer(
            grid_original=env_spec['grid'],
            grid_used=ast['grid_used'],
            raw_path=ast['raw_path'],
            final_path=ast['final_path'],
            open_log=ast['open_log'],
            closed_log=ast['closed_log'],
            start=env_spec['start'],
            goal=env_spec['goal'],
            env_name=env_name,
        )

        log_path = os.path.join(output_dir, f"log_{env_name}.txt")
        res = run_episode(agent, env, max_steps=eval_max_steps, frame_renderer=renderer, episode_label=env_name, log_path=log_path)
        
        gif_path = os.path.join(output_dir, f"hybrid_{env_name}.gif")
        renderer.save_gif(gif_path, fps=gif_fps)

        success_str = 'YES' if res['success'] else 'NO '
        
        line_part1 = f"{env_name:<22}  steps={res['steps']:3d}  reward={res['reward']:+5d}  success={success_str:<3}  final_pos={res['final_pos']}"
        line_part2 = f"{'':22}  A*: closed={len(ast['closed_log'])} final_nodes={len(ast['final_path'])} final_len={path_length(ast['final_path']):.3f}"
        line_part3 = f"{'':22}  GIF -> {gif_path}\n"
        
        full_line = f"{line_part1}\n{line_part2}\n{line_part3}"
        print(full_line)
        summary_lines.append(full_line)

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.writelines(summary_lines)
        
    print(f"\nSummary saved -> {summary_path}")

if __name__ == '__main__':
   # print("Improved A* Statik Harita Testleri Basliyor...\n")
    #run_astar()
    
    print("\nHibrit (Q-Learning + A*) Egitimi Basliyor...\n")
    run_hybrid(total_episodes=3000, eval_max_steps=500, gif_fps=6)