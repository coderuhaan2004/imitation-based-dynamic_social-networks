import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import networkx as nx
from typing import Tuple, List, Dict, Callable, Optional
from dataclasses import dataclass

# Import your existing modules
import sys
sys.path.append('src')
from simulation_params import SimulationParams, SimulationLogger, NetworkState
from graph_init import *
from partner_update import *
from strategy_update import *

def create_network_animation(params: SimulationParams,
                            init_func: Callable,
                            graph_update_func: Callable,
                            model_name: str,
                            output_file: str = "network_evolution.mp4",
                            fps: int = 30,
                            log_every: int = 1,
                            **update_kwargs):
    """
    Create animated visualization of network evolution.
    
    Parameters:
    - params: Simulation parameters
    - init_func: Network initialization function
    - graph_update_func: Graph update function
    - model_name: Name for the animation
    - output_file: Output video filename
    - fps: Frames per second
    - log_every: Log network state every N timesteps (for performance)
    - update_kwargs: Additional arguments for update function
    """
    print(f"Creating animation: {model_name}")
    print(f"Total timesteps: {params.num_timesteps}")
    
    # Initialize network
    state = NetworkState(params.N, params.initial_cooperators)
    init_func(state)
    
    # Store all network states
    adjacency_history = []
    strategy_history = []
    edge_count_history = []
    
    # Run simulation and store states
    print("Running simulation...")
    for t in range(params.num_timesteps + 1):
        if t % log_every == 0 or t == params.num_timesteps:
            adjacency_history.append(state.adjacency.copy())
            strategy_history.append(state.strategies.copy())
            cc, cd, dd = state.get_edge_types()
            edge_count_history.append((cc, cd, dd))
        
        if t < params.num_timesteps:
            graph_update_func(state, params, **update_kwargs)
        
        if t % 100 == 0:
            print(f"  Progress: {t}/{params.num_timesteps}")
    
    print(f"Simulation complete. Creating animation with {len(adjacency_history)} frames...")
    
    # Create figure and axis
    fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=(12, 5),
                                             gridspec_kw={'width_ratios': [4, 1]})
    
    # Initial plot
    colored_matrix = np.zeros((params.N, params.N, 3))
    im = ax_main.imshow(colored_matrix, interpolation='nearest')
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    
    # Add border
    for spine in ax_main.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    
    # Title text
    title_text = ax_main.text(0.5, 1.05, '', transform=ax_main.transAxes,
                              ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Legend
    ax_legend.axis('off')
    legend_y = 0.9
    legend_spacing = 0.15
    
    ax_legend.add_patch(plt.Rectangle((0.1, legend_y), 0.2, 0.1, 
                                     facecolor='green', edgecolor='black'))
    ax_legend.text(0.35, legend_y + 0.05, 'CC Edge\n(Cooperator-Cooperator)', 
                  va='center', fontsize=10)
    
    ax_legend.add_patch(plt.Rectangle((0.1, legend_y - legend_spacing), 0.2, 0.1,
                                     facecolor='orange', edgecolor='black'))
    ax_legend.text(0.35, legend_y - legend_spacing + 0.05, 'CD Edge\n(Cooperator-Defector)',
                  va='center', fontsize=10)
    
    ax_legend.add_patch(plt.Rectangle((0.1, legend_y - 2*legend_spacing), 0.2, 0.1,
                                     facecolor='red', edgecolor='black'))
    ax_legend.text(0.35, legend_y - 2*legend_spacing + 0.05, 'DD Edge\n(Defector-Defector)',
                  va='center', fontsize=10)
    
    ax_legend.add_patch(plt.Rectangle((0.1, legend_y - 3*legend_spacing), 0.2, 0.1,
                                     facecolor='white', edgecolor='black'))
    ax_legend.text(0.35, legend_y - 3*legend_spacing + 0.05, 'No Edge',
                  va='center', fontsize=10)
    
    # Edge count text
    edge_text = ax_legend.text(0.1, legend_y - 4.5*legend_spacing, '',
                              fontsize=10, va='top')
    
    def update_frame(frame):
        """Update function for animation"""
        adjacency = adjacency_history[frame]
        strategies = strategy_history[frame]
        cc, cd, dd = edge_count_history[frame]
        
        # Create colored adjacency matrix
        colored_matrix = np.zeros((params.N, params.N, 3))
        
        for i in range(params.N):
            for j in range(params.N):
                if adjacency[i, j] == 1:
                    if strategies[i] == 1 and strategies[j] == 1:
                        colored_matrix[i, j] = [0, 1, 0]  # CC - green
                    elif strategies[i] == 0 and strategies[j] == 0:
                        colored_matrix[i, j] = [1, 0, 0]  # DD - red
                    else:
                        colored_matrix[i, j] = [1, 0.647, 0]  # CD - orange
                else:
                    colored_matrix[i, j] = [1, 1, 1]  # No edge - white
        
        im.set_array(colored_matrix)
        
        # Update title with timestep
        actual_timestep = frame * log_every
        title_text.set_text(f'{model_name}\nt = {actual_timestep} / {params.num_timesteps}')
        
        # Update edge counts
        edge_text.set_text(f'Edge Counts:\nCC: {cc}\nCD: {cd}\nDD: {dd}\nTotal: {cc+cd+dd}')
        
        return [im, title_text, edge_text]
    
    # Create animation
    print("Generating animation...")
    anim = animation.FuncAnimation(
        fig, 
        update_frame,
        frames=len(adjacency_history),
        interval=1000//fps,  # milliseconds between frames
        blit=True,
        repeat=True
    )
    
    # Save animation
    print(f"Saving animation to {output_file}...")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=1800)
    anim.save(output_file, writer=writer)
    
    print(f"Animation saved successfully!")
    plt.close()
    
    return anim

def create_side_by_side_animation(params: SimulationParams,
                                  init_funcs: Dict[str, Callable],
                                  graph_update_func: Callable,
                                  output_file: str = "network_comparison.mp4",
                                  fps: int = 30,
                                  log_every: int = 1):
    """
    Create side-by-side comparison animation of different initializations.
    
    Parameters:
    - params: Simulation parameters
    - init_funcs: Dictionary of {name: initialization_function}
    - graph_update_func: Graph update function
    - output_file: Output video filename
    - fps: Frames per second
    - log_every: Log every N timesteps
    """
    print("Creating side-by-side comparison animation")
    
    # Run simulations for each initialization
    all_histories = {}
    
    for name, init_func in init_funcs.items():
        print(f"Running simulation for {name}...")
        state = NetworkState(params.N, params.initial_cooperators)
        init_func(state)
        
        adjacency_history = []
        strategy_history = []
        edge_count_history = []
        
        for t in range(params.num_timesteps + 1):
            if t % log_every == 0 or t == params.num_timesteps:
                adjacency_history.append(state.adjacency.copy())
                strategy_history.append(state.strategies.copy())
                cc, cd, dd = state.get_edge_types()
                edge_count_history.append((cc, cd, dd))
            
            if t < params.num_timesteps:
                graph_update_func(state, params)
        
        all_histories[name] = {
            'adjacency': adjacency_history,
            'strategies': strategy_history,
            'edge_counts': edge_count_history
        }
    
    # Create figure with subplots
    num_plots = len(init_funcs)
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))
    if num_plots == 1:
        axes = [axes]
    
    # Initialize plots
    ims = []
    title_texts = []
    
    for idx, (name, ax) in enumerate(zip(init_funcs.keys(), axes)):
        colored_matrix = np.zeros((params.N, params.N, 3))
        im = ax.imshow(colored_matrix, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        
        title_text = ax.text(0.5, 1.05, '', transform=ax.transAxes,
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ims.append(im)
        title_texts.append(title_text)
    
    def update_frame(frame):
        """Update all subplots"""
        artists = []
        
        for idx, (name, im, title_text) in enumerate(zip(init_funcs.keys(), ims, title_texts)):
            history = all_histories[name]
            adjacency = history['adjacency'][frame]
            strategies = history['strategies'][frame]
            cc, cd, dd = history['edge_counts'][frame]
            
            # Create colored matrix
            colored_matrix = np.zeros((params.N, params.N, 3))
            
            for i in range(params.N):
                for j in range(params.N):
                    if adjacency[i, j] == 1:
                        if strategies[i] == 1 and strategies[j] == 1:
                            colored_matrix[i, j] = [0, 1, 0]  # CC - green
                        elif strategies[i] == 0 and strategies[j] == 0:
                            colored_matrix[i, j] = [1, 0, 0]  # DD - red
                        else:
                            colored_matrix[i, j] = [1, 0.647, 0]  # CD - orange
                    else:
                        colored_matrix[i, j] = [1, 1, 1]  # No edge - white
            
            im.set_array(colored_matrix)
            
            actual_timestep = frame * log_every
            title_text.set_text(f'{name}\nt = {actual_timestep}\nCC:{cc} CD:{cd} DD:{dd}')
            
            artists.extend([im, title_text])
        
        return artists
    
    # Create animation
    print("Generating comparison animation...")
    num_frames = len(all_histories[list(init_funcs.keys())[0]]['adjacency'])
    
    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=num_frames,
        interval=1000//fps,
        blit=True,
        repeat=True
    )
    
    # Save animation
    print(f"Saving animation to {output_file}...")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=2400)
    anim.save(output_file, writer=writer)
    
    print("Animation saved successfully!")
    plt.close()
    
    return anim

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    print("="*70)
    print("ANIMATED NETWORK EVOLUTION")
    print("="*70)
    
    # Create parameters
    params = SimulationParams(
        N=20,
        b=100.0,
        c=50.0,
        tau_g=1.0,
        tau_s=1.0,
        num_timesteps=950,  # 5T timesteps
        initial_cooperators=15
    )
    
    # 1. Single animation - Erdős-Rényi initialization
    print("\n1. Creating animation: Erdős-Rényi Initialization")
    create_network_animation(
        params,
        initialize_erdos_renyi,
        extreme_popularity_update,
        "Extreme Popularity: Erdős-Rényi Init",
        output_file="er_network_evolution.mp4",
        fps=30,
        log_every=5  # Log every 5 timesteps for smoother animation
    )
    
    # 2. Single animation - Cooperator Clique initialization
    print("\n2. Creating animation: Cooperator Clique Initialization")
    create_network_animation(
        params,
        initialize_cooperator_clique,
        extreme_popularity_update,
        "Extreme Popularity: Cooperator Clique Init",
        output_file="cclique_network_evolution.mp4",
        fps=30,
        log_every=5
    )
    
    # 3. Single animation - Barabási-Albert initialization
    print("\n3. Creating animation: Barabási-Albert Initialization")
    create_network_animation(
        params,
        initialize_barabasi_albert_random_assignment,
        extreme_popularity_update,
        "Extreme Popularity: Barabási-Albert Init",
        output_file="ba_network_evolution.mp4",
        fps=30,
        log_every=5
    )
    
    # 4. Side-by-side comparison animation
    print("\n4. Creating side-by-side comparison animation")
    comparison_inits = {
        'Erdős-Rényi': initialize_erdos_renyi,
        'Cooperator Clique': initialize_cooperator_clique,
        'Barabási-Albert': initialize_barabasi_albert_random_assignment
    }
    
    create_side_by_side_animation(
        params,
        comparison_inits,
        extreme_popularity_update,
        output_file="network_comparison.mp4",
        fps=30,
        log_every=5
    )
    
    print("\n" + "="*70)
    print("All animations created successfully!")
    print("="*70)
    print("\nOutput files:")
    print("  - er_network_evolution.mp4")
    print("  - cclique_network_evolution.mp4")
    print("  - ba_network_evolution.mp4")
    print("  - network_comparison.mp4")