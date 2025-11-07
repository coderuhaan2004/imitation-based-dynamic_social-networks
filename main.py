import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple, List, Dict, Callable, Optional
from dataclasses import dataclass
import seaborn as sns
from matplotlib.colors import ListedColormap
from simulation_params import SimulationParams, SimulationLogger, NetworkState
from graph_init import *
from partner_update import *
from strategy_update import *

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def run_fixed_strategy_simulation(params: SimulationParams,
                                 init_func: Callable,
                                 graph_update_func: Callable,
                                 model_name: str,
                                 num_runs: int = 1,
                                 **update_kwargs) -> SimulationLogger:
    """
    Run simulation with fixed strategies (no strategy updates)
    """
    # Average over multiple runs
    all_cooperator_fractions = []
    all_avg_payoffs = []
    
    for run in range(num_runs):
        state = NetworkState(params.N, params.initial_cooperators)
        init_func(state)
        
        logger = SimulationLogger()
        
        for t in range(params.num_timesteps):
            logger.log(state, params, t)
            
            # Update network structure only (no strategy updates)
            graph_update_func(state, params, **update_kwargs)
        
        # Log final state
        logger.log(state, params, params.num_timesteps)
        
        all_cooperator_fractions.append(logger.cooperator_fractions)
        all_avg_payoffs.append(logger.avg_payoffs)
    
    # Create averaged logger
    avg_logger = SimulationLogger()
    for t in range(params.num_timesteps + 1):
        coop_fracs = [run[t] for run in all_cooperator_fractions]
        payoffs = [run[t] for run in all_avg_payoffs]
        
        avg_logger.cooperator_fractions.append(np.mean(coop_fracs))
        avg_logger.avg_payoffs.append(np.mean(payoffs))
        avg_logger.timesteps_logged.append(t)
    
    return avg_logger

def run_coevolution_simulation(params: SimulationParams,
                              init_func: Callable,
                              graph_update_func: Callable,
                              strategy_update_func: Callable,
                              model_name: str,
                              num_runs: int = 100,
                              **update_kwargs) -> Tuple[float, float]:
    """
    Run coevolutionary simulation and return final cooperator fraction and average payoff
    """
    final_coop_fracs = []
    final_payoffs = []
    
    for run in range(num_runs):
        state = NetworkState(params.N, params.initial_cooperators)
        init_func(state)
        
        for t in range(params.num_timesteps):
            # Update network with probability based on tau_g
            if np.random.random() < 1.0 / params.tau_g:
                graph_update_func(state, params, **update_kwargs)
            
            # Update strategy with probability based on tau_s
            if np.random.random() < 1.0 / params.tau_s:
                strategy_update_func(state, params, **update_kwargs)
        
        # Record final state
        final_coop_fracs.append(state.get_cooperator_fraction())
        final_payoffs.append(state.get_avg_payoff(params.b, params.c))
    
    return np.mean(final_coop_fracs), np.mean(final_payoffs)

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_network_evolution_paper_style(logger: SimulationLogger, params: SimulationParams,
                                      title: str = "Network Evolution"):
    """
    Plot adjacency matrices in the exact style of Figure 1 from the paper
    """
    # Get snapshots at key timesteps: t=0, t=T, t=2T, ..., t=5T
    snapshots = []
    for i, (t, adj) in enumerate(logger.adjacency_snapshots):
        if t % params.T == 0 or t == params.num_timesteps - 1:
            snapshots.append((t, adj, logger.strategy_snapshots[i][1]))
    
    # If we have more than 3 snapshots, take first, middle, last
    if len(snapshots) > 3:
        indices = [0, len(snapshots)//2, -1]
        snapshots = [snapshots[i] for i in indices]
    
    fig, axes = plt.subplots(1, len(snapshots), figsize=(5*len(snapshots), 5))
    if len(snapshots) == 1:
        axes = [axes]
    
    for idx, (timestep, adjacency, strategies) in enumerate(snapshots):
        # Create colored adjacency matrix matching paper style
        colored_matrix = np.zeros((params.N, params.N, 3))  # RGB matrix
        
        for i in range(params.N):
            for j in range(params.N):
                if adjacency[i, j] == 1:
                    if strategies[i] == 1 and strategies[j] == 1:
                        colored_matrix[i, j] = [0, 1, 0]  # CC - green
                    elif strategies[i] == 0 and strategies[j] == 0:
                        colored_matrix[i, j] = [1, 0, 0]  # DD - red
                    else:
                        colored_matrix[i, j] = [1, 0.5, 0]  # CD - orange
                else:
                    colored_matrix[i, j] = [1, 1, 1]  # No edge - white
        
        axes[idx].imshow(colored_matrix, interpolation='nearest')
        axes[idx].set_title(f't = {timestep}', fontsize=16)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        
        # Add border
        for spine in axes[idx].spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
    
    plt.suptitle(title, fontsize=18, y=0.95)
    plt.tight_layout()
    plt.show()

def plot_payoff_evolution_fixed_strategy(loggers: Dict[str, SimulationLogger], 
                                        params: SimulationParams):
    """
    Plot payoff evolution for fixed strategy case (Figure 2 in paper)
    """
    plt.figure(figsize=(10, 6))
    
    T = params.T
    baseline_payoff = (params.b - params.c) * (params.N - 1)  # Complete graph all cooperation
    
    for name, logger in loggers.items():
        timesteps = np.array(logger.timesteps_logged) / T
        payoffs = logger.avg_payoffs
        
        plt.plot(timesteps, payoffs, label=name, linewidth=2)
    
    plt.axhline(y=baseline_payoff, color='black', linestyle='--', 
                label='Complete cooperation baseline', alpha=0.7)
    
    plt.xlabel('Time (in units of T)', fontsize=14)
    plt.ylabel('Payoff per Capita', fontsize=14)
    plt.title('Evolution of Payoff per Capita - Fixed Strategies\n(Extreme Popularity Update)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_timescale_experiment(results: Dict, behavioral_model: str, params: SimulationParams):
    """
    Plot timescale ratio experiments (Figures 3, 4, 5 in paper)
    """
    plt.figure(figsize=(10, 6))
    
    baseline_payoff = (params.b - params.c) * (params.N - 1)  # Complete graph all cooperation
    
    for init_name, data in results.items():
        ratios = list(data.keys())
        payoffs = list(data.values())
        
        # Sort by ratio for proper plotting
        sorted_indices = np.argsort(ratios)
        ratios_sorted = [ratios[i] for i in sorted_indices]
        payoffs_sorted = [payoffs[i] for i in sorted_indices]
        
        plt.plot(ratios_sorted, payoffs_sorted, 'o-', label=init_name, linewidth=2, markersize=6)
    
    plt.axhline(y=baseline_payoff, color='black', linestyle='--', 
                label='Complete cooperation', alpha=0.5)
    
    plt.xscale('log')
    plt.xlabel('Timescale Ratio $\\tau_s / \\tau_g$', fontsize=14)
    plt.ylabel('Payoff per Capita', fontsize=14)
    plt.title(f'Timescale Experiment: {behavioral_model}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN EXPERIMENTS - REPRODUCING PAPER RESULTS
# ============================================================================

def run_fixed_strategy_experiments():
    """Run fixed strategy experiments (Section 5.1)"""
    print("Running Fixed Strategy Experiments...")
    
    params = SimulationParams()
    
    # Different initialization functions
    init_functions = {
        'rBA': initialize_barabasi_albert_random_assignment,
        'hBA': initialize_barabasi_albert_highest_degree, 
        'ER': initialize_erdos_renyi,
        'CClique': initialize_cooperator_clique,
        'Complete': initialize_complete,
        'SBM': initialize_stochastic_block
    }
    
    loggers = {}
    
    for name, init_func in init_functions.items():
        print(f"  Running {name}...")
        logger = run_fixed_strategy_simulation(
            params, init_func, extreme_popularity_update, name, num_runs=10
        )
        loggers[name] = logger
    
    # Plot results
    plot_payoff_evolution_fixed_strategy(loggers, params)
    
    # Plot network evolution for key initializations
    key_initializations = ['ER', 'CClique', 'rBA']
    for name in key_initializations:
        print(f"  Plotting network evolution for {name}...")
        # Run one detailed simulation for visualization
        params_viz = SimulationParams()
        state = NetworkState(params_viz.N, params_viz.initial_cooperators)
        init_functions[name](state)
        logger_viz = SimulationLogger()
        
        for t in range(params_viz.num_timesteps):
            logger_viz.log(state, params_viz, t)
            extreme_popularity_update(state, params_viz)
        
        logger_viz.log(state, params_viz, params_viz.num_timesteps)
        plot_network_evolution_paper_style(logger_viz, params_viz, 
                                         f"Network Evolution: {name} Initialization")
    
    return loggers

def run_timescale_experiments():
    """Run timescale ratio experiments (Section 5.2)"""
    print("Running Timescale Ratio Experiments...")
    
    base_params = SimulationParams()
    
    # Timescale ratios to test (logarithmic scale)
    tau_ratios = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
    
    # Different initialization functions
    init_functions = {
        'rBA': initialize_barabasi_albert_random_assignment,
        'hBA': initialize_barabasi_albert_highest_degree,
        'ER': initialize_erdos_renyi, 
        'CClique': initialize_cooperator_clique,
        'Complete': initialize_complete,
        'SBM': initialize_stochastic_block
    }
    
    # Behavioral models to test
    behavioral_models = {
        'imitate-payoff': imitate_payoff_update,
        'conditional-cooperation-v1': lambda s, p: conditional_cooperation_update(s, p, v_cd=0.2, v_dc=0.8),
        'conditional-cooperation-v2': lambda s, p: conditional_cooperation_update(s, p, v_cd=0.5, v_dc=0.8),
        'pairwise-comparison': pairwise_comparison_update
    }
    
    all_results = {}
    
    for bm_name, bm_func in behavioral_models.items():
        print(f"  Behavioral Model: {bm_name}")
        bm_results = {}
        
        for init_name, init_func in init_functions.items():
            print(f"    Initialization: {init_name}")
            init_results = {}
            
            for ratio in tau_ratios:
                params = SimulationParams(tau_s=ratio)  # tau_g fixed at 1.0
                coop_frac, avg_payoff = run_coevolution_simulation(
                    params, init_func, extreme_popularity_update, bm_func, 
                    f"{init_name}_{bm_name}", num_runs=50
                )
                init_results[ratio] = avg_payoff
            
            bm_results[init_name] = init_results
        
        all_results[bm_name] = bm_results
        
        # Plot results for this behavioral model
        plot_timescale_experiment(bm_results, bm_name, base_params)
    
    return all_results

def plot_all_network_evolutions():
    """Plot network evolution for all three main graph generators"""
    print("Plotting Network Evolution for All Graph Generators...")
    
    params = SimulationParams(num_timesteps=950)  # 5T timesteps
    
    graph_generators = {
        'Erdős-Rényi (ER)': initialize_erdos_renyi,
        'Barabási-Albert (rBA)': initialize_barabasi_albert_random_assignment,
        'Cooperator Clique (CClique)': initialize_cooperator_clique
    }
    
    for name, init_func in graph_generators.items():
        print(f"  Generating evolution for {name}...")
        
        # Run simulation with detailed logging
        state = NetworkState(params.N, params.initial_cooperators)
        init_func(state)
        logger = SimulationLogger()
        
        for t in range(params.num_timesteps):
            logger.log(state, params, t)
            extreme_popularity_update(state, params)
        
        logger.log(state, params, params.num_timesteps)
        
        # Plot in paper style
        plot_network_evolution_paper_style(logger, params, 
                                         f"Network Evolution: {name}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    
    print("Reproducing Paper Results...")
    print("=" * 50)
    
    # 1. Fixed strategy experiments (Section 5.1)
    print("\n1. Fixed Strategy Experiments (Section 5.1)")
    fixed_strategy_results = run_fixed_strategy_experiments()
    
    # 2. Network evolution for all graph generators
    print("\n2. Network Evolution Visualization")
    plot_all_network_evolutions()
    
    # 3. Timescale ratio experiments (Section 5.2) 
    print("\n3. Timescale Ratio Experiments (Section 5.2)")
    print("Note: This will take a while (100 runs per configuration)...")
    
    # Uncomment to run timescale experiments (computationally intensive)
    timescale_results = run_timescale_experiments()
    
    print("\nAll experiments completed!")
    print("Results should now match the paper's figures.")