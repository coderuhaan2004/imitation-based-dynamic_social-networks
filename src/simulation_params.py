import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple, List, Dict, Callable, Optional
from dataclasses import dataclass
import seaborn as sns
from matplotlib.colors import ListedColormap

@dataclass
class SimulationParams:
    """Parameters for the simulation matching the paper"""
    N: int = 20  # Number of agents
    b: float = 100.0  # Benefit
    c: float = 50.0  # Cost
    tau_g: float = 1.0  # Graph-theoretic timescale
    tau_s: float = 1.0  # Strategic timescale
    num_timesteps: int = 950  # Total timesteps (5T where T = N(N-1)/2 = 190)
    initial_cooperators: int = 15  # Initial number of cooperators
    T: int = 190  # Number of dyads = N(N-1)/2


class NetworkState:
    """Represents the current state of the network"""
    
    def __init__(self, N: int, initial_cooperators: int):
        self.N = N
        self.adjacency = np.zeros((N, N), dtype=int)
        self.strategies = np.zeros(N, dtype=int)
        
        # Initialize cooperators randomly
        coop_indices = np.random.choice(N, initial_cooperators, replace=False)
        self.strategies[coop_indices] = 1
    
    def compute_payoffs(self, b: float, c: float) -> np.ndarray:
        """Compute payoff vector for all agents"""
        degrees = self.adjacency.sum(axis=1)
        neighbors_coop = self.adjacency @ self.strategies
        
        payoffs = np.zeros(self.N)
        for i in range(self.N):
            if self.strategies[i] == 1:  # Cooperator
                payoffs[i] = b * neighbors_coop[i] - c * degrees[i]
            else:  # Defector
                payoffs[i] = b * neighbors_coop[i]
        
        return payoffs
    
    def get_avg_payoff(self, b: float, c: float) -> float:
        """Get average payoff per capita"""
        payoffs = self.compute_payoffs(b, c)
        return payoffs.mean()
    
    def get_cooperator_fraction(self) -> float:
        """Get fraction of cooperators"""
        return self.strategies.sum() / self.N
    
    def get_num_cooperators(self) -> int:
        """Get number of cooperators"""
        return self.strategies.sum()
    
    def get_edge_types(self) -> Tuple[int, int, int]:
        """Count CC, CD, DD edges"""
        cc_edges = 0
        cd_edges = 0
        dd_edges = 0
        
        for i in range(self.N):
            for j in range(i+1, self.N):
                if self.adjacency[i, j] == 1:
                    if self.strategies[i] == 1 and self.strategies[j] == 1:
                        cc_edges += 1
                    elif self.strategies[i] == 0 and self.strategies[j] == 0:
                        dd_edges += 1
                    else:
                        cd_edges += 1
        
        return cc_edges, cd_edges, dd_edges

class SimulationLogger:
    """Logs simulation data"""

    def __init__(self, log_file: str = None):
        self.cooperator_fractions = []
        self.avg_payoffs = []
        self.edge_type_history = []
        self.adjacency_snapshots = []
        self.strategy_snapshots = []
        self.timesteps_logged = []
        self.log_file = log_file
        
        # Initialize log file
        if self.log_file:
            with open(self.log_file, 'w') as f:
                f.write("timestep,cooperator_fraction,avg_payoff,cc_edges,cd_edges,dd_edges,total_edges\n")

    def log(self, state: NetworkState, params: SimulationParams, timestep: int):
        """Log current state"""
        coop_frac = state.get_cooperator_fraction()
        avg_payoff = state.get_avg_payoff(params.b, params.c)
        cc, cd, dd = state.get_edge_types()
        
        self.cooperator_fractions.append(coop_frac)
        self.avg_payoffs.append(avg_payoff)
        self.edge_type_history.append((cc, cd, dd))
        self.timesteps_logged.append(timestep)
        
        # Write to log file
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{timestep},{coop_frac:.4f},{avg_payoff:.4f},{cc},{cd},{dd},{cc+cd+dd}\n")
        
        # Save snapshots at specific intervals (every T timesteps)
        if timestep % params.T == 0 or timestep == params.num_timesteps - 1:
            self.adjacency_snapshots.append((timestep, state.adjacency.copy()))
            self.strategy_snapshots.append((timestep, state.strategies.copy()))

    def print_summary(self):
        """Print summary statistics"""
        print(f"\nSimulation Summary:")
        print(f"  Total timesteps: {len(self.cooperator_fractions)}")
        print(f"  Initial cooperator fraction: {self.cooperator_fractions[0]:.3f}")
        print(f"  Final cooperator fraction: {self.cooperator_fractions[-1]:.3f}")
        print(f"  Initial avg payoff: {self.avg_payoffs[0]:.2f}")
        print(f"  Final avg payoff: {self.avg_payoffs[-1]:.2f}")
        
        if len(self.edge_type_history) > 0:
            cc, cd, dd = self.edge_type_history[-1]
            print(f"  Final edge counts - CC: {cc}, CD: {cd}, DD: {dd}")
        
        if self.log_file:
            print(f"  Log file saved to: {self.log_file}")   