import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple, List, Dict, Callable, Optional
from dataclasses import dataclass
import seaborn as sns
from matplotlib.colors import ListedColormap
from src.simulation_params import NetworkState

def initialize_erdos_renyi(state: NetworkState, p: float = 0.2):
    """Initialize network with Erdős-Rényi model"""
    state.adjacency = np.zeros((state.N, state.N), dtype=int)
    for i in range(state.N):
        for j in range(i+1, state.N):
            if np.random.random() < p:
                state.adjacency[i, j] = 1
                state.adjacency[j, i] = 1

def initialize_barabasi_albert(state: NetworkState, m: int = 3):
    """Initialize network with Barabási-Albert preferential attachment"""
    state.adjacency = np.zeros((state.N, state.N), dtype=int)
    G = nx.barabasi_albert_graph(state.N, m)
    state.adjacency = nx.to_numpy_array(G, dtype=int)

def initialize_cooperator_clique(state: NetworkState):
    """Initialize with cooperators forming a clique, defectors isolated"""
    state.adjacency = np.zeros((state.N, state.N), dtype=int)
    coop_indices = np.where(state.strategies == 1)[0]
    
    # Connect all cooperators to each other
    for i in coop_indices:
        for j in coop_indices:
            if i != j:
                state.adjacency[i, j] = 1
                state.adjacency[j, i] = 1

def initialize_complete(state: NetworkState):
    """Initialize as complete graph"""
    state.adjacency = np.ones((state.N, state.N), dtype=int)
    np.fill_diagonal(state.adjacency, 0)

def initialize_stochastic_block(state: NetworkState):
    """Initialize with Stochastic Block Model (2 communities)"""
    state.adjacency = np.zeros((state.N, state.N), dtype=int)
    
    # Create two communities
    comm_size = state.N // 2
    community1 = list(range(comm_size))
    community2 = list(range(comm_size, state.N))
    
    # High probability within communities, low between
    p_in = 0.8
    p_out = 0.02
    
    for i in range(state.N):
        for j in range(i+1, state.N):
            if (i in community1 and j in community1) or (i in community2 and j in community2):
                if np.random.random() < p_in:
                    state.adjacency[i, j] = 1
                    state.adjacency[j, i] = 1
            else:
                if np.random.random() < p_out:
                    state.adjacency[i, j] = 1
                    state.adjacency[j, i] = 1

def initialize_barabasi_albert_random_assignment(state: NetworkState, m: int = 3):
    """BA graph with random cooperator assignment (rBA)"""
    initialize_barabasi_albert(state, m)
    # Strategies are already randomly assigned in NetworkState.__init__

def initialize_barabasi_albert_highest_degree(state: NetworkState, m: int = 3):
    """BA graph with cooperators assigned to highest degree nodes (hBA)"""
    # First create BA graph
    G = nx.barabasi_albert_graph(state.N, m)
    state.adjacency = nx.to_numpy_array(G, dtype=int)
    
    # Get node degrees and assign cooperators to highest degree nodes
    degrees = state.adjacency.sum(axis=1)
    sorted_indices = np.argsort(degrees)[::-1]  # descending order
    
    # Reset all to defectors
    state.strategies = np.zeros(state.N, dtype=int)
    
    # Assign cooperators to highest degree nodes
    num_cooperators = np.sum(state.strategies) if hasattr(state, 'initial_cooperators') else 15
    for i in range(min(num_cooperators, state.N)):
        state.strategies[sorted_indices[i]] = 1