import numpy as np
from src.simulation_params import SimulationParams, NetworkState
from src.graph_init import *

# ============================================================================
# BEHAVIORAL MODELS
# ============================================================================

def imitate_payoff_update(state: NetworkState, params: SimulationParams):
    """
    Imitate-payoff behavioral model (BM1).
    Every timestep, pick random edges and imitate if neighbor has higher payoff.
    """
    # Get all existing edges
    edges = []
    for i in range(state.N):
        for j in range(i+1, state.N):
            if state.adjacency[i, j] == 1:
                edges.append((i, j))
    
    if len(edges) == 0:
        return
    
    # Pick one random edge
    i, j = edges[np.random.randint(0, len(edges))]
    
    # Compute payoffs
    payoffs = state.compute_payoffs(params.b, params.c)
    
    # Randomly choose ego (with 50% probability)
    if np.random.random() < 0.5:
        ego, alter = i, j
    else:
        ego, alter = j, i
    
    # Ego imitates alter if alter has higher payoff
    if payoffs[alter] > payoffs[ego]:
        state.strategies[ego] = state.strategies[alter]

def conditional_cooperation_update(state: NetworkState, params: SimulationParams, 
                                   v_cd: float = 0.2, v_dc: float = 0.8):
    """
    Conditional cooperation (voter model) - BM2
    """
    # Pick a random node
    i = np.random.randint(0, state.N)
    
    # Get neighbors
    neighbors = np.where(state.adjacency[i] == 1)[0]
    if len(neighbors) == 0:
        return
    
    # Calculate fraction of cooperators in neighborhood
    coop_neighbors = np.sum(state.strategies[neighbors])
    frac_coop = coop_neighbors / len(neighbors)
    
    if state.strategies[i] == 1:  # Current cooperator
        if frac_coop < v_cd:  # Too many defectors, switch to defect
            state.strategies[i] = 0
    else:  # Current defector
        if frac_coop > v_dc:  # Enough cooperators, switch to cooperate
            state.strategies[i] = 1

def pairwise_comparison_update(state: NetworkState, params: SimulationParams, beta: float = 1.0):
    """
    Pairwise comparison rule - BM3
    """
    # Pick a random node
    i = np.random.randint(0, state.N)
    
    # Get neighbors
    neighbors = np.where(state.adjacency[i] == 1)[0]
    if len(neighbors) == 0:
        return
    
    # Compute payoffs
    payoffs = state.compute_payoffs(params.b, params.c)
    
    # Calculate probability to flip for each alternative-strategy neighbor
    flip_prob = 0.0
    count_alternative = 0
    
    for j in neighbors:
        if state.strategies[j] != state.strategies[i]:
            count_alternative += 1
            payoff_diff = payoffs[j] - payoffs[i]
            p_ij = 1.0 / (1.0 + np.exp(-beta * payoff_diff))
            flip_prob += p_ij
    
    if count_alternative > 0:
        flip_prob /= count_alternative
        
        # Flip strategy with probability flip_prob
        if np.random.random() < flip_prob:
            state.strategies[i] = 1 - state.strategies[i]  # Flip strategy