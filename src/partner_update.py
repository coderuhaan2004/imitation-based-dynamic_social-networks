import numpy as np
from src.simulation_params import SimulationParams, NetworkState
from src.graph_init import *

# ============================================================================
# GRAPH-THEORETIC MODELS OF PARTNER UPDATE RULE
# ============================================================================

def extreme_popularity_update(state: NetworkState, params: SimulationParams):
    """
    Extreme popularity model: cooperators are always befriended,
    defectors are always unfriended.
    """
    # Choose a random pair
    i = np.random.randint(0, state.N)
    j = np.random.randint(0, state.N)
    
    if i == j:
        return
    
    # Determine who is ego (with 50% probability)
    if np.random.random() < 0.5:
        ego, alter = i, j
    else:
        ego, alter = j, i
    
    if state.strategies[alter] == 1:  # Alter is cooperator
        # Form edge if it doesn't exist
        state.adjacency[ego, alter] = 1
        state.adjacency[alter, ego] = 1
    else:  # Alter is defector
        # Break edge if it exists
        state.adjacency[ego, alter] = 0
        state.adjacency[alter, ego] = 0

def active_linking_update(state: NetworkState, params: SimulationParams,
                          alpha_C: float = 0.5, alpha_D: float = 0.3,
                          beta_CC: float = 0.0, beta_CD: float = 0.3,
                          beta_DD: float = 0.6):
    """
    Active linking model from Pacheco et al. 2006
    """
    # Choose a random pair
    i = np.random.randint(0, state.N)
    j = np.random.randint(0, state.N)
    
    if i == j:
        return
    
    s_i, s_j = state.strategies[i], state.strategies[j]
    edge_exists = state.adjacency[i, j]
    
    # Calculate formation and breaking rates
    if s_i == 1 and s_j == 1:  # Both cooperators
        formation_rate = alpha_C**2 / params.tau_g
        breaking_rate = beta_CC / params.tau_g
    elif s_i == 0 and s_j == 0:  # Both defectors
        formation_rate = alpha_D**2 / params.tau_g
        breaking_rate = beta_DD / params.tau_g
    else:  # Mixed
        formation_rate = alpha_C * alpha_D / params.tau_g
        breaking_rate = beta_CD / params.tau_g
    
    if edge_exists == 0:
        # Attempt to form edge
        if np.random.random() < formation_rate:
            state.adjacency[i, j] = 1
            state.adjacency[j, i] = 1
    else:
        # Attempt to break edge
        if np.random.random() < breaking_rate:
            state.adjacency[i, j] = 0
            state.adjacency[j, i] = 0