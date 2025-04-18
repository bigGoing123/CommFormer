from gym.envs.registration import register

register(
    id='PredatorPrey-v0',
    entry_point='ic3net_envs.predator_prey_env:PredatorPreyEnv',
)

register(
    id='TrafficJunction-v0',
    entry_point='ic3net_envs.traffic_junction_env:TrafficJunctionEnv',
)

register(
    id='CooperativeNavigation-v0',
    entry_point='ic3net_envs.cooperative_navigation:CN_Env',
)