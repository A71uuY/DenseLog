from gym.envs.registration import register

register(id="Farm-ori-v0", entry_point="gym_farm.envs:FarmEnvOri")
register(id="Farm-fair-v0", entry_point="gym_farm.envs:FarmEnvFair")

