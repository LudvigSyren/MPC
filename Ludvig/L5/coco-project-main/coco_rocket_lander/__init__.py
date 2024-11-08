from .env.rocketlander import RocketLander
from gymnasium.envs.registration import register

register(
    id='coco_rocket_lander/RocketLander-v0',
    entry_point=RocketLander,
    max_episode_steps=None,
)

from .env.env_cfg import EnvConfig
