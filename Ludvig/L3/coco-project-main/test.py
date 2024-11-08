import time

import numpy as np

from coco_rocket_lander.algs.mpc import Oracle
from coco_rocket_lander.algs.pid import PID_Benchmark
from coco_rocket_lander.env.env_cfg import EnvConfig, UserArgs
from coco_rocket_lander.env.rocketlander import RocketLander
from coco_rocket_lander.env.system_model import SystemModel

CONTROLLER = 'MPC'  # 'PID' or 'MPC'
ENABLE_WIND = True

class Controller:
    def __init__(self, controller, type):
        self.controller = controller
        self.type = type

    def get_action(self, state, target):
        if self.type == 'PID':
            Fe, Fs, psi =  self.controller.pid_algorithm(state, x_target=target[0], y_target=target[1])
            return [Fe, Fs, psi]
        else:
            r_ = np.array([target[0], target[1], 0, 0, target[2], 0]) if len(target) == 3 else np.array(target)
            x_ = np.array(state[:6]) if len(state) > 6 else np.array(state)
            u, _ = self.controller.mpc_step(r_, x_)
            return u[0].flatten().tolist()

pids = PID_Benchmark()
env = RocketLander(args={'initial_position': (0.65, 1, 0.6),
                         'initial_state': (0.7, 1, 0.65, -15, 0.12, 0.05),
                         'enable_wind': ENABLE_WIND}, render_mode="human")
env.reset()


def build_controller() -> Controller:
    if CONTROLLER == 'PID':
        return Controller(PID_Benchmark(), 'PID')
    
    elif CONTROLLER == 'MPC':
        # Stuff for MPC
        K = 10
        dt = 1 / EnvConfig.fps * K      # This is not a problem, it's a chain of integrators anyways! It's like assuming to apply the same input for K steps
        model = SystemModel(env)
        model.calculate_linear_system_matrices()
        model.discretize_system_matrices(sample_time=dt)
        Ad, Bd = model.get_discrete_linear_system_matrices()

        Q = np.diag([2, 1, 5, 5, 1, 5])
        R = np.diag([1, 1, 1])
        R = 0.1 * R
        S = np.diag([5, 1, 10, 10, 1, 10])
        N = 30
        u_lb = [0, -1, -1]
        u_ub = [1, 1, 1]

        myMPC = Oracle(Ad, Bd, T=N)
        myMPC.mpc_setup(u_lb=u_lb, u_ub=u_ub, Q=Q, R=R, S=S)
        return Controller(myMPC, 'MPC')

    else:
        raise NotImplementedError(f"Controller {CONTROLLER} not implemented")


target = env.get_landing_position()
controller = build_controller()


while True:
    s = env.state 
    action = controller.get_action(s, target)
    obs, reward, done, info, _ = env.step(action)

    if done:
        print('Done!')
        print('Landed?', reward)
        # pause for a second before closing the window
        time.sleep(2) 
        env.close()
        # Pause until the window is closed
        break
