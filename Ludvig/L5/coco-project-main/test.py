import time

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from coco_rocket_lander.algs.mpc import Oracle
from coco_rocket_lander.algs.pid import PID_Benchmark
from coco_rocket_lander.env.env_cfg import EnvConfig, UserArgs
from coco_rocket_lander.env.rocketlander import RocketLander
from coco_rocket_lander.env.system_model import SystemModel

CONTROLLER = 'MPC'  # 'PID' or 'MPC'
ENABLE_WIND = False
mpc_type = 'LTV' # LTI, LPV or LTV 
K = 8 
dt = 1 / EnvConfig.fps * K # This is not a problem, it's a chain of integrators anyways! It's like assuming to apply the same input for K steps
#INITIAL_THETA = -0.2
INITIAL_THETA = -0.375

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
            if mpc_type == 'LTI':
                u = self.controller.mpc_step(r_, x_)
            elif mpc_type == 'LPV':
                model = SystemModel(env)
                u_eq = np.array([model.mass * model.gravity, 0, 0])
                model.calculate_linear_system_matrices(x_eq=x_,u_eq=u_eq)
                model.discretize_system_matrices(sample_time=dt)
                Ad, Bd = model.get_discrete_linear_system_matrices()
                #d = model.step_rocket(x_,u_eq,dt)-(Ad@x_+Bd@u_eq)
                d = np.zeros(6)
                u = self.controller.mpc_step(r_, x_, A = Ad, B = Bd, d = d)
            elif mpc_type == 'LTV':
                model = SystemModel(env)
                As,Bs,ds = [],[],[]
                x = np.copy(x_) 
                nu = self.controller.n_in
                u0 = np.array([model.mass * model.gravity, 0, 0])
                for i in range(self.controller.N):
                    uc = self.controller.ulast[i]
                    model.calculate_linear_system_matrices(x_eq=x, u_eq = u0)
                    model.discretize_system_matrices(sample_time=dt)
                    Ad, Bd = model.get_discrete_linear_system_matrices()
                    d = model.step_rocket(x,uc,dt)-(Ad@x+Bd@uc)
                    print(d)
                    #d[3]+=0.5
                    #d = np.zeros(6)
                    As.append(Ad)
                    Bs.append(Bd)
                    ds.append(d)
                    x = model.step_rocket(x,uc,dt)
                u = self.controller.mpc_step(r_, x_, A = As, B = Bs, LTV=True, d = ds)
                self.controller.ulast = np.vstack((u[1:],u[-1])) # Append last u twice 

            #print(u)
            t_solve = self.controller._problem.solver_stats.solve_time 
            return u[0].flatten().tolist(),t_solve



def build_controller(solver=cp.OSQP, warm_start=False, N = 50) -> Controller:
    if CONTROLLER == 'PID':
        return Controller(PID_Benchmark(), 'PID')
    
    elif CONTROLLER == 'MPC':
        # Stuff for MPC
        dt = 1 / EnvConfig.fps * K      # This is not a problem, it's a chain of integrators anyways! It's like assuming to apply the same input for K steps
        model = SystemModel(env)
        model.calculate_linear_system_matrices()
        model.discretize_system_matrices(sample_time=dt)
        Ad, Bd = model.get_discrete_linear_system_matrices()

        Q = np.diag([2, 1, 5, 5, 1, 5])
        R = np.diag([1, 1, 1])
        R = 0.1 * R
        S = np.diag([5, 1, 10, 10, 1, 10])
        u_lb = [0, -0.5, -0.5]
        u_ub = [1, 0.5, 0.5]

        #x_lb = np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf])
        #x_ub = np.array([np.inf,np.inf,np.inf,np.inf,np.inf])
        x_lb = -np.inf
        x_ub = np.inf

        myMPC = Oracle(Ad, Bd, N=N, solver = solver, warm_start=warm_start, d = np.zeros(6))
        myMPC.mpc_setup(u_lb=u_lb, u_ub=u_ub, x_lb=x_lb, x_ub=x_ub, Q=Q, R=R, S=S, LTV = mpc_type=='LTV')
        return Controller(myMPC, 'MPC')

    else:
        raise NotImplementedError(f"Controller {CONTROLLER} not implemented")


scenarios = [{'solver': cp.CLARABEL, 'warm_start':False, 'N':10},]

all_times = []
for sc in scenarios:
    env = RocketLander(args={#'initial_position': (0.65, 1, 0.6),
                             'initial_state': (0.5, 1, 0, -15, INITIAL_THETA, -0.1),
                             'enable_wind': ENABLE_WIND}, render_mode="human")
    env.reset()

    target = env.get_landing_position()
    controller = build_controller(solver=sc['solver'], warm_start=sc['warm_start'], N = sc['N'])
    
    
    solve_times = []
    
    print(sc)
    while True:
        s = env.state 
        action,solve_time= controller.get_action(s, target)
        solve_times.append(solve_time)
        obs, reward, done, info, _ = env.step(action)
    
        if done:
            env.close()
            break
    all_times.append(solve_times)
#for i in range(len(all_times)):
#    if 'warm_start' in scenarios[i]:
#        warm_str = ' | Warm' if scenarios[i]['warm_start'] else ' | Cold'
#    else:
#        warm_str = ''
#    plt.plot(all_times[i], label=scenarios[i]['solver']+warm_str+ ' | N='+str(scenarios[i]['N']))
#plt.ylabel('Solve time')
#plt.legend()
#plt.yscale('log')
#plt.show()
#
#plt.show()
