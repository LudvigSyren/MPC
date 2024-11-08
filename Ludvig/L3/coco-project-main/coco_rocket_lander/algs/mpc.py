

import cvxpy as cp
import numpy as np

from coco_rocket_lander.env.system_model import SystemModel


def cp_flatten(x: cp.Variable) -> np.ndarray:
    """Flatten a cvxpy variable into a numpy array

    Args:
        x (cp.Variable): The cvxpy variable to be flattened

    Returns:
        np.ndarray: The flattened variable
    """
    return cp.reshape(x, (-1, 1), order='C')


def cp_unflatten(x: cp.Variable, last_dim: int) -> cp.Variable:
    """Reshape a cvxpy variable

    Args:
        x (cp.Variable): The cvxpy variable to be reshaped
        last_dim (int): The last dimension

    Returns:
        cp.Variable: The reshaped variable
    """
    return cp.reshape(x, (-1, last_dim), order='C')

def format_constraint(x = None, n = None, default = None):
    """Just make sure that the constraints are in the correct format"""
    if x is None:
        x = default

    if isinstance(x, np.ndarray):
        return np.nan_to_num(x)
    elif isinstance(x, list):
        return np.nan_to_num(np.array(x).reshape(1, -1))
    else:
        return np.nan_to_num(np.array([x] * n).reshape(1, -1)) 
    
class Oracle:
    def __init__(self, A: np.ndarray, B: np.ndarray, T: int = 50, solver: str = None):
        """Oracle MPC algorithm. This knows exactly the deterministic system dynamics and serves as a benchmark.

        Args:
            sys (control.ss): The discrete-time system to be used as model
            T (int, optional): The prediction horizon. Defaults to 50.
            solver (str, optional): The solver. If None, let CVXPY find the most suitable solver.
        """
        self.A = A
        self.B = B
        self.T = T

        self.n_in = B.shape[1]
        self.n_x = A.shape[0]
        self.solver = solver
    

    def _construct_problem(self, state_constraints: bool = False):
        """Construct the Oracle MPC optimization problem
        """
        self._u = cp.Variable((self.T, self.n_in))
        self._x = cp.Variable((self.T + 1, self.n_x))

        self._r = cp.Parameter((self.n_x,))
        self._x0 = cp.Parameter((self.n_x,))

        self._u_lb = cp.Parameter((1, self.n_in))
        self._u_ub = cp.Parameter((1, self.n_in))
        self._x_lb = cp.Parameter((1, self.n_x))
        self._x_ub = cp.Parameter((1, self.n_x))

        self._Q = cp.Parameter((self.n_x, self.n_x), PSD=True)        
        self._S = cp.Parameter((self.n_x, self.n_x), PSD=True)
        self._R = cp.Parameter((self.n_in, self.n_in), PSD=True)

        self._constraints = [ self._u >= self._u_lb,
                              self._u <= self._u_ub,
                              self._x[0, :] == self._x0 ]
        
        if state_constraints:
            self._constraints += [ self._x >= self._x_lb,
                                   self._x <= self._x_ub ]
        
        self._cost = 0
        for t in range(self.T):
            # Predictive model constrains the dynamics
            self._constraints.append(self._x[t+1, :] == self.A @ self._x[t, :] + self.B @ self._u[t, :])

            self._cost += cp.quad_form(cp_flatten(self._x[t, :] - self._r), self._S if t == self.T - 1 else self._Q) \
                            + cp.quad_form(self._u[t, :], self._R)
           
        # TODO: We should probably formulate this as a DPP problem to speed up the computation!
        self._problem = cp.Problem(cp.Minimize(self._cost), self._constraints)


    def mpc_setup(self,
        u_lb = -np.inf, 
        u_ub = np.inf, 
        x_lb = -np.inf,
        x_ub = np.inf,
        Q: float | np.ndarray = 1.0,
        R: float | np.ndarray = 0.1,
        S: float | np.ndarray = 1.0) -> None:
        """Setup the MPC problem!

        Args:
            u_lb (float, optional): Lower bound on the input. Defaults to -np.inf.
            u_ub (float, optional): Upper bound on the input. Defaults to np.inf.
            x_lb (float, optional): Lower bound on the output. Defaults to -np.inf.
            x_ub (float, optional): Upper bound on the output. Defaults to np.inf.
            Q (float | np.ndarray, optional): Weight on the output error. If a scalar, it will be converted to a diagonal matrix. Defaults to 1.0.
            R (float | np.ndarray, optional): Weight on the input. If a scalar, it will be converted to a diagonal matrix. Defaults to 0.1.
            S (float | np.ndarray, optional): Weight on the terminal output error. If a scalar, it will be converted to a diagonal matrix. Defaults to 1.0.
        """
        state_constraints = x_lb != -np.inf or x_ub != np.inf
        self._construct_problem(state_constraints=state_constraints)

        self._u_lb.value = format_constraint(u_lb, self.n_in, -np.inf)
        self._u_ub.value = format_constraint(u_ub, self.n_in, np.inf)
        self._x_lb.value = format_constraint(x_lb, self.n_x, -np.inf)
        self._x_ub.value = format_constraint(x_ub, self.n_x, np.inf)
        
        self._Q.value = Q * np.eye(self.n_x) if np.isscalar(Q) else Q
        self._S.value = S * np.eye(self.n_x) if np.isscalar(S) else S
        self._R.value = R * np.eye(self.n_in) if np.isscalar(R) else R


    def mpc_step(self, 
                 reference: np.ndarray,
                 x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Perform one step of the MPC

        Args:
            reference: (np.ndarray): Reference signal (shape: (n_out, ))
            x0 (np.ndarray): Current state of the system (shape: (n_x, ))

        Returns:
            np.ndarray: The generated optimal input sequence  (shape: (T, n_in))
            np.ndarray: The predicted output  (shape: (T, n_out))
        """
        self._r.value = reference.reshape((self.n_x, )) if isinstance(reference, np.ndarray) else np.array([reference] * self.n_x).reshape((self.n_x,))
        self._x0.value = x0

        self._problem.solve(solver=self.solver)
        
        return self._u.value, self._x.value