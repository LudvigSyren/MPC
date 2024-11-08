

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
    def __init__(self, A, B, N: int = 50, solver: str = None, warm_start: bool = False, d = 0):
        """Oracle MPC algorithm. This knows exactly the deterministic system dynamics and serves as a benchmark.

        Args:
            sys (control.ss): The discrete-time system to be used as model
            T (int, optional): The prediction horizon. Defaults to 50.
            solver (str, optional): The solver. If None, let CVXPY find the most suitable solver.
        """
        self.A = A
        self.B = B
        self.d = d 
        self.N = N 

        self.n_in = B.shape[1]
        self.n_x = A.shape[0]
        self.solver = solver
        self.warm_start = warm_start
        self.ulast = np.zeros((N,self.n_in))
    
    def _construct_problem(self, state_constraints: bool = False, Q = None, R = None, S = None, LTV = False ):
        """Construct the MPC optimization problem
        """
        u = cp.Variable((self.N, self.n_in))
        x = cp.Variable((self.N + 1, self.n_x))


        r = cp.Parameter((self.n_x,))
        x0 = cp.Parameter((self.n_x,))

        u_lb = cp.Parameter((1, self.n_in))
        u_ub = cp.Parameter((1, self.n_in))
        x_lb = cp.Parameter((1, self.n_x))
        x_ub = cp.Parameter((1, self.n_x))

        if LTV:
            self._A = [cp.Parameter((self.n_x, self.n_x)) for _ in range(self.N)]
            self._B = [cp.Parameter((self.n_x, self.n_in)) for _ in range(self.N)]
            self._d = [cp.Parameter((self.n_x)) for _ in range(self.N)]
        else:
            self._A = cp.Parameter((self.n_x, self.n_x))        
            self._B = cp.Parameter((self.n_x, self.n_in))
            self._d = cp.Parameter((self.n_x))


        constraints =  [u >= u_lb,
                        u <= u_ub,
                        x[0, :] == x0 ]
        
        if state_constraints:
            constraints += [x >= x_lb,
                            x <= x_ub ]
        
        cost = 0
        if LTV:
            for k in range(self.N):
                # Dynamics 
                constraints.append(x[k+1, :] == self._A[k] @ x[k, :] + self._B[k] @ u[k, :] + self._d[k])
                # Running cost
                cost += cp.quad_form(x[k, :] - r, Q) + cp.quad_form(u[k, :], R)
        else:
            for k in range(self.N):
                # Dynamics 
                constraints.append(x[k+1, :] == self._A @ x[k, :] + self._B @ u[k, :] + self._d)
                # Running cost
                cost += cp.quad_form(x[k, :] - r, Q) + cp.quad_form(u[k, :], R)

        #  Add terminal cost
        cost += cp.quad_form(x[self.N,:]-r, S)
           
        # Form the problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        # Store variables and problem for later use
        self._problem = problem 
        self._cost = cost
        self._constraints = constraints 

        self._u = u  
        self._x = x 
        self._r = r
        self._x0 = x0 
        self._u_lb = u_lb
        self._u_ub = u_ub
        self._x_lb = x_lb
        self._x_ub = x_ub

    def mpc_setup(self,
        u_lb = -np.inf, 
        u_ub = np.inf, 
        x_lb = -np.inf,
        x_ub = np.inf,
        Q: float | np.ndarray = 1.0,
        R: float | np.ndarray = 0.1,
        S: float | np.ndarray = 1.0,
        LTV=False) -> None:
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
        state_constraints = False 
        self._construct_problem(state_constraints=state_constraints, Q=Q, R=R,S=S, LTV=LTV)

        self._u_lb.value = format_constraint(u_lb, self.n_in, -np.inf)
        self._u_ub.value = format_constraint(u_ub, self.n_in, np.inf)
        self._x_lb.value = format_constraint(x_lb, self.n_x, -np.inf)
        self._x_ub.value = format_constraint(x_ub, self.n_x, np.inf)
        if LTV: 
            for i in range(self.N):
                self._A[i].value = self.A
                self._B[i].value = self.B
                self._d[i].value = self.d
        else:
            self._A.value = self.A
            self._B.value = self.B
            self._d.value = self.d
        

    def mpc_step(self, 
                 reference: np.ndarray,
                 x0: np.ndarray,A =None, B=None, d = 0, LTV=False) -> tuple[np.ndarray, np.ndarray]:
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
        if A is not None:
            if LTV: 
                for i in range(self.N):
                    self._A[i].value = A[i]
                    self._B[i].value = B[i]
                    self._d[i].value = d[i]
            else:
                self._A.value = A
                self._B.value = B 
                self._d.value = d

        self._problem.solve(solver=self.solver, warm_start = self.warm_start)
        
        return self._u.value
