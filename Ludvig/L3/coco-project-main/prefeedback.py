import numpy as np

from coco_rocket_lander.env.rocketlander import RocketLander
from coco_rocket_lander.env.system_model import SystemModel

env = RocketLander(args={'initial_position': (0.65, 1, 0.6),
                         'initial_state': (0.7, 1, 0.65, -15, 0.12, 0.05),
                         'enable_wind': False}, render_mode="human")

dt = 0.1 
model = SystemModel(env)

model.calculate_linear_system_matrices()
model.discretize_system_matrices(sample_time=dt)
Ad, Bd = model.get_discrete_linear_system_matrices()

def dense_matrices(N,A,B):
    # Setup Phi
    Np,Nc = N,N 
    nx,nu = np.shape(B)
    Gamma = np.zeros(((Np+1)*nx,Nc*nu));
    Phi = np.zeros(((Np+1)*nx,nx));
    Phi[:nx,:] = np.identity(nx);

    Btot = B; 
    Atot = A;

    print(np.shape(Gamma))
    for i in range(1,Nc+1): 
        for j in range(Nc-i+1):
            Gamma[((i+j)*nx):(i+j+1)*nx,j*nu:(j+1)*nu] = Btot; 
        Phi[i*nx:(i+1)*nx,:] = Atot;
        Atot@=A;
        Btot=A @ Btot;
    # Set ui = u_Nc for i>Nc 
    for i in range(Nc,Np):
        Gamma[(nx*i):nx*(i+1),:] = A @ Gamma[(nx*(i-1)):nx*i,:];
        Gamma[(nx*i):nx*(i+1),-nu:] +=B;

        Phi[i*nx:(i+1)*nx,:] = A @ Phi[nx*(i-1):nx*i,:]

    return Phi, Gamma

N = 15
F,G = dense_matrices(N,Ad,Bd) 

print(F)
print(G)

Q = np.diag([2, 1, 5, 5, 1, 5])
R = np.diag([1, 1, 1])
R = 0.1 * R
