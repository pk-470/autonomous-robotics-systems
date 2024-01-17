from constants import *
from functions import *
from swarm_opt import *
from grad_descent import *

MAX_EPOCHS = 1000

# Swarm optimization for Rosenbrock
swarm_opt_rosenbrock = Swarm_Opt(
    opt_function=rosenbrock, particles=20, a_init=1, b_init=0.5, c_init=0.5
)
swarm_opt_rosenbrock.optimize(max_epochs=MAX_EPOCHS, animate=True)

# Swarm optimization for Rastrigin
swarm_opt_rastrigin = Swarm_Opt(
    opt_function=rastrigin, particles=20, a_init=1, b_init=0.5, c_init=0.5
)
swarm_opt_rastrigin.optimize(max_epochs=MAX_EPOCHS, animate=True)

# Gradiant descent for Rosenbrock
grad_descent_rosenbrock = Grad_Descent(
    opt_function=rosenbrock, opt_function_grad=rosenbrock_grad
)
grad_descent_rosenbrock.optimize(max_epochs=MAX_EPOCHS, animate=True)

# Gradiant descent for Rastrigin
grad_descent_rastrigin = Grad_Descent(
    opt_function=rastrigin, opt_function_grad=rastrigin_grad
)
grad_descent_rastrigin.optimize(max_epochs=MAX_EPOCHS, animate=True)
