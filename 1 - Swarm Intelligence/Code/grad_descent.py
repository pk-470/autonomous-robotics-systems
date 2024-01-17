import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from tqdm import tqdm

from constants import *
from functions import *


class Grad_Descent:
    """
    A class that performs gradient descent.
    """

    def __init__(self, opt_function, opt_function_grad, learning_rate=0.0001):
        # Function to be optimized
        self.opt_function = opt_function

        # Grad of function to be optimized
        self.opt_function_grad = opt_function_grad

        # Randomly initialize particle position
        self.position = np.random.uniform(GRID_MIN, GRID_MAX, (2,))

        # Optimization parameters
        self.learning_rate = learning_rate

        # Optimal value history
        self.opt_value_history = [self.opt_function(*self.position)]

    def _opt_step(self):
        """
        Updates the position of the descent in the direction opposite
        to the gradient.
        """
        # Update the position
        self.position -= self.learning_rate * self.opt_function_grad(*self.position)

        # Update the value history
        self.opt_value_history.append(self.opt_function(*self.position))

    def _terminating_condition(self):
        """
        Terminate whenever the norm of the gradient gets smaller than
        the constant GRAD_EPSILON.
        """
        return np.linalg.norm(self.opt_function_grad(*self.position)) < GRAD_EPSILON

    def optimize(self, max_epochs, animate=False):
        """
        Optimization loop.
        """
        print()
        print(
            "-------------------------------- Optimizing "
            f"{self.opt_function.__name__.capitalize()} function "
            "using gradient descent "
            f"over {max_epochs} epochs --------------------------------"
        )

        progress_bar = tqdm(range(max_epochs), leave=False)

        if animate:
            fig = plt.figure()

            def epoch_gen():
                """
                Generator for epochs. Terminates when the maximum number of epochs
                is reached, or as defined by _terminating_condition.
                """
                epoch = 0
                while epoch < max_epochs and not self._terminating_condition():
                    yield epoch
                    epoch += 1

            def anim_step(epoch):
                """
                Performs optimization and produces the plots
                for the animation at each epoch.
                """
                # Optimization step
                self._opt_step()

                # Plot
                plt.title(
                    f"{self.opt_function.__name__.capitalize()} function (epoch {epoch})"
                )

                if epoch > 0:
                    self._particle.remove()
                    self._grad_dir.remove()
                self._particle_plot()

                progress_bar.update()

            # Create and save animation (optimization happens
            # within the animation loop)
            animation = FuncAnimation(
                fig,
                anim_step,
                frames=epoch_gen(),
                init_func=self._contour_plot,
                save_count=max_epochs,
                repeat=False,
            )
            path_run = self._get_path_run("./animations/grad_descent")
            animation.save(f"{path_run}_{self.opt_function.__name__}.gif", fps=20)

            plt.clf()

        else:
            epoch = 0
            while epoch < max_epochs and not self._terminating_condition():
                # Optimization step
                self._opt_step(epoch)

                progress_bar.update()
                epoch += 1

        progress_bar.close()

        # Plot optimal value history
        self.plot_opt_value_history()

        # Print results
        x_opt, y_opt = self.position
        print(f"Optimal point: [{x_opt:.4g}, {y_opt:.4g}]")
        print(f"Optimal value: {self.opt_function(x_opt, y_opt):.4g}")
        print()

    # Functions for plotting and saving plots

    def _get_path_run(self, path):
        """
        Gets the correct run label for the data produced.
        """
        run = 0
        for name in os.listdir(path):
            prev_run = int(name.split("_")[0])
            if prev_run > run:
                run = prev_run
        run += 1

        return f"{path}/{run}"

    def _contour_plot(self):
        """
        Makes a contour plot of the function that is being optimized.
        """
        x, y = np.meshgrid(
            np.linspace(GRID_MIN, GRID_MAX, 200), np.linspace(GRID_MIN, GRID_MAX, 200)
        )
        z = self.opt_function(x, y)
        plt.contourf(x, y, z, cmap="plasma")
        plt.colorbar()

    def _particle_plot(self):
        """
        Plots the particles and their velocities.
        """
        # Plot particles
        self._particle = plt.scatter(*self.position, color="black")

        # Plot gradient
        self._grad_dir = plt.quiver(
            *self.position,
            *[-self.learning_rate * x for x in self.opt_function_grad(*self.position)],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="black",
        )

    def plot(self):
        """
        Produces a plot of the current state of the swarm.
        """
        plt.title(f"{self.opt_function.__name__.capitalize()} function")
        self._contour_plot()
        self._particle_plot()
        plt.show()

    def plot_opt_value_history(self):
        """
        Produces and saves a plot of the optimal value history.
        """
        plt.plot(
            list(range(1, len(self.opt_value_history) + 1)), self.opt_value_history
        )
        plt.xlabel("Epochs")
        plt.ylabel(f"{self.opt_function.__name__.capitalize()} function value")
        plt.title(
            "Evolution of the value found for the\n"
            f"{self.opt_function.__name__.capitalize()} function\n"
            "with respect to the epochs"
        )
        plt.tight_layout()
        path_run = self._get_path_run("./history/grad_descent")
        plt.savefig(f"{path_run}_{self.opt_function.__name__}.png")
        plt.clf()


if __name__ == "__main__":
    grad_descent = Grad_Descent(
        opt_function=rosenbrock, opt_function_grad=rosenbrock_grad
    )
    grad_descent.optimize(max_epochs=1000, animate=True)
