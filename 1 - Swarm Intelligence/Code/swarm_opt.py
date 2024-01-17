import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from constants import *
from functions import *


class Swarm_Opt:
    """
    A class that performs swarm optimization.
    """

    def __init__(self, opt_function, particles, a_init=0.9, b_init=1, c_init=1):
        # Function to be optimized
        self.opt_function = opt_function

        # Randomly initialize particle positions
        self.positions = np.random.uniform(GRID_MIN, GRID_MAX, (particles, 2))

        # Randomly initialize particle velocities
        self.velocities = np.empty((particles, 2))
        for i in range(particles):
            t = np.random.uniform(0, 2 * np.pi)
            v = np.random.uniform() * np.asarray((np.cos(t), np.sin(t)))
            self.velocities[i] = v

        # Previous best positions for each particle
        self.prev_best = self.positions.copy()

        # Optimization parameters
        self._a = a_init
        self._b = b_init
        self._c = c_init

        # Optimal value history
        self.opt_value_history = [self.opt_function(*self._global_best())]

    def _global_best(self):
        """
        Retrieves the particle with the best (lowest) score.
        """
        min_score = np.inf
        for i, pos in enumerate(self.positions):
            score = self.opt_function(*pos)
            if score < min_score:
                min_score = score
                min_idx = i

        return self.positions[min_idx]

    def _update_prev_best(self):
        """
        Updates the previous best location for each particle
        if a better one is found.
        """
        for i, pos in enumerate(self.positions):
            prev_best_pos = self.prev_best[i]
            if self.opt_function(*pos) < self.opt_function(*prev_best_pos):
                self.prev_best[i] = pos

    def _set_parameters(self, epoch):
        """
        Modifies the parameters during the optimization process.
        """
        if epoch < 500:
            self._a -= 0.001
        else:
            self._b = 2
            self._c = 2

    def _opt_step(self, epoch):
        """
        Updates the velocity, position and previous best position
        for each particle.
        """
        # Set the optimization parameters
        self._set_parameters(epoch)

        # Update velocities
        r = np.random.uniform()
        self.velocities = (
            self._a * self.velocities
            + self._b * r * (self.prev_best - self.positions)
            + self._c * r * (self._global_best() - self.positions)
        )

        # Clip speeds that exceed MAX_SPEED
        for i, v in enumerate(self.velocities):
            speed = np.linalg.norm(v)
            if speed > MAX_SPEED:
                self.velocities[i] = v * MAX_SPEED / speed

        # Update positions
        self.positions = self.positions + self.velocities

        # Update the list of previous best locations
        self._update_prev_best()

        # Update the value history
        self.opt_value_history.append(self.opt_function(*self._global_best()))

    def _terminating_condition(self):
        """
        Terminate whenever the sum of the improvement from the previous best value
        for the last 10 epochs is less than SWARM_EPSILON.
        """
        return (
            len(self.opt_value_history) > 10
            and np.sum(
                np.abs(x - y)
                for x, y in zip(
                    self.opt_value_history[-10:], self.opt_value_history[-11:-1]
                )
            )
            < SWARM_EPSILON
        )

    def optimize(self, max_epochs, animate=False):
        """
        Optimization loop.
        """
        print()
        print(
            "-------------------------------- Optimizing "
            f"{self.opt_function.__name__.capitalize()} function "
            "using swarm optimization "
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
                self._opt_step(epoch)

                # Plot
                plt.title(
                    f"{self.opt_function.__name__.capitalize()} function (epoch {epoch})"
                )

                if epoch > 0:
                    self._particles_plot.remove()
                    self._velocities_plot.remove()
                self._swarm_plot()

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
            path_run = self._get_path_run("./animations/swarm_opt")
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
        x_opt, y_opt = self._global_best()
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

    def _swarm_plot(self):
        """
        Plots the particles and their velocities.
        """
        # Plot particles
        self._particles_plot = plt.scatter(*zip(*self.positions), color="black")

        # Plot velocities
        self._velocities_plot = plt.quiver(
            *zip(*self.positions),
            *zip(*self.velocities),
            angles="xy",
            scale_units="xy",
            scale=1,
            color="black",
        )

    def plot(self):
        """
        Produces and shows a plot of the current state of the swarm.
        """
        plt.title(f"{self.opt_function.__name__.capitalize()} function")
        self._contour_plot()
        self._swarm_plot()
        plt.show()
        plt.clf()

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
            "Evolution of the global best value found for the\n"
            f"{self.opt_function.__name__.capitalize()} function\n"
            "with respect to the epochs"
        )
        plt.tight_layout()
        path_run = self._get_path_run("./history/swarm_opt")
        plt.savefig(f"{path_run}_{self.opt_function.__name__}.png")
        plt.clf()


if __name__ == "__main__":
    swarm_opt = Swarm_Opt(opt_function=rosenbrock, particles=20)
    swarm_opt.optimize(max_epochs=1000, animate=True)
