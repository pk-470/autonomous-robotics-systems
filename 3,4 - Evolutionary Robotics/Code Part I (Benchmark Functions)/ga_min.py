import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from constants import *


class GA_Min:
    def __init__(self, config):
        self.config = config
        self.function = config["function"]
        self.initialize(config["organisms_num"])
        self.make_output_dir(OUTPUT_PATH)

        # Histories for plots
        self.history_best = []
        self.history_average = []
        self.history_worst = []
        self.history_diversity = []

        self.decay_rate = 1

    # ==================== Initialization functions ====================

    def initialize(self, organisms_num):
        """
        Randomly initialize the organisms (which consist of points (x, y) on the plane).
        """
        self.organisms_num = organisms_num
        self.organisms = np.random.uniform(GRID_MIN, GRID_MAX, (organisms_num, 2))

    # ==================== Evaluation functions ====================

    def evaluate_single(self, organism):
        """
        Evaluate a single organism by applying the function that is
        to be minimized.
        """
        return self.function(*organism)

    def evaluate_all(self):
        """
        Evaluate all organisms.
        """
        return np.array([self.evaluate_single(organism) for organism in self.organisms])

    def sort_by_evaluation(self):
        """
        Sort the organisms in ascending order depending on their evaluation.
        Note that since we are interested in minimizing the function, the
        resulting list is sorted from best to worst.
        """
        organisms_sorted, values_sorted = zip(
            *sorted(zip(self.organisms, self.evaluate_all()), key=lambda ov: ov[1])
        )
        return np.array(organisms_sorted), np.array(values_sorted)

    def global_best(self):
        """
        Returns the organism with the best evaluation.
        """
        return self.sort_by_evaluation()[0][0]

    def diversity(self):
        """
        Returns the diversity of the population as measured by the sum
        of the Euclidean distances between the organisms.
        """
        return np.sum(
            np.linalg.norm(x - y)
            for i, x in enumerate(self.organisms)
            for y in self.organisms[i + 1 :]
        )

    # ==================== Crossover functions ====================

    def average_best_worst(self, best, worst):
        """
        Combine the genetic material of the best and worst
        organisms by taking their average in pairs.
        """
        return (best + worst[::-1]) / 2

    def average_best(self, best):
        """
        Combines the genetic material of the best
        organisms by taking the average of all.
        """
        return np.average(best, axis=0)

    # ==================== Mutation functions ====================

    def mutate(self, organisms):
        """
        Mutate some organisms by adding random noise in the range [-1, 1].
        """
        organisms += self.decay_rate * np.random.uniform(-1, 1, (len(organisms), 2))
        return organisms

    # ==================== Genetic algorithm ====================

    def _clip_organisms(self, organisms):
        """
        Clip the coordinates of each organism to ensure
        that they stay within the displayed grid.
        """
        for i, organism in enumerate(organisms):
            if organism[0] < GRID_MIN:
                organisms[i][0] = GRID_MIN
            if organism[0] > GRID_MAX:
                organisms[i][0] = GRID_MAX
            if organism[1] < GRID_MIN:
                organisms[i][1] = GRID_MIN
            if organism[1] > GRID_MAX:
                organisms[i][1] = GRID_MAX

        return organisms

    def _log_histories(self, values_sorted):
        """
        Log the histories for plotting.
        """
        self.history_best.append(values_sorted[0])
        self.history_worst.append(values_sorted[-1])
        self.history_average.append(np.average(values_sorted))
        self.history_diversity.append(self.diversity())

    def generation(self):
        """
        Create a new generation of organisms from the
        current one.
        """
        new_generation = []
        organisms_sorted, values_sorted = self.sort_by_evaluation()

        if self.config["mode"] == 1:
            # Evaluation & Selection
            best_fifth = organisms_sorted[: self.organisms_num // 5]
            middle = organisms_sorted[
                self.organisms_num // 5 : -self.organisms_num // 5
            ]
            worst_fifth = organisms_sorted[-self.organisms_num // 5 :]

            # Crossover & Mutation
            best_fifth_avg = self.average_best(best_fifth)
            worst_fifth_evolved = self.average_best_worst(best_fifth, worst_fifth)
            worst_fifth_mutated = self.mutate(worst_fifth_evolved)
            middle_mutated = self.mutate(middle)

            # Reproduction
            new_generation.append(best_fifth_avg)
            new_generation.extend(best_fifth[:-1])
            new_generation.extend(middle_mutated)
            new_generation.extend(worst_fifth_mutated)

            if self.decay_rate > MIN_DECAY_RATE:
                self.decay_rate -= DECAY_RATE_DEC

        elif self.config["mode"] == 2:
            best_solutions = organisms_sorted[: self.organisms_num * 20 // 100]

            elements = []
            for s in best_solutions:
                elements.append(s[0])
                elements.append(s[1])

            # Mutation
            for _ in range(self.organisms_num):
                e1 = np.random.choice(elements) * np.random.uniform(0.99, 1.01)
                e2 = np.random.choice(elements) * np.random.uniform(0.99, 1.01)

                new_generation.append((e1, e2))

        # Update the organisms with the new generation
        self.organisms = self._clip_organisms(np.array(new_generation))

        # Log the histories for plotting
        self._log_histories(values_sorted)

    def genetic_algorithm(self, generations, animate=False):
        """
        Applies the genetic algorithm in order to minimize a function.
        Also produces an animation of the evolution process.
        """
        print()
        print(
            f"-------------------------------- Genetic algorithm ({self.config['mode']}) "
            f"applied to {self.function.__name__.capitalize()} function "
            f"over {generations} generations --------------------------------"
        )

        progress_bar = tqdm(range(generations), leave=False)

        if animate:
            fig = plt.figure()

            def anim_step(generation):
                """
                Defines each step of the animation.
                """
                self.generation()

                plt.title(
                    f"{self.function.__name__.capitalize()} function (generation {generation})"
                )

                if generation > 0:
                    self._organisms_plot.remove()
                self.plot_organisms()

                progress_bar.update()

            # Create and save the animation
            animation = FuncAnimation(
                fig,
                anim_step,
                frames=generations,
                init_func=self.contour_plot,
                save_count=generations,
                repeat=False,
            )
            animation.save(f"{self.output_path}/{self.function.__name__}.gif", fps=20)

            plt.clf()

        else:
            for _ in range(generations):
                self.generation()
                progress_bar.update()

        progress_bar.close()

        # Create the plots
        self.plot_history_best_worst_avg()
        self.plot_history_diversity()

        # Save the configuration dictionary
        with open(f"{self.output_path}/config.json", "w") as config_out:
            config = self.config
            config["function"] = config["function"].__name__
            config["generations"] = generations
            json.dump(config, config_out)

        x_opt, y_opt = self.global_best()
        print(f"Optimal point: [{x_opt:.4g}, {y_opt:.4g}]")
        print(f"Optimal value: {self.function(x_opt, y_opt):.4g}")
        print()

    # ==================== Plotting functions ====================

    def contour_plot(self):
        """
        Creates a contour plot of the function.
        """
        x, y = np.meshgrid(
            np.linspace(GRID_MIN, GRID_MAX, 200), np.linspace(GRID_MIN, GRID_MAX, 200)
        )
        z = self.function(x, y)
        plt.contourf(x, y, z, cmap="plasma")
        x_min = x.ravel()[z.argmin()]
        y_min = y.ravel()[z.argmin()]
        plt.plot(x_min, y_min, "2", markersize=20, c="white")
        plt.colorbar()

    def plot_organisms(self):
        """
        Plots the organisms.
        """
        self._organisms_plot = plt.scatter(*zip(*self.organisms), color="black")

    def plot(self):
        """
        Creates and shows a contour plot of the function along with the organisms.
        """
        plt.title(f"{self.function.__name__.capitalize()} function")
        self.contour_plot()
        self.plot_organisms()
        plt.show()
        plt.clf()

    def plot_history_best_worst_avg(self):
        """
        Produces and saves a plot of the best, worst and average values history.
        """
        plt.plot(
            list(range(1, len(self.history_best) + 1)), self.history_best, label="best"
        )
        plt.plot(
            list(range(1, len(self.history_average) + 1)),
            self.history_average,
            label="average",
        )
        plt.plot(
            list(range(1, len(self.history_worst) + 1)),
            self.history_worst,
            label="worst",
        )
        plt.xlabel("Generations")
        plt.ylabel(f"{self.function.__name__.capitalize()} function value")
        plt.title(
            "Evolution of the global best, worst and average values\n"
            f"found for the {self.function.__name__.capitalize()} function\n"
            "with respect to the generations"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/{self.function.__name__}_best_worst_avg.png")
        plt.clf()

    def plot_history_diversity(self):
        """
        Produces and saves a plot of the diversity history.
        """
        plt.plot(
            list(range(1, len(self.history_diversity) + 1)), self.history_diversity
        )
        plt.xlabel("Generations")
        plt.ylabel(f"Diversity")
        plt.title(
            "Evolution of the population diversity during minimization of the\n"
            f"{self.function.__name__.capitalize()} function\n"
            "with respect to the generations"
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/{self.function.__name__}_diversity.png")
        plt.clf()

    def make_output_dir(self, path):
        """
        Creates the output directory in which the animations
        and plots will be saved.
        """
        run = 0
        for name in os.listdir(path):
            prev_run = int(name.split("_")[1])
            if prev_run > run:
                run = prev_run
        run += 1

        self.output_path = f"{path}/run_{run}"
        os.mkdir(self.output_path)
