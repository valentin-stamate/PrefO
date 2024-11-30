import random

import numpy
import pygad
from pygad import GA

function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44


def generation_callback(ga_instance: GA):
    # Track progress, print intermediate results, or perform other actions
    generation = ga_instance.generations_completed
    fitness = ga_instance.best_solution()[1]
    print(f"Generation: {generation}, Fitness: {fitness}")


def fitness_func(ga_instance: GA, solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / numpy.abs(output - desired_output)
    return fitness


def mutation(offspring_sol, mutation_probability):
    for i in range(len(offspring_sol)):
        if random.random() < mutation_probability:
            offspring_sol[i] = 1 - offspring_sol[i]


def custom_crossover(parent_sol1, parent_sol2):
    # Implement your custom crossover logic here
    # Example: Single-point crossover
    crossover_point = random.randint(1, len(parent_sol1) - 2)  # Avoid first and last genes
    offspring = parent_sol1[:crossover_point] + parent_sol2[crossover_point:]
    return offspring


def custom_population_init(ga_instance: GA):
    """
    Custom function to initialize the population.

    Args:
        ga_instance (pygad.GA): The PyGAD instance.

    Returns:
        list: The initialized population (using default initialization).
    """

    population = ga_instance.initialize_population()  # Use default initialization
    return population


def main():

    init_population = numpy.array([]);

    ga_instance = GA(num_generations=50,
                     num_parents_mating=4,
                     initial_population=custom_population_init,
                     fitness_func=fitness_func,
                     mutation_type=mutation,
                     mutation_probability=0.01,
                     crossover_type=custom_crossover,
                     sol_per_pop=4,
                     num_genes=10000,
                     init_range_low=-2,
                     init_range_high=5,
                     parent_selection_type='sss',
                     keep_parents=1,
                     gene_type=int,
                     gene_space=[0, 1],
                     # mutation_type='random',
                     # mutation_percent_genes=10,
                     keep_elitism=True)

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    # prediction = numpy.sum(numpy.array(function_inputs)*solution)
    # print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

    print(f'Done')


if __name__ == '__main__':
    main()
