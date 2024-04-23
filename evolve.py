import random
import numpy as np
import wordle

POPULATION_SIZE = 20

class Individual:
    def __init__(self):
        self.distribution = [random.random() for _ in range(26)]
        self.fitness = 0
        self.env = wordle.Wordle()

    def construct_guess(self, information = []): # TODO: implement different strategies using inheritance
        indices = np.arange(1, len(self.distribution)+1)
        samples = random.choices(indices, weights=self.distribution, k=5)
        return ''.join([chr(s + 96) for s in samples])

class Evolution: 
    def __init__(self):
        self.population = []
        for _ in range(POPULATION_SIZE):
            self.population.append(Individual())

    def run_generation(self):
        for p in self.population:
            for _ in range(6):
                fitness = p.env.guessWord(p.construct_guess())
                if np.sum(fitness) > 9:
                    print(p.env)
            # TODO: change fitness after implementing guesses based on any information instead of random guessing
            p.fitness = np.sum(p.env.results) 
        best_fitness = 0
        best_guesses = []
        best_distribution = []
        for p in self.population:
            if p.fitness > best_fitness:
                best_fitness = p.fitness
                best_guesses = p.env
                best_distribution = p.distribution
        return best_fitness, best_guesses, best_distribution
        
            
env = Evolution()
fitness, guesses, distribution = env.run_generation()
print(guesses)
print(distribution)