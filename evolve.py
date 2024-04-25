import random
import numpy as np
import wordle

POPULATION_SIZE = 20

class Individual:
    def __init__(self):
        self.distribution = [random.random() for _ in range(26)]
        self.fitness = 0
        self.env = wordle.Wordle()

    def construct_guess(self, prevGuess, prevGuessResults): # TODO: implement different strategies using inheritance
        #TODO check which letters were correct and leave those
        indices = np.arange(1, len(self.distribution)+1)
        samples = random.choices(indices, weights=self.distribution, k=5)

        if len(prevGuessResults) == 0:       
            return ''.join([chr(s + 96) for s in samples])
        else:
            newGuess = []
            for i in range(5): 
                if prevGuessResults[i] == 2:
                    newGuess.append(prevGuess[i])
                else:
                    newGuess.append(chr(samples[i] + 96))
            return ''.join(newGuess)

        

class Evolution: 
    def __init__(self):
        self.population = []
        for _ in range(POPULATION_SIZE):
            self.population.append(Individual())

    def run_generation(self):
        for p in self.population:
            prevGuessResults = []
            prevGuess = ''
            for _ in range(6):
                fitness, guess = p.env.guessWord(p.construct_guess(prevGuess, prevGuessResults))
                prevGuessResults = fitness
                prevGuess = guess
                if np.sum(fitness) > 9:
                    print(p.env)
            # TODO: change fitness after implementing guesses based on any information instead of random guessing
            p.fitness = (p.env.results[-1]) 

            
        best_fitness = 0
        best_guesses = []
        best_distribution = []
        for p in self.population:
            if np.sum(p.fitness) > best_fitness:
                best_fitness = np.sum(p.fitness)
                best_guesses = p.env
                best_distribution = p.distribution
        return best_fitness, best_guesses, best_distribution
    
    #TODO actually mutate the distributions
    def mutate(self):
        self.population = []
        for _ in range(POPULATION_SIZE):
            self.population.append(Individual())
        

class Algorithm:
    def __init__(self):
        self.nGenerations = 100
        self.generation = Evolution()
        self.fitnesses = []
        self.guessesList = []
        self.distributions = []

    def run(self):
        for g in range(self.nGenerations):
            fitness, guesses, distribution = self.generation.run_generation()
            self.fitnesses.append(fitness)
            self.guessesList.append(guesses)
            self.distributions.append(distribution)

            if np.sum(fitness) > 9:
                print(guesses)
            self.generation.mutate()

        print(self.fitnesses)

alg = Algorithm()
alg.run()


#env = Evolution()
#fitness, guesses, distribution = env.run_generation()
#print(guesses)
#print(distribution)

