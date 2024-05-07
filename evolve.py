import random
import numpy as np
import wordle
import pandas as pd

POPULATION_SIZE = 20
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.5

class Individual:
    def __init__(self):
        self.distribution = [[random.random() for _ in range(26)] for _ in range(5)]
        self.fitness = 0
        self.env = wordle.Wordle()

    def construct_guess(self, prevGuess, prevGuessResults): # TODO: implement different strategies using inheritance
        #TODO check which letters were correct and leave those
        samples = []
        for i in range(5):
            indices = np.arange(1, len(self.distribution[i])+1)
            samples.append(random.choices(indices, weights=self.distribution[i], k=1)[0])

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
                #if np.sum(fitness) > 9:
                #    print(p.env)
            # TODO: change fitness after implementing guesses based on any information instead of random guessing
            p.fitness = (p.env.results[-1]) 

            
        best_fitness = 0
        best_guesses = []
        best_distribution = []
        for p in self.population:
            if np.sum(p.fitness) > np.sum(best_fitness):
                best_fitness = p.fitness
                best_guesses = p.env
                best_distribution = p.distribution
        return best_fitness, best_guesses, best_distribution
    
    #TODO actually mutate the distributions
    def mutate(self):
        parents = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:int(POPULATION_SIZE/2)]
        
        children = []
        for p in parents:
            kid1 = Individual()
            kid2 = Individual()

            # cross-over
            kid1.distribution, kid2.distribution = self.crossover(p.distribution, parents[random.randint(0, len(parents)-1)].distribution)
            
            # bit mutation
            kid1.distribution = self.add_variation(kid1.distribution)
            kid2.distribution = self.add_variation(kid2.distribution)
            children.extend([kid1, kid2])
        
        self.population = children[:POPULATION_SIZE]

    def add_variation(self, distribution):
        # total places = 26 * 5 (letters * positions)
        bits_to_switch = int(len(distribution)*len(distribution[0]) * MUTATION_RATE)

        for _ in range(bits_to_switch):
            square = random.randint(0,4)
            index = random.randint(0, len(distribution)-1)
            distribution[square][index] = max(0, min(1, distribution[square][index] + random.uniform(-0.5,0.5)))
        return distribution
    
    def crossover(self, dist1, dist2):
        if random.random() < CROSSOVER_RATE:
            crossover_index = random.randint(0, len(dist1)-1)
            new_dist1 = dist1.copy()
            new_dist2 = dist2.copy()
            new_dist1[crossover_index:], new_dist2[:crossover_index] = new_dist2[crossover_index:], new_dist1[:crossover_index]
            return new_dist1, new_dist2
        return dist1, dist2

class Algorithm:
    def __init__(self):
        self.nGenerations = 100
        self.generation = Evolution()
        self.fitnesses = []
        self.guessesList = []
        self.distributions = []
        self.output = pd.DataFrame(columns = ['generation', 'fitness', 'distribution'])

    def run(self):
        for g in range(self.nGenerations):
            fitness, guesses, distribution = self.generation.run_generation()
            self.fitnesses.append(np.sum(fitness))
            self.guessesList.append(guesses)
            self.distributions.append(distribution)

            if np.sum(fitness) > 9:
                print(g)
                print(guesses)
            self.generation.mutate()
            self.output = self.output._append({'generation': g, 'fitness': fitness, 'distribution': distribution}, ignore_index=True)
        print(self.fitnesses)
        self.output.to_csv('csv/output.csv', index=False)

alg = Algorithm()
alg.run()


#env = Evolution()
#fitness, guesses, distribution = env.run_generation()
#print(guesses)
#print(distribution)

