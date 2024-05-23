import random
import numpy as np
import wordle
import pandas as pd
import matplotlib.pyplot as plt

POPULATION_SIZE = 100
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.2

LETTERS = 26  # a - z
POSITIONS = 5 # 1 - 5
LAYERS = 4    # white, grey, green, yellow

class Individual:
    def __init__(self):
        self.initial = np.random.rand(POSITIONS, LETTERS) # random 26 x 5 matrix - for breeding purposes
        self.weights = np.random.rand(LAYERS, POSITIONS, LETTERS, POSITIONS, LETTERS) # random 26 x 5 x 26 x 5 matrix
        
        self.fitness = 0
        self.env = wordle.Wordle()

    def construct_guess(self, prevGuess, prevGuessResults): # TODO: implement different strategies using inheritance
        #TODO check which letters were correct and leave those
        samples = []
        dis = self.new_distribution()
        for i in range(POSITIONS):
            indices = np.arange(1, len(dis[i])+1)
            samples.append(random.choices(indices, weights=dis[i], k=1)[0])

        if len(prevGuessResults) == 0:       
            return ''.join([chr(s + 96) for s in samples])
        else:
            newGuess = []
            for i in range(POSITIONS): 
                if prevGuessResults[i] == 2:
                    newGuess.append(prevGuess[i])
                else:
                    newGuess.append(chr(samples[i] + 96))
            return ''.join(newGuess)

    def new_distribution(self):
        distribution = np.zeros((POSITIONS, LETTERS))
        for i in range(POSITIONS):
            for j in range(LETTERS):
                distribution[i][j] = max(0, min(1, self.initial[i][j] 
                                                + np.sum(self.weights[0][i][j].dot(self.env.green.T)) 
                                                + np.sum(self.weights[1][i][j].dot(self.env.yellow.T))
                                                + np.sum(self.weights[2][i][j].dot(self.env.white.T))
                                                + np.sum(self.weights[3][i][j].dot(self.env.gray.T))))
        return distribution
        

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
        weights = 0
        for p in self.population:
            if np.sum(p.fitness) > np.sum(best_fitness):
                best_fitness = p.fitness
                best_guesses = p.env
                best_distribution = p.initial
                weights = p.weights
        return best_fitness, best_guesses, best_distribution, weights
    
    def mutate(self):
        parents = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:int(POPULATION_SIZE/8)]
        
        children = []
        for _ in range(4):
            for p in parents:
                kid1 = Individual()
                kid2 = Individual()
                spouse = parents[random.randint(0, len(parents)-1)]
            
                # cross-over
                kid1.initial, kid2.initial = self.crossover_init(p.initial, spouse.initial)
                kid1.weights, kid2.weights = self.crossover_init(p.weights, spouse.weights)

                # bit mutation
                kid1.initial = self.add_variation_init(kid1.initial)
                kid2.initial = self.add_variation_init(kid2.initial)
                for l in range(LAYERS):
                    for p in range(POSITIONS):
                        for c in range(LETTERS):
                            kid1.weights[l][p][c] = self.add_variation_init(kid1.weights[l][p][c])
                            kid2.weights[l][p][c] = self.add_variation_init(kid2.weights[l][p][c])

                children.extend([kid1, kid2])
       
        self.population = children[:POPULATION_SIZE]

    def add_variation_init(self, distribution):
        # total places = 26 * 5 (letters * positions)
        bits_to_switch = int(len(distribution)*len(distribution[0]) * MUTATION_RATE)

        for _ in range(bits_to_switch):
            square = random.randint(0,4)
            index = random.randint(0, len(distribution)-1)
            distribution[square][index] = max(0, min(1, distribution[square][index] + random.uniform(-0.5,0.5)))
        return distribution
    
    def crossover_init(self, dist1, dist2):
        if random.random() < CROSSOVER_RATE:
            crossover_index = random.randint(0, len(dist1)-1)
            new_dist1 = dist1.copy()
            new_dist2 = dist2.copy()
            new_dist1[crossover_index:], new_dist2[:crossover_index] = new_dist2[crossover_index:], new_dist1[:crossover_index]
            return new_dist1, new_dist2
        return dist1, dist2

class Algorithm:
    def __init__(self):
        self.nGenerations = 20
        self.generation = Evolution()
        self.fitnesses = []
        self.initial_distributions = []
        self.weights = []
        self.spots = []

    def run(self):
        for g in range(self.nGenerations):
            print("gen",g)
            fitness_score, visual_rep, initial_dis, weights  = self.generation.run_generation()
            self.fitnesses.append(np.sum(fitness_score))
            self.initial_distributions.append(initial_dis)
            self.weights.append(weights)
            if np.sum(fitness_score) > 9:
                print(g)
                print(visual_rep)
            self.generation.mutate()
        print(self.fitnesses)
    
    def initial_analysis(self):
        # Plot the initial weights of letters a-z for generation 0, 5, 10, 15, 19
        indexes = [0, 4, 9, 14, 19]
        fig, ax = plt.subplots(len(indexes), 1, figsize=(10, 8))

        for idx, generation_idx in enumerate(indexes):
            heatmap = ax[idx].imshow(self.initial_distributions[generation_idx], cmap='viridis', aspect='auto')
            ax[idx].set_title(f'Generation {generation_idx}')
            ax[idx].set_xticks(np.arange(0, LETTERS, 1))
            ax[idx].set_yticks(np.arange(0, POSITIONS, 1))
            ax[idx].set_xticklabels([chr(ord("a") + i) for i in range(LETTERS)])
            ax[idx].set_yticklabels(range(1, POSITIONS + 1))
            plt.colorbar(heatmap, ax=ax[idx], orientation='vertical')

        plt.tight_layout()
        plt.show()

      
alg = Algorithm()
alg.run()
alg.initial_analysis()

#env = Evolution()
#fitness, guesses, distribution = env.run_generation()
#print(guesses)
#print(distribution)

