import random
import numpy as np
import wordle
import pandas as pd
import matplotlib.pyplot as plt

POPULATION_SIZE = 100
MUTATION_RATE = 0.5
CROSSOVER_RATE = 0.3
GENERATIONS = 20


LETTERS = 26  # a - z
POSITIONS = 5 # 1 - 5
LAYERS = 4    # white, grey, green, yellow

class Individual:
    def __init__(self):
        self.initial = np.random.rand(POSITIONS, LETTERS) # random 26 x 5 matrix - for breeding purposes
        self.weights = [[[[[random.uniform(-1, 1) for _ in range(LETTERS)] for _ in range(POSITIONS)] for _ in range(LETTERS)] for _ in range(POSITIONS)] for _ in range(LAYERS)]
        self.weights = np.array(self.weights)
        #self.weights = np.random.rand(LAYERS, POSITIONS, LETTERS, POSITIONS, LETTERS) # random 26 x 5 x 26 x 5 matrix
        
        self.fitness = 0
        self.env = wordle.Wordle()

    def construct_guess(self, prevGuess, prevGuessResults):
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
                if prevGuessResults[i] == 5:
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
            for i in range(6):
                fitness, guess = p.env.guessWord(p.construct_guess(prevGuess, prevGuessResults))
                prevGuessResults = fitness
                prevGuess = guess
                if np.sum(fitness) > 24:
                    p.fitness = fitness * (6-i)
                    break
            # TODO: change fitness after implementing guesses based on any information instead of random guessing
            p.fitness = fitness

            
        best_fitness = 0
        fitnesses = []
        best_guesses = []
        best_distribution = []
        weights = []
        for p in self.population:
            fitnesses.append(np.sum(p.fitness))

            if np.sum(p.fitness) > np.sum(best_fitness):
                best_fitness = p.fitness
                best_guesses = p.env
                best_distribution = p.initial
                weights = p.weights
        return fitnesses, best_fitness, best_guesses, best_distribution, weights
    
    def mutate(self):
        parents = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:int(POPULATION_SIZE/4)]
        
        children = []
        for _ in range(2):
            for p in parents:
                kid1 = Individual()
                kid2 = Individual()
                spouse = parents[random.randint(0, len(parents)-1)]
            
                # cross-over
                kid1.initial, kid2.initial = self.crossover(p.initial, spouse.initial)
                kid1.weights, kid2.weights = self.crossover(p.weights, spouse.weights)

                # bit mutation
                kid1.initial = self.add_variation_init(kid1.initial)
                kid2.initial = self.add_variation_init(kid2.initial)
                for l in range(LAYERS):
                    for p in range(POSITIONS):
                        for c in range(LETTERS):
                            kid1.weights[l][p][c] = self.add_variation_weights(kid1.weights[l][p][c])
                            kid2.weights[l][p][c] = self.add_variation_weights(kid2.weights[l][p][c])

                children.extend([kid1, kid2])
       
        self.population = children[:POPULATION_SIZE]

    def add_variation_init(self, distribution):
        # total places = 26 * 5 (letters * positions)
        bits_to_switch = int(len(distribution)*len(distribution[0]) * MUTATION_RATE)

        for _ in range(bits_to_switch):
            square = random.randint(0,POSITIONS-1)
            index = random.randint(0, LETTERS-1)
            distribution[square][index] = max(0, min(1, distribution[square][index] + random.uniform(-1,1)))
        return distribution
    
    def add_variation_weights(self, distribution):
        # total places = 26 * 5 (letters * positions)
        bits_to_switch = int(len(distribution)*len(distribution[0]) * MUTATION_RATE)

        for _ in range(bits_to_switch):
            square = random.randint(0,POSITIONS-1)
            index = random.randint(0, LETTERS-1)
            distribution[square][index] = distribution[square][index] + random.uniform(-1,1)
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
        self.nGenerations = GENERATIONS
        self.generation = Evolution()
        self.best_fitnesses = []
        self.best_init = []
        self.best_weights = []
        self.all_fitnesses = []

    def run(self):
        for g in range(self.nGenerations):
            print("gen",g)
            all_fitnesses, best_fitness, best_visual, best_init, best_weights  = self.generation.run_generation()
            self.all_fitnesses.append(all_fitnesses)
            self.best_fitnesses.append(best_fitness)
            self.best_init.append(best_init)
            self.best_weights.append(best_weights)
            if np.sum(best_fitness) > 24:
                print(best_visual)
            self.generation.mutate()
        print(f"all fitnesses {self.all_fitnesses}")
        print(f"best fitnesses {self.best_fitnesses}")
    
    def initial_analysis(self):
        # Plot the initial weights of letters a-z for generation 0, 5, 10, 15, 19
        #indexes = [0, 19, 39, 59, 79, 99]
        indexes = [0, 4, 9, 14, 19]
        fig, ax = plt.subplots(len(indexes), 1, figsize=(10, 8))

        for idx, generation_idx in enumerate(indexes):
            heatmap = ax[idx].imshow(self.best_init[generation_idx], cmap='viridis', aspect='auto')
            ax[idx].set_title(f'Generation {generation_idx}')
            ax[idx].set_xticks(np.arange(0, LETTERS, 1))
            ax[idx].set_yticks(np.arange(0, POSITIONS, 1))
            ax[idx].set_xticklabels([chr(ord("a") + i) for i in range(LETTERS)])
            ax[idx].set_yticklabels(range(1, POSITIONS + 1))
            plt.colorbar(heatmap, ax=ax[idx], orientation='vertical')

        plt.tight_layout()
        plt.show()
    
    def plot_generations(self):
        # Define your data - replace this with your own data
        generations = self.all_fitnesses
        num_individuals = len(generations[0])

        avg_scores = []
        best_scores = []
        stds = []
        for gen in generations:
            avg = sum(gen) / num_individuals
            std = np.std(gen)
            stds.append(std)
            avg_scores.append(avg)
            best_scores.append(max(gen))

        # Calculate average +/- standard deviation values
        avg_min_std = [a - s for a, s in zip(avg_scores, stds)]
        avg_plus_std = [a + s for a, s in zip(avg_scores, stds)]

        # Plotting
        fig, ax = plt.subplots()
        ax.set_title("Scores Over Generations")
        ax.set_ylabel("Score")
        ax.set_xlabel("Generation Index")

        ax.plot(range(len(generations)), [25 for _ in range(len(generations))], label = 'Optimal score')
        ax.plot(range(len(generations)), avg_scores, label="Average Score", linestyle='--')
        ax.fill_between(range(len(generations)), avg_min_std, avg_plus_std, alpha=0.5)
        ax.plot(range(len(generations)), best_scores, label="Best Score", color='red', marker='o')
        ax.legend()

        plt.show()

    def informed_analysis(self, green, yellow, white, gray):
        # Plot the initial weights of letters a-z for generation 0, 5, 10, 15, 19
        #indexes = [0, 19, 39, 59, 79, 99]
        indexes = [0, 4, 9, 14, 19]
        fig, ax = plt.subplots(len(indexes), 1, figsize=(10, 8))
        
        distributions = []
        for init in self.best_init:

            distribution = np.zeros((POSITIONS, LETTERS))

            for idx in indexes:
                for i in range(POSITIONS):
                    for j in range(LETTERS):
                        distribution[i][j] = max(0, min(1, init[i][j] 
                                                        + np.sum(self.best_weights[idx][0][i][j].dot(green.T)) 
                                                        + np.sum(self.best_weights[idx][1][i][j].dot(yellow.T))
                                                        + np.sum(self.best_weights[idx][2][i][j].dot(white.T))
                                                        + np.sum(self.best_weights[idx][3][i][j].dot(gray.T))))
                distributions.append(distribution)
        for idx, generation_idx in enumerate(indexes):
            heatmap = ax[idx].imshow(distributions[generation_idx], cmap='viridis', aspect='auto')
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
alg.plot_generations()


green  = np.zeros((POSITIONS, LETTERS))
yellow = np.zeros((POSITIONS, LETTERS))
white  = np.ones((POSITIONS, LETTERS))
gray   = np.zeros((POSITIONS, LETTERS))
for i in range(POSITIONS):
    white[i][0] = 0 # a
    white[i][4] = 0 # e
    white[i][2] = 0 # c
    white[i][1] = 0 # b
    white[i][15] = 0 # p
    gray[i][2] = 1  # c
    gray[i][1] = 1  # b


green[3][0] = 1
green[1][4] = 1
yellow[2][15] = 1
# word is petal
# e and a are guessed correctly and p wrong location, c and b also guessed but nothing
alg.informed_analysis(green, yellow, white, gray)
#env = Evolution()
#fitness, guesses, distribution = env.run_generation()
#print(guesses)
#print(distribution)

