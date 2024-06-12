import random
import numpy as np
import wordle
import pandas as pd
import matplotlib.pyplot as plt

POPULATION_SIZE = 100
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.3
GENERATIONS = 1000


LETTERS = 26  # a - z
POSITIONS = 5 # 1 - 5

class Individual:
    def __init__(self):
        self.distribution = np.random.rand(POSITIONS,LETTERS)         
        self.fitness = 0
        self.env = wordle.Wordle()

    def construct_guess(self, prevGuess, prevGuessResults):
        samples = []
        for i in range(POSITIONS):
            indices = np.arange(1, len(self.distribution[i])+1)
            samples.append(random.choices(indices, weights=self.distribution[i], k=1)[0])
        
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
                if np.sum(fitness) > 24:
                    break
            p.fitness = fitness

            
        best_fitness = 0
        fitnesses = []
        best_guesses = []
        best_distribution = []
        for p in self.population:
            fitnesses.append(np.sum(p.fitness))

            if np.sum(p.fitness) > np.sum(best_fitness):
                best_fitness = p.fitness
                best_guesses = p.env
                best_distribution = p.distribution
        return fitnesses, best_fitness, best_guesses, best_distribution
    
    def mutate(self):
        parents = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:int(POPULATION_SIZE/4)]
        
        children = []
        for _ in range(2):
            for p in parents:
                kid1 = Individual()
                kid2 = Individual()
                spouse = parents[random.randint(0, len(parents)-1)]
            
                # cross-over
                kid1.distribution, kid2.distribution = self.crossover(p.distribution, spouse.distribution)

                # bit mutation
                kid1.distribution = self.add_variation_init(kid1.distribution)
                kid2.distribution = self.add_variation_init(kid2.distribution)

                children.extend([kid1, kid2])
       
        self.population = children[:POPULATION_SIZE]

    def add_variation_init(self, distribution):
        # total places = 26 * 5 (letters * positions)
        bits_to_switch = int(len(distribution)*len(distribution[0]) * MUTATION_RATE)

        for _ in range(bits_to_switch):
            square = random.randint(0,POSITIONS-1)
            index = random.randint(0, LETTERS-1)
            distribution[square][index] = min(1, max(0, distribution[square][index] + random.uniform(-0.5,0.5)))
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
        self.all_fitnesses = []

    def run(self):
        for g in range(self.nGenerations):
            print("gen",g)
            all_fitnesses, best_fitness, best_visual, best_init  = self.generation.run_generation()
            self.all_fitnesses.append(all_fitnesses)
            self.best_fitnesses.append(best_fitness)
            self.best_init.append(best_init)
            if np.sum(best_fitness) > 24:
                print(best_visual)
            self.generation.mutate()
        print(f"all fitnesses {self.all_fitnesses}")
        print(f"best fitnesses {self.best_fitnesses}")
    
    def initial_analysis(self):
        # Plot the initial weights of letters a-z for generation 
        indexes = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
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

    


      
alg = Algorithm()
alg.run()
alg.initial_analysis()
alg.plot_generations()
