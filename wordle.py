import random
import numpy as np

POSITIONS = 5
LETTERS = 26
class Wordle: 
    def __init__(self):
        self.green  = np.zeros((POSITIONS, LETTERS))
        self.yellow = np.zeros((POSITIONS, LETTERS))
        self.white  = np.ones((POSITIONS, LETTERS))
        self.gray   = np.zeros((POSITIONS, LETTERS))

        self.results = []

        with open("sss.txt", "r") as f:
            self.words = f.read().splitlines()
        self.target = self.words[random.randrange(0, len(self.words))]
        #self.target = 'motor' # TODO: Remove after testing
    
    def __str__(self):
        string = ""
        for result in self.results:
            for block in result:
                if block == 5:
                    string = string + "\U0001f7e9"
                elif block == 3:
                    string = string + "\U0001F7E8"
                else:
                    string = string + "\U00002B1C"
            string = string + "\n"
        return f'Goal: {self.target}\n{string}'
        
    def guessWord(self, guess):    
        """
        params
        guess (string): 5 letter guess

        returns
        result (int[]): array of score per letter
        """   
        if len(self.results) > 6: 
            print("Game is already finished.")
            return None
        
        result = []
        occurrences = letter_occurrence(guess)
        for i in range(len(guess)):
            self.white[i][char2int(guess[i])] = 0
            if guess[i] == self.target[i]:
                self.green[i][char2int(guess[i])] = 1
                result.append(5)
            elif guess[i] in self.target:
                # TODO: deze implementatie betekent dat de interne representatie naar de agent geel krijgt ongeacht de hoeveelste gele hij is. Dit is een
                # hele handige denk ik - maar vult een stukje nodige redenatie in. Het betekent ook dat het aantal intern geel niet impliceert dat er meerdere 
                #zijn als er later een groene gevonden wordt
                self.yellow[i][char2int(guess[i])] = 1
                count = self.target.count(guess[i])
                if count >= occurrences[i]:
                    result.append(3)
                else:
                    result.append(0)
            else:
                self.gray[i][char2int(guess[i])] = 1 #TODO: implementation could be that we do this instantly for the entire grid but i would prefer not to to make it more 'human'
                result.append(0)
        self.results.append(result)
        return result, guess
        

def char2int(char):
    return ord(char) - ord('a')

def letter_occurrence(guess):
    occurrences = []
    seen = {}
    for letter in guess:
        if letter in seen:
            seen[letter] += 1
        else:
            seen[letter] = 1
        occurrences.append(seen[letter])
    return occurrences
