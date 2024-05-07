import random
import numpy as np
     
class Wordle: 
    def __init__(self):
        self.white = {chr(i): True for i in range(ord('a'), ord('z')+1)}
        self.gray = {chr(i): False for i in range(ord('a'), ord('z')+1)}
        self.green = {chr(i): [] for i in range(ord('a'), ord('z')+1)}
        self.yellow = {chr(i): [] for i in range(ord('a'), ord('z')+1)}        
        
        self.results = []

        with open("wordlist.txt", "r") as f:
            self.words = f.read().splitlines()
        self.target = self.words[random.randrange(0, len(self.words))]
        #self.target = 'motor' # TODO: Remove after testing
    
    def __str__(self):
        string = ""
        for result in self.results:
            for block in result:
                if block == 2:
                    string = string + "\U0001f7e9"
                elif block == 1:
                    string = string + "\U0001F7E8"
                else:
                    string = string + "\U00002B1C"
            string = string + "\n"
        return f'Goal: {self.target}\n{string}'
        
    def guessWord(self, guess):
               
        if len(self.results) > 6: 
            print("Game is already finished.")
            return None
        
        result = []
        occurrences = letter_occurrence(guess)
        for i in range(len(guess)):
            self.white[guess[i]] = False
            if guess[i] == self.target[i]:
                self.green = add_to_dict(self.green, guess[i], i)
                result.append(2)
            elif guess[i] in self.target:
                # TODO: deze implementatie betekent dat de interne representatie naar de agent geel krijgt ongeacht de hoeveelste gele hij is. Dit is een
                # hele handige denk ik - maar vult een stukje nodige redenatie in. Het betekent ook dat het aantal intern geel niet impliceert dat er meerdere 
                #zijn als er later een groene gevonden wordt
                self.yellow = add_to_dict(self.yellow, guess[i], i) 
                count = self.target.count(guess[i])
                if count >= occurrences[i]:
                    result.append(1)
                else:
                    result.append(0)
            else:
                self.gray[guess[i]] = True
                result.append(0)
        self.results.append(result)
        return result, guess
        

def add_to_dict(dict, char, loc):
    if dict[char]:
        dict[char].append(loc)
    else:
        if loc in dict[char]:
            return dict
        else:
            dict[char].append(loc)
    return dict


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
