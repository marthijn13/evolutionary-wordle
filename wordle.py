import random
import numpy as np

class Wordle: 
    words = []
    def __init__(self):
        self.words = []
        self.word = ''
        self.nGuess = 0
        self.loadWordlist()
    
    def __str__(self):
        return f'Goal: {self.word}, guesses: {self.nGuess}'

    def loadWordlist(self):
        with open("wordlist.txt", "r") as f:
            self.words = f.read().splitlines()

    def play(self):
        self.word = self.words[random.randrange(0, len(self.words))]
        
    def guessWord(self, guess):
        if self.nGuess > 6: 
            print("Done")
            return None
        
        results = []
        occurrences = letter_occurrence(guess)
        for i in range(len(guess)):
            if guess[i] == self.word[i]:
                results.append(2)
            elif guess[i] in self.word:
                count = self.word.count(guess[i])
                if count >= occurrences[i]:
                    results.append(1)
                else:
                    results.append(0)
            else:
                results.append(0)
        self.nGuess += 1
        return results
        

def letter_occurrence(word):
    occurrences = []
    seen = {}
    for letter in word:
        if letter in seen:
            seen[letter] += 1
        else:
            seen[letter] = 1
        occurrences.append(seen[letter])
    return occurrences

wordle = Wordle()
wordle.play()
print(wordle.guessWord("xtxtt"))
print(wordle)
