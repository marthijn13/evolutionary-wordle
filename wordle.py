import random

class Wordle: 
    def __init__(self):
        self.nGuess = 0
        with open("wordlist.txt", "r") as f:
            self.words = f.read().splitlines()
        self.target = self.words[random.randrange(0, len(self.words))]
    
    def __str__(self):
        return f'Goal: {self.target}, guesses: {self.nGuess}'
        
    def guessWord(self, guess):
        if self.nGuess > 6: 
            print("Done")
            return None
        
        results = []
        occurrences = letter_occurrence(guess)
        for i in range(len(guess)):
            if guess[i] == self.target[i]:
                results.append(2)
            elif guess[i] in self.target:
                count = self.target.count(guess[i])
                if count >= occurrences[i]:
                    results.append(1)
                else:
                    results.append(0)
            else:
                results.append(0)
        self.nGuess += 1
        return results
        

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

wordle = Wordle()
print(wordle.guessWord("xtxtt"))
print(wordle)
