import random

class Wordle: 
    def __init__(self):
        self.results = []
        with open("wordlist.txt", "r") as f:
            self.words = f.read().splitlines()
        self.target = self.words[random.randrange(0, len(self.words))]
    
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
            if guess[i] == self.target[i]:
                result.append(2)
            elif guess[i] in self.target:
                count = self.target.count(guess[i])
                if count >= occurrences[i]:
                    result.append(1)
                else:
                    result.append(0)
            else:
                result.append(0)
        self.results.append(result)
        return result
        

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
