import random


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
        for i in range(len(guess)):
            if guess[i] == self.word[i]:
                results.append(2)
            elif guess[i] in self.word:
                results.append(1) #TODO Fix the orange blocks
            else:
                results.append(0)
        self.nGuess += 1
        return results
        


wordle = Wordle()
wordle.play()
print(wordle.guessWord("thing"))
print(wordle.guessWord("facet"))
print(wordle)
