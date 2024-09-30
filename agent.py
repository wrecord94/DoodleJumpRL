import random


class RandomAgent:
    def __init__(self):
        self.choices = {0: 'DO_NOTHING',
                        1: 'RIGHT',
                        2: 'LEFT'}

    def choose_action(self):
        action = random.choice(self.choices)
        print(f"Random action chosen is: {action}")

        return action