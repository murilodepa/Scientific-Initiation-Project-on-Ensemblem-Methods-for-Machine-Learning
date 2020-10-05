import random
class RandomClassifier:
    def __init__(self):
        print("Creating random...")
        self.max_v = 0
        self.min_v = 0
        
    def fit(self, x, y):
        self.max_v = max(y)
        self.min_v = min(y)
        #for v in y:
        #    if v > self.max_v:
        #        self.max_v = v
        #    if v < self.min_v:
        #        self.min_v = v
        
    def predict(self, x):
        return [random.randint(self.min_v, self.max_v) for _ in x]
