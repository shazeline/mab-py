import random

class BernoulliArm():
  def __init__(self, p):
    self.p = p

  def draw(self):
    return 1.0 if random.random() <= self.p else 0.0

class NormalArm():
  def __init__(self, mu, sigma):
    self.mu = mu
    self.sigma = sigma

  def draw(self):
    return random.gauss(self.mu, self.sigma)