import math
import random

class BasePolicy():
  def __init__(self, plays, values):
    self.plays = plays
    self.values = values

  def initialize(self, n_arms):
    self.plays = [0 for bandit in range(n_arms)]
    self.values = [0.0 for bandit in range(n_arms)]

  def ind_max(self, values):
    return values.index(max(values))

  def update(self, chosen_arm, reward):
    self.plays[chosen_arm] += 1
    arm_plays = self.plays[chosen_arm]
    arm_value = self.values[chosen_arm]
    self.values[chosen_arm] = (arm_value*(arm_plays-1) + reward)/float(arm_plays)

class EpsilonGreedy(BasePolicy):
  """
  Exploits the best arm with probability epsilon.
  Explores a random arm with probability 1-epsilon.
  """
  def __init__(self, epsilon, plays=[], values=[]):
    self.epsilon = epsilon
    self.name = 'EpsilonGreedy: %s' % epsilon
    BasePolicy.__init__(self, plays, values)

  def select_arm(self):
    if random.random() <= self.epsilon:
      return random.randrange(len(self.values))
    else:
      return self.ind_max(self.values)

class ABTest(EpsilonGreedy):
  """Always explores arms with equal probability."""
  def __init__(self, plays=[], values=[]):
    self.name = 'A/B Test'
    BasePolicy.__init__(self, 1.0, plays, values)

class AnnealingEpsilonGreedy(BasePolicy):
  """Epsilon Greedy policy where epsilon approaches 1.0 over time."""
  def __init__(self, plays=[], values=[]):
    self.name = 'AnnealingEpsilonGreedy'
    BasePolicy.__init__(self, plays, values)

  def select_arm(self):
    epsilon = 1/math.log(sum(self.plays) + 1.0000001)
    if random.random() <= epsilon:
      return random.randrange(len(self.values))
    else:
      return self.ind_max(self.values)

class Softmax(BasePolicy):
  """Selects arm based on observed payoff rates."""
  def __init__(self, temperature, plays=[], values=[]):
    self.temperature = temperature
    self.name = 'Softmax: %s' % temperature
    BasePolicy.__init__(self, plays, values)

  def select_arm(self):
    denominator = sum([math.exp(v/self.temperature) for v in self.values])
    probs = [math.exp(v/self.temperature)/denominator for v in self.values]
    z = random.random()
    cum_prob = 0.0
    for i, prob in enumerate(probs):
      cum_prob += prob
      if cum_prob > z:
        return i
    return len(probs) - 1

class AnnealingSoftmax(BasePolicy):
  """Softmax policy where temperature approaches 0.0 over time."""
  def __init__(self, plays=[], values=[]):
    self.name = 'AnnealingSoftmax'
    BasePolicy.__init__(self, plays, values)

  def select_arm(self):
    temperature = 1 / math.log(sum(self.plays) + 1.0000001)
    denominator = sum([math.exp(v/temperature) for v in self.values])
    probs = [math.exp(v/temperature)/denominator for v in self.values]
    z = random.random()
    cum_prob = 0.0
    for i, prob in enumerate(probs):
      cum_prob += prob
      if cum_prob > z:
        return i
    return len(probs) - 1

class UCB1(BasePolicy):
  """Selects arm according to confidence in payoff rates"""
  def __init__(self, plays=[], values=[]):
    self.name = 'UCB1'
    BasePolicy.__init__(self, plays, values)

  def select_arm(self):
    n_arms = len(self.plays)
    for arm in range(n_arms):
      if self.plays[arm] == 0:
        return arm

    ucb_values = [0.0 for arm in range(n_arms)]
    total_counts = sum(self.plays)
    for arm in range(n_arms):
      bonus = math.sqrt((2 * math.log(total_counts)) / float(self.plays[arm]))
      ucb_values[arm] = self.values[arm] + bonus
    return self.ind_max(ucb_values)