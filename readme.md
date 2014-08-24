# Multi-Armed Bandit Policy Simulator

## Overview
This library contains a set of bandit policies along with a graphing tool to help evaluate the relative performance between policies on varying arms and payoff rates. The following policies have already been implemented:

* Standard A/B Test
* Epsilon-Greedy
* Annealing Epsilon-Greedy
* Softmax
* Annealing Softmax
* UCB1

The following policies have yet to be implemented:

* UCB2
* Exp3
* Thompson Sampling

## Installing Pylab

    sudo apt-get install python-matplotlib

## Example Usage

```python
from arms import BernoulliArm
from plotter import Plotter
from policies import *
```

Comparison between various Epsilon-Greedy policies:

```python
arms = [BernoulliArm(mu) for mu in [.1, .1, .7, 1.0]]
policies = [EpsilonGreedy(step * 0.2) for step in range(6)]
plotter = Plotter(arms, policies)
plotter.plot_results(num_trials=1000, num_pulls=200, metric='reward')
plotter.plot_results(num_trials=1000, num_pulls=200, metric='cumulative_reward')
```

![alt text](http://i.imgur.com/3Z4KQgp.png)
![alt text](http://i.imgur.com/bwKDBDQ.png)

Comparison between alternative policies:

```python
policies = [AnnealingEpsilonGreedy(), AnnealingSoftmax(), UCB1()]
plotter = Plotter(arms, policies)
plotter.plot_results(num_trials=1000, num_pulls=200, metric='reward')
plotter.plot_results(num_trials=1000, num_pulls=200, metric='cumulative_reward')
```
![alt text](http://i.imgur.com/EvYnAaY.png)
![alt text](http://i.imgur.com/jItsDBA.png)