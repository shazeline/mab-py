from pylab import *

class Plotter():
  def __init__(self, arms, policies):
    self.arms = arms
    self.policies = policies

  def simulate_policy(self, policy, num_pulls):
    """Simulate a specified number of pulls using a given policy."""
    chosen_arms = []
    rewards = []
    cumulative_rewards = []
    for pull in range(num_pulls):
      arm = policy.select_arm()
      reward = self.arms[arm].draw()
      policy.update(arm, reward)
      chosen_arms.append(arm)
      rewards.append(reward)
      last_total = 0
      if len(cumulative_rewards) > 0:
        last_total = cumulative_rewards[pull-1]
      cumulative_rewards.append(last_total + reward)
    return rewards, cumulative_rewards

  def average_list(self, data, trials):
    return [entry/float(trials) for entry in data]

  def run_trials(self, num_trials, policy, num_pulls):
    """Simulate a policy for over a number of trials and return key metrics."""
    tot_rewards = [0.0] * num_pulls
    tot_cumulative_rewards = [0.0] * num_pulls
    for simulation in range(num_trials):
      policy.initialize(len(self.arms))
      rewards, cumulative_rewards = self.simulate_policy(policy, num_pulls)
      for i, value in enumerate(rewards):
        tot_rewards[i] += value
      for i, value in enumerate(cumulative_rewards):
        tot_cumulative_rewards[i] += value
    return self.average_list(tot_rewards, num_trials), \
           self.average_list(tot_cumulative_rewards, num_trials)

  def rstyle(self, ax):
    """
    Styles an axis to appear like ggplot2. Must be called after all plot and
    axis manipulation operations have been carried out.
    """
    # Set the style of the major and minor grid lines, filled blocks.
    ax.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
    ax.grid(True, 'minor', color='0.92', linestyle='-', linewidth=0.7)
    ax.patch.set_facecolor('0.85')
    ax.set_axisbelow(True)

    # Set minor tick spacing to 1/2 of the major ticks.
    ax.xaxis.set_minor_locator(MultipleLocator( (plt.xticks()[0][1]-plt.xticks()[0][0]) / 2.0 ))
    ax.yaxis.set_minor_locator(MultipleLocator( (plt.yticks()[0][1]-plt.yticks()[0][0]) / 2.0 ))

    # Remove axis border.
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_alpha(0)

    # Restyle the tick lines.
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(5)
        line.set_color("gray")
        line.set_markeredgewidth(1.4)

    # Remove the minor tick lines.
    for line in ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines(minor=True):
        line.set_markersize(0)

    # Only show bottom left ticks, pointing out of axis.
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if ax.legend_ <> None:
        lg = ax.legend_
        lg.get_frame().set_linewidth(0)
        lg.get_frame().set_alpha(0.5)

  def plot_results(self, num_trials=1000, num_pulls=200, metric='cumulative_reward'):
    """Plot the performance of various policies for a specified metric."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('%s over %s pulls and %s trials' % (metric, num_pulls, num_trials))
    for policy in self.policies:
      reward, cumulative_reward = self.run_trials(num_trials, policy, num_pulls)
      plot_type = {
        'reward' : reward,
        'cumulative_reward' : cumulative_reward
      }
      plot(range(num_pulls), plot_type[metric], label = policy.name)
    ax.legend(loc=4)
    self.rstyle(ax)
    plt.show()