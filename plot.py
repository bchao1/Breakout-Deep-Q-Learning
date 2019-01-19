import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.style.use('ggplot')

def avg(r):
    return [np.mean(r[i - 100:i]) for i in range(100, len(r))]


clipped_scores = np.load('./logs/clipped_log.npy')

unclipped_scores = np.load('./logs/unclipped_log.npy')
avg_unclipped = avg(unclipped_scores)
idx = list(range(100, len(unclipped_scores)))

baseline_color = (120 / 255, 10 /255, 153 / 255, 0.5)

plt.figure(num = None, figsize = (10, 6), dpi =80)
plt.title('Reward Over Time - {} Episodes'.format(len(unclipped_scores)))

plt.plot(unclipped_scores, color = (255 / 255, 99 / 255, 20 / 255, 0.35), label = 'Reward')
plt.plot(idx, avg_unclipped, color = (255 / 255, 99 / 255, 20 / 255, 1), label = 'Average reward over 100 episodes')
plt.plot(idx, [30 for _ in idx], color = (20 / 255, 83 / 255, 255 / 255, 0.5), label = 'Baseline (average reward = 1)')

plt.legend()
plt.savefig('reward.png')
