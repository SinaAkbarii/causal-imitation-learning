from utils.scm import SCMFull
from utils.calcpi import calc_pi_oracle, calc_pi_data2
import numpy as np


"""
    Generate samples from the graph of Figure 3.a.
    Run this snippet to reproduce results of Table 1.
"""


if __name__ == '__main__':
    model = SCMFull()

    # oracle policies:
    pi1_naive, pi2_naive, pi_opt = calc_pi_oracle(model)

    expert_y = model.y_dist()  # expert distribution of reward
    # oracle policy reward distributions:
    y_naive1 = model.y_do_pi(pi1_naive)
    y_naive2 = model.y_do_pi(pi2_naive)
    y_opt = model.y_do_pi(pi_opt)

    sample_size = 1000
    data = model.gen_data(sample_size)

    pi1_naive_d, pi2_naive_d, pi_opt_d = calc_pi_data2(data)
    # estimated from data policy rewards:
    reward_naive1 = model.y_do_pi(pi1_naive_d)
    reward_naive2 = model.y_do_pi(pi2_naive_d)
    reward_opt = model.y_do_pi(pi_opt_d)

    def KL(p, q):
        kl = 0
        for x in range(len(p)):
            kl += p[x]*np.log(p[x]/q[x])
        return kl


    print("rewards:")
    print(f'expert:{expert_y}')
    print(f'naive1:{reward_naive1}')
    print(f'naive2:{reward_naive2}')
    print(f'alg2:{reward_opt}')

    print("Expected:")
    print(f'expert: {np.dot(expert_y, [0, 1, 2])}')
    print(f'naive1: {np.dot(reward_naive1, [0, 1, 2])}')
    print(f'naive2: {np.dot(reward_naive2, [0, 1, 2])}')
    print(f'alg2: {np.dot(reward_opt, [0, 1, 2])}')

    print("KL-Y:")
    print(f'naive1:{KL(expert_y,reward_naive1)}')
    print(f'naive2:{KL(expert_y, reward_naive2)}')
    print(f'alg2:{KL(expert_y, reward_opt)}')

    print("KL-pi:")
    print(f'naive1:{KL(pi1_naive[0], pi1_naive_d[0])}')
    print(f'naive2C0:{KL(pi2_naive[0], pi2_naive_d[0])}')
    print(f'naive2C1:{KL(pi2_naive[1], pi2_naive_d[1])}')
    print(f'alg2:{KL(pi_opt[0], pi_opt_d[0])}')
