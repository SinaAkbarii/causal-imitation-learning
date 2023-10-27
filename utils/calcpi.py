import numpy as np


def calc_pi_oracle(scm):
    # oracles:
    dist_s = scm.s_dist()
    pi_naive1 = [scm.x_dist(), scm.x_dist()]
    pi_naive2 = scm.x_given_c()
    p1_opt = (scm.pu[1] + dist_s[1] - 1) / (2 * scm.pu[1] - 1)
    pi_opt = [[1 - p1_opt, p1_opt], [1 - p1_opt, p1_opt]]
    return pi_naive1, pi_naive2, pi_opt


def calc_pi_data(data):
    # data-based:
    num_s = len([d for d in data if d[3] == 1])  # number of samples with s=1
    dist_s_data = np.array([len(data) - num_s, num_s]) / len(data)

    num1 = 0
    denom1 = 0
    num2 = 0
    denom2 = 0
    for _, c, x, s, _ in data:
        if c == 0:
            if x == 0:
                denom1 += 1
                if s == 1:
                    num1 += 1
            else:
                denom2 += 1
                if s == 1:
                    num2 += 1

    p_s_given00 = num1 / denom1
    p_s_given10 = num2 / denom2
    p_opt_data = (dist_s_data[1] - p_s_given00) / (p_s_given10 - p_s_given00)
    pi_opt_data = [[1 - p_opt_data, p_opt_data], [1 - p_opt_data, p_opt_data]]

    num1 = 0
    denom1 = 0
    num2 = 0
    denom2 = 0
    for _, c, x, _, _ in data:
        if c == 0:
            denom1 += 1
            if x == 1:
                num1 += 1
        else:
            denom2 += 1
            if x == 1:
                num2 += 1
    p_x_given0 = num1 / denom1
    p_x_given1 = num2 / denom2
    p_x = (num1 + num2) / len(data)
    pi_naive1_data = [[1 - p_x, p_x], [1 - p_x, p_x]]
    pi_naive2_data = [[1 - p_x_given0, p_x_given0], [1 - p_x_given1, p_x_given1]]

    return pi_naive1_data, pi_naive2_data, pi_opt_data


def calc_pi_data2(data):
    # data-based:
    num_s = len([d for d in data if d[3] == 1])  # number of samples with s=1
    dist_s_data = np.array([len(data) - num_s, num_s]) / len(data)

    num1 = 0
    denom1 = 0
    num2 = 0
    denom2 = 0
    for _, c, x, s, _, _ in data:
        if c == 0:
            if x == 0:
                denom1 += 1
                if s == 1:
                    num1 += 1
            else:
                denom2 += 1
                if s == 1:
                    num2 += 1

    p_s_given00 = num1 / denom1
    p_s_given10 = num2 / denom2
    p_opt_data = (dist_s_data[1] - p_s_given00) / (p_s_given10 - p_s_given00)
    pi_opt_data = [[1 - p_opt_data, p_opt_data], [1 - p_opt_data, p_opt_data]]

    num1 = 0
    denom1 = 0
    num2 = 0
    denom2 = 0
    for _, c, x, _, _, _ in data:
        if c == 0:
            denom1 += 1
            if x == 1:
                num1 += 1
        else:
            denom2 += 1
            if x == 1:
                num2 += 1
    p_x_given0 = num1 / denom1
    p_x_given1 = num2 / denom2
    p_x = (num1 + num2) / len(data)
    pi_naive1_data = [[1 - p_x, p_x], [1 - p_x, p_x]]
    pi_naive2_data = [[1 - p_x_given0, p_x_given0], [1 - p_x_given1, p_x_given1]]

    return pi_naive1_data, pi_naive2_data, pi_opt_data