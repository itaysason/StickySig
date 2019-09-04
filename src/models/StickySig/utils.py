import numpy as np


def expectation_step_sample(sequences, strand_infos, weights, log_pi, log_e, log_alpha, log_1m_alpha):
    m, k = log_e.shape
    r = log_alpha.shape[0]
    expected_exposures = np.zeros(k)
    expected_stickiness = np.zeros((r, k))
    expected_non_stickiness = np.zeros((r, k))
    expected_emissions = np.zeros((m, k))
    log_likelihood = 0
    for seq, strand, weight in zip(sequences, strand_infos, weights):
        log_exposures, log_stickiness, log_non_stickiness, log_pr_xt_qi, log_prob =\
            forward_backward(seq, strand, log_pi, log_e, log_alpha, log_1m_alpha)
        expected_exposures += np.exp(log_exposures) * weight
        expected_stickiness += np.exp(log_stickiness) * weight
        expected_non_stickiness += np.exp(log_non_stickiness) * weight
        log_likelihood += weight * log_prob

        pr_xt_qi = np.exp(log_pr_xt_qi) * weight
        for i in range(len(seq)):
            expected_emissions[seq[i]] += pr_xt_qi[i]
    return expected_exposures, expected_stickiness, expected_non_stickiness, expected_emissions, log_likelihood


def forward_backward(seq, strand, log_pi, log_e, log_alpha, log_1m_alpha):
    m, k = log_e.shape
    r = log_alpha.shape[0]

    log_fwd = np.empty((len(seq), k))
    log_bwd = np.empty((len(seq), k))

    log_fwd[0] = log_pi + log_e[seq[0]]
    log_bwd[-1] = 0

    for i in range(1, len(seq)):
        tmp = logsumexp(log_fwd[i - 1] + log_1m_alpha[strand[i]])
        log_fwd[i] = log_e[seq[i]] + np.logaddexp(log_pi + tmp, log_fwd[i - 1] + log_alpha[strand[i]])

        tmp = logsumexp(log_bwd[-i] + log_e[seq[-i]] + log_pi)
        np.logaddexp(log_1m_alpha[strand[-i]] + tmp,
                     log_alpha[strand[-i]] + log_e[seq[-i]] + log_bwd[-i], log_bwd[- i - 1])

    # compute a, b, c and score
    log_prob = logsumexp(log_fwd[-1])
    log_stickiness = np.log(np.zeros((r, k)))
    log_non_stickiness = np.log(np.zeros((r, k)))
    log_exposures = log_fwd[0] + log_bwd[0] - log_pi
    for i in range(len(seq) - 1):
        curr_log_emissions = log_e[seq[i + 1]]
        np.logaddexp(log_fwd[i] + curr_log_emissions + log_bwd[i + 1], log_stickiness[strand[i]],
                     log_stickiness[strand[i]])
        np.logaddexp(logsumexp(log_fwd[i] + log_1m_alpha) + curr_log_emissions + log_bwd[i + 1],
                     log_exposures, log_exposures)
        np.logaddexp(log_fwd[i] + logsumexp(log_pi + curr_log_emissions + log_bwd[i + 1]),
                     log_non_stickiness[strand[i]], log_non_stickiness[strand[i]])

    log_stickiness += log_alpha - log_prob
    log_exposures += log_pi - log_prob
    log_non_stickiness += log_1m_alpha - log_prob

    # pr_xt_qi = Pr[x_t = q_i | seq]
    log_pr_xt_qi = log_fwd + log_bwd - log_prob

    return log_exposures, log_stickiness, log_non_stickiness, log_pr_xt_qi, log_prob


def predict(seq, strand, log_pi, log_e, log_alpha, log_1m_alpha, algorithm='viterbi'):
    if algorithm == 'viterbi':
        return predict_viterbi(seq, strand, log_pi, log_e, log_alpha, log_1m_alpha)
    elif algorithm == 'map':
        return predict_map(seq, strand, log_pi, log_e, log_alpha, log_1m_alpha)
    else:
        raise ValueError('algorithm can only be viterbi or map')


def predict_viterbi(seq, strand, log_pi, log_e, log_alpha, log_1m_alpha):
    m, k = log_e.shape
    arange_k = np.arange(k)
    T = len(seq)

    log_viterbi = np.empty((T, 2 * k))
    ptr = np.empty((T, 2 * k), dtype='int')

    log_viterbi[0, :k] = log_pi + log_e[seq[0]]
    log_viterbi[0, k:] = np.log(0)

    for i in range(1, T):
        larger_idx = np.argmax([log_viterbi[i-1, :k], log_viterbi[i-1, k:]], axis=0)
        larger_val = np.max([log_viterbi[i-1, :k], log_viterbi[i-1, k:]], axis=0)
        tmp = larger_val + log_1m_alpha[strand[i]]
        top_k = np.argmax(tmp)
        top_val = tmp[top_k]

        log_viterbi[i, :k] = log_e[seq[i]] + log_pi + top_val
        log_viterbi[i, k:] = log_e[seq[i]] + log_alpha[strand[i]] + larger_val

        ptr[i, :k] = larger_idx[top_k] * k + top_k
        ptr[i, k:] = arange_k + larger_idx * k

    most_probable_end_state = np.argmax(log_viterbi[-1])
    log_likelihood = log_viterbi[-1, most_probable_end_state]

    # backtracking
    signatures_sequence = np.zeros(T, dtype='int')
    signatures_sequence[-1] = most_probable_end_state

    for i in range(1, T):
        signatures_sequence[-i-1] = ptr[-i, signatures_sequence[-i]]

    stickiness_sequence = signatures_sequence // k
    signatures_sequence = signatures_sequence % k
    return signatures_sequence, stickiness_sequence, log_likelihood


def debug_predict_viterbi(seq, strand, log_pi, log_e, log_alpha, log_1m_alpha):
    signatures_sequence, stickiness_sequence, log_likelihood = predict_viterbi(seq, strand, log_pi, log_e, log_alpha, log_1m_alpha)
    T = len(seq)
    test_ll = 0
    test_ll += log_pi[signatures_sequence[0]] + log_e[seq[0], signatures_sequence[0]]

    for i in range(1, T):
        if stickiness_sequence[i] == 0:
            test_ll += log_1m_alpha[strand[i], signatures_sequence[i-1]] + log_pi[signatures_sequence[i]]
        elif stickiness_sequence[i] == 1:
            if signatures_sequence[i] != signatures_sequence[i-1]:
                print('stickiness is 1 but the signatures are not the same')
                raise ValueError
            test_ll += log_alpha[strand[i], signatures_sequence[i - 1]]
        else:
            print('stickiness is  not 0 or 1')
            raise ValueError
        test_ll += log_e[seq[i], signatures_sequence[i]]

    print(test_ll - log_likelihood)


def predict_map(seq, strand, log_pi, log_e, log_alpha, log_1m_alpha):
    m, k = log_e.shape
    r = log_alpha.shape[0]

    log_fwd = np.empty((len(seq), 2 * k))
    log_bwd = np.empty((len(seq), 2 * k))

    log_fwd[0] = log_pi + log_e[seq[0]]
    log_bwd[-1] = 0

    for i in range(1, len(seq)):
        tmp = logsumexp(log_fwd[i - 1] + log_1m_alpha[strand[i]])
        log_fwd[i] = log_e[seq[i]] + np.logaddexp(log_pi + tmp, log_fwd[i - 1] + log_alpha[strand[i]])

        tmp = logsumexp(log_bwd[-i] + log_e[seq[-i]] + log_pi)
        np.logaddexp(log_1m_alpha[strand[-i]] + tmp,
                     log_alpha[strand[-i]] + log_e[seq[-i]] + log_bwd[-i], log_bwd[- i - 1])

    log_prob = logsumexp(log_fwd[-1])
    log_pr_xt_qi = log_fwd + log_bwd - log_prob
    return


def logsumexp(a):
    """
    To reduce running time I implemented logsumexp myself, the scipy version has too much additional things I don't need
    :param a:
    :return:
    """
    a_max = np.max(a)
    return a_max + np.log(np.sum(np.exp(a - a_max)))
