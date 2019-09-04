import numpy as np
from src.models.StickySig.utils import expectation_step_sample, predict
import time


np.warnings.filterwarnings('ignore')


class StickySig:
    def __init__(self, k):
        """
        k - number of topics in the model
        m - number of words
        r - number of stickiness parameters for each signature
        pi - dict dict with samples as keys, the exposure of each sample to the topics. pi[sample] will be the exposure
            of the sample.
        e - matrix kxm, topic distribution over words. e[i] is the i'th topic
        alpha - matrix rxk, topic stickiness. alpha[i, j] the i'th stickiness type of signature j.
        :param k:
        """
        self.k = k
        self.m, self.r = None, None
        self.pi = {}
        self.e = None
        self.alpha = None

    def process_data(self, data):
        processed_data = {}
        m = 0
        r = -1
        for name, curr_data in data.items():
            if name not in self.pi:
                if 'initial_pi' in curr_data:
                    if len(np.array(curr_data['initial_pi']).shape) == 1:
                        if len(np.array(curr_data['initial_pi'])) == self.k:
                            self.pi[name] = np.array(curr_data['initial_pi'])
                        else:
                            raise ValueError('Given initial_pi for sample {} but with wrong format: {}'.format(
                                name, curr_data['initial_pi']))
                    else:
                        raise ValueError('Given initial_pi for sample {} but with wrong format: {}'.format(
                            name, curr_data['initial_pi']))
                else:
                    self.pi[name] = np.random.dirichlet([0.5] * self.k)
            curr_data = data[name]
            curr_sequences = []
            sample_strand_info = []

            for k in curr_data.keys():
                seq = curr_data[k]['Sequence']
                if len(seq) == 0:
                    continue
                strand_info = np.array(curr_data[k]['StrandInfo']) - 1
                strand_info[0] = -1
                m = max(m, max(seq))
                r = max(r, max(strand_info))
                splits = np.where(strand_info == -1)[0]
                if 0 in splits:
                    splits = splits[1:]
                sample_strand_info.extend(np.split(strand_info, splits))
                curr_sequences.extend(np.split(seq, splits))

            # counting sequences with repetitions
            curr_unique_sequences = []
            curr_unique_strand_info = []
            curr_count_sequences = []
            for seq1, strand_info1 in zip(curr_sequences, sample_strand_info):
                flag = True
                for i, (seq2, strand_info2) in enumerate(zip(curr_unique_sequences, curr_unique_strand_info)):
                    # this is to reduce time, if it is not true than the sequences aren't the same for sure
                    if len(seq1) == len(seq2) and seq1[0] == seq2[0] and seq1[-1] == seq2[-1]:
                        if len(strand_info1) == len(strand_info2) and strand_info1[-1] == strand_info2[-1]:
                            if np.all(seq1 == seq2):
                                if np.all(strand_info1[1:] == strand_info2[1:]):
                                    flag = False
                                    curr_count_sequences[i] += 1
                                    break

                if flag:
                    curr_unique_sequences.append(seq1)
                    curr_unique_strand_info.append(strand_info1)
                    curr_count_sequences.append(1)

            processed_data[name] = {'sequences': curr_unique_sequences, 'weights': curr_count_sequences,
                                    'StrandInfo': curr_unique_strand_info}

        m = int(m + 1)
        r = int(r + 1)
        if self.m is None:
            self.m = m
        if self.r is None:
            self.r = r

        if self.m < m:
            raise ValueError('Input has more words than the model')
        if self.r < r:
            raise ValueError('Input has more sticky types than the model')

        return processed_data

    def expectation_step(self, data):
        """
        Compute the Expectation step using forward/backward algorithm.
        :param data: array of arrays of ints
        :return:
        """
        k, m, r = self.k, self.m, self.r
        log_e = np.log(self.e.T)
        log_alpha = np.log(self.alpha)
        log_1m_alpha = np.log(1 - self.alpha)
        samples_log_pi = {sample: np.log(pi) for sample, pi in self.pi.items()}

        expected_exposures = {}
        expected_stickiness = np.zeros((r, k))
        expected_non_stickiness = np.zeros((r, k))
        expected_emissions = np.zeros((m, k))
        log_likelihood = {}
        for sample, sample_data in data.items():
            sequences = sample_data['sequences']
            strand_infos = sample_data['StrandInfo']
            weights = sample_data['weights']
            log_pi = samples_log_pi[sample]

            curr_exposures, curr_stickiness, curr_non_stickiness, curr_emissions, curr_log_likelihood =\
                expectation_step_sample(sequences, strand_infos, weights, log_pi, log_e, log_alpha, log_1m_alpha)

            expected_exposures[sample] = curr_exposures
            expected_stickiness += curr_stickiness
            expected_non_stickiness += curr_non_stickiness
            expected_emissions += curr_emissions
            log_likelihood[sample] = curr_log_likelihood

        expected_emissions = expected_emissions.T
        return expected_exposures, expected_stickiness, expected_non_stickiness, expected_emissions, log_likelihood

    def maximization_step(self, expected_exposures, expected_stickiness, expected_non_stickiness, expected_emissions, learn_params):
        if 'pi' in learn_params:
            for s, v in expected_exposures.items():
                self.pi[s] = v / v.sum()
        if 'alpha' in learn_params:
            if self.r != 0:
                self.alpha = expected_stickiness / (expected_stickiness + expected_non_stickiness)
        if 'e' in learn_params:
            self.e = expected_emissions / expected_emissions.sum(1, keepdims=True)

    def fit(self, data, learn_params=None, epsilon=1e-10, max_iterations=1000, print_progress=True):
        processed_data = self.process_data(data)
        learn_params = ['pi', 'e', 'alpha'] if learn_params is None else learn_params
        self.e = np.random.dirichlet([0.5] * self.m, self.k) if self.e is None else self.e
        # alpha is a uniform number between 0.2 and 0.4
        self.alpha = 0.2 * np.random.random((self.r, self.k)) + 0.2 if self.alpha is None else self.alpha

        expected_exposures, expected_stickiness, expected_non_stickiness, expected_emissions, log_likelihood =\
            self.expectation_step(processed_data)
        prev_score = 0
        for s in log_likelihood.values():
            prev_score += s

        for iteration in range(max_iterations):
            tic1 = time.clock()
            self.maximization_step(expected_exposures, expected_stickiness, expected_non_stickiness,
                                   expected_emissions, learn_params)
            expected_exposures, expected_stickiness, expected_non_stickiness, expected_emissions, log_likelihood = \
                self.expectation_step(processed_data)
            score = 0
            for s in log_likelihood.values():
                score += s
            tic2 = time.clock()
            relative_improvement = (score - prev_score) / (-score)
            if print_progress:
                print('iteration {}, score {}, relative improvement {}, time {}'.format(
                    iteration, score, relative_improvement, tic2 - tic1))
            if relative_improvement < epsilon:
                break
            prev_score = score

        return log_likelihood

    def refit(self, data, epsilon=1e-10, max_iterations=1000):
        log_likelihood = {}
        for i, (sample, sample_data) in enumerate(data.items()):
            sample_ll = self.fit({sample: sample_data}, ['pi'], epsilon, max_iterations, print_progress=False)
            print('refitted sample {} out of {}'.format(i, len(data)))
            log_likelihood[sample] = sample_ll[sample]
        return log_likelihood

    def log_probability(self, data):
        for sample, sample_data in data.items():
            if sample not in self.pi:
                raise ValueError('sample {} was never seen before.')
        processed_data = self.process_data(data)
        _, _, _, _, log_likelihood = self.expectation_step(processed_data)
        return log_likelihood

    def predict(self, data, algorithm='viterbi'):
        prediction_dict = {}
        log_e = np.log(self.e.T)
        alpha = np.zeros((self.r + 1, self.k))
        alpha[1:] = self.alpha
        log_alpha = np.log(alpha)
        log_1m_alpha = np.log(1 - alpha)
        for sample, sample_data in data.items():
            sample_predictions = {}
            if sample not in self.pi:
                raise ValueError('sample {} was never seen before.')
            log_pi = np.log(self.pi[sample])
            for chromosome, chromosome_data in data[sample].items():
                seq = chromosome_data['Sequence']
                strand = chromosome_data['StrandInfo']
                if len(seq) > 0:
                    sigs, sticks, ll = predict(seq, strand, log_pi, log_e, log_alpha, log_1m_alpha, algorithm)
                else:
                    sigs, sticks, ll = [], [], 0
                sample_predictions[chromosome] = {}
                sample_predictions[chromosome]['signatures'] = sigs
                sample_predictions[chromosome]['stickiness'] = sticks
                sample_predictions[chromosome]['log-likelihood'] = ll
            prediction_dict[sample] = sample_predictions
        return prediction_dict

    def get_params(self):
        return {'pi': self.pi.copy(), 'alpha': self.alpha.copy(), 'e': self.e.copy()}
