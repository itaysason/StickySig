import os
from src.utils import load_json
import numpy as np
import matplotlib.pyplot as plt


def analyze_loco_socres():
    print('Analyzing LOCO')
    d = 'experiments/LOCO/ICGC-BRCA'
    for dir_p in os.listdir(d):
        dir_path = os.path.join(d, dir_p)
        s = 0
        for k in os.listdir(dir_path):
            curr_file = load_json(os.path.join(dir_path, k))['scores']
            if 'same-allele-same-strand' in curr_file.keys():
                curr_file = curr_file['same-allele-same-strand']
            for sample in curr_file:
                s += curr_file[sample]

        print(dir_p, len(os.listdir(dir_path)), s)


def analyze_sample_cv_scores():
    print('\nAnalyzing sampleCV')
    d = 'experiments/sampleCV/ICGC-BRCA'
    for dir_p in os.listdir(d):
        dir_path = os.path.join(d, dir_p)
        s = 0
        for k in os.listdir(dir_path):
            curr_file = load_json(os.path.join(dir_path, k))['scores']['testScore']
            for sample in curr_file:
                s += curr_file[sample]

        print(dir_p, len(os.listdir(dir_path)), s)


def get_stickiness_loco():
    print('\nAnalyzing sampleCV')
    d = 'experiments/sampleCV/ICGC-BRCA'
    sig_names = ['sig' + str(i) for i in [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]]
    for dir_p in os.listdir(d):
        if 'cosmic' not in dir_p:
            continue
        if dir_p == 'MM_cosmic' or dir_p == 'MM':
            continue
        dir_path = os.path.join(d, dir_p)
        num_runs = len(os.listdir(dir_path))
        tmp_alpha = np.array(load_json(os.path.join(dir_path, os.listdir(dir_path)[0]))['parameters']['alpha'])
        num_stick, num_sig = tmp_alpha.shape
        all_alpha = np.zeros((num_stick, num_sig, num_runs))
        for i, k in enumerate(os.listdir(dir_path)):
            all_alpha[:, :, i] = load_json(os.path.join(dir_path, k))['parameters']['alpha']

        for stick in range(num_stick):
            plt.boxplot(all_alpha[stick].T)
            plt.title('{}, {}'.format(dir_p.split('_')[0], stick + 1))
            plt.show()


def analyze_number_of_signatures_sample_cv():
    print('\nAnalyzing StickySig1_sampleCV')
    d = 'experiments/hyper_parameter_selection/sampleCV/ICGC-BRCA'
    scores = []
    names = []
    for dir_p in os.listdir(d):
        dir_path = os.path.join(d, dir_p)
        s = 0
        for k in os.listdir(dir_path):
            curr_file = load_json(os.path.join(dir_path, k))['scores']['testScore']
            for sample in curr_file:
                s += curr_file[sample]
        scores.append(s)
        names.append(int(dir_p))
        print(dir_p, len(os.listdir(dir_path)), s)

    scores = np.array(scores)
    names = np.array(names)
    i = np.argsort(names)
    names = names[i]
    scores = scores[i]
    plt.plot(names, scores)
    plt.show()


def BIC():
    print('\nplot BIC')
    d = 'experiments/num_signatures/ICGC-BRCA/StickySig-same-allele'
    scores = []
    names = []
    for dir_p in os.listdir(d):
        dir_path = os.path.join(d, dir_p)
        max_score = np.log(0)
        for k in os.listdir(dir_path):
            curr_file = load_json(os.path.join(dir_path, k))['scores']
            curr_score = 0
            for sample in curr_file:
                curr_score += curr_file[sample]
            if max_score < curr_score:
                max_score = curr_score
        curr_k = int(dir_p)
        names.append(curr_k)
        penalty = np.log(3479652) * (656 * curr_k - 560)
        bic = penalty - 2 * max_score
        scores.append(bic)

        print(dir_p, len(os.listdir(dir_path)), penalty)

    scores = np.array(scores)
    names = np.array(names)
    i = np.argsort(names)
    names = names[i]
    scores = scores[i]
    plt.plot(names, scores)
    plt.show()


def stickiness():
    print('\nplot BIC')
    d = 'experiments/trained_models/ICGC-BRCA/StickySig-same-allele_cosmic'
    sig_names = ['sig' + str(i) for i in [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]]
    num_runs = len(os.listdir(d))
    num_sigs = len(sig_names)
    all_alphas = np.zeros((num_runs, num_sigs))
    all_scores = np.zeros(num_runs)
    for i, dir_p in enumerate(os.listdir(d)):
        dir_path = os.path.join(d, dir_p)
        all_alphas[i] = np.array(load_json(dir_path)['parameters']['alpha'])[0]
        q = load_json(dir_path)['scores']
        for score in q.values():
            all_scores[i] += score

    plt.boxplot(all_alphas)
    plt.show()


def boxplot_scores():
    print('\nplot BIC boxplot')
    d = 'experiments/num_signatures/ICGC-BRCA/StickySig-same-allele'
    scores = []
    names = []
    for dir_p in os.listdir(d):
        dir_path = os.path.join(d, dir_p)
        curr_scores = []
        for k in os.listdir(dir_path):
            curr_file = load_json(os.path.join(dir_path, k))['scores']
            curr_score = 0
            for sample in curr_file:
                curr_score += curr_file[sample]
            curr_scores.append(curr_score)
        curr_k = int(dir_p)
        names.append(curr_k)
        penalty = np.log(3479652) * (656 * curr_k - 560)
        bic = penalty - 2 * np.array(curr_scores)
        scores.append(bic)
        print(dir_p, len(os.listdir(dir_path)), penalty)

    scores = np.array(scores)
    names = np.array(names)
    i = np.argsort(names)
    names = names[i]
    scores = scores[i]
    plt.boxplot(scores.T)
    plt.show()
    return


def arange_corr_matrix(corr_mat):
    num_sigs = len(corr_mat)
    corr_mat_tmp = corr_mat.copy()

    for i in range(num_sigs):
        corr_mat_tmp[i:, i] = -1

    all_groups = []
    group_scores = []
    num_sigs_in_groups = 0
    num_conflicts = 0
    while num_sigs_in_groups < num_sigs:
        x, y = np.unravel_index(np.argmax(corr_mat_tmp), corr_mat_tmp.shape)
        corr_mat_tmp[x, y] = -1
        x_group = -1
        y_group = -1
        for group_num, g in enumerate(all_groups):
            if x in g:
                x_group = group_num
            if y in g:
                y_group = group_num
            if x_group >= 0 and y_group >= 0:
                break
        if x_group == y_group and x_group >= 0:
            group_scores[x_group] += 1
        elif x_group >= 0 and y_group == -1:
            all_groups[x_group].append(y)
            group_scores[x_group] += 1
            num_sigs_in_groups += 1
        elif y_group >= 0 and x_group == -1:
            all_groups[y_group].append(x)
            group_scores[y_group] += 1
            num_sigs_in_groups += 1
        elif x_group == -1 and y_group == -1:
            all_groups.append([x, y])
            group_scores.append(1)
            num_sigs_in_groups += 2
        else:
            f = True
            for x1 in all_groups[x_group]:
                for x2 in all_groups[y_group]:
                    if corr_mat[x1, x2] <= 0.6:
                        f = False
                        break
            if f:
                all_groups[x_group].extend(all_groups[y_group])
                all_groups = [all_groups[i] for i in range(len(all_groups)) if i != y_group]
                group_scores[x_group] + group_scores[y_group]
                group_scores = [group_scores[i] for i in range(len(group_scores)) if i != y_group]
            else:
                num_conflicts += 1

    sigs_arangment = []
    for i in np.argsort(-np.array(group_scores)):
        sigs_arangment.extend(all_groups[i])
    return sigs_arangment


def arange_by_corr(points):
    points = [np.array(p) for p in points]
    t = [i for i in range(len(points))]
    while len(points) > 1:
        corr_mat = np.corrcoef(points)
        for i in range(len(corr_mat)):
            corr_mat[i:, i] = -1

        x, y = np.unravel_index(np.argmax(corr_mat), corr_mat.shape)
        points[x] = (points[x] + points.pop(y)) / 2
        t[x] = [t[x], t.pop(y)]

    def flatten_t(curr_t, semi_flattened_t):
        for r in curr_t:
            if isinstance(r, int):
                semi_flattened_t.append(r)
            else:
                semi_flattened_t = flatten_t(r, semi_flattened_t)
        return semi_flattened_t

    flattened_t = []
    return flatten_t(t, flattened_t)


def arange_by_corr2(corr_mat):
    t = [i for i in range(len(corr_mat))]
    corr_mat = corr_mat.copy()

    def flatten_t(curr_t, semi_flattened_t):
        for r in curr_t:
            if isinstance(r, int):
                semi_flattened_t.append(r)
            else:
                semi_flattened_t = flatten_t(r, semi_flattened_t)
        return semi_flattened_t

    while len(t) > 1:
        tmp = corr_mat.copy()
        for i in range(len(tmp)):
            tmp[i:, i] = -1

        x, y = np.unravel_index(np.argmax(tmp), tmp.shape)
        corr_mat[x] = (corr_mat[x] + corr_mat[y]) / 2
        corr_mat[:, x] = (corr_mat[:, x] + corr_mat[:, y]) / 2
        corr_mat = np.delete(corr_mat, y, axis=0)
        corr_mat = np.delete(corr_mat, y, axis=1)
        t[x] = [t[x], t.pop(y)]

    flattened_t = []
    return flatten_t(t, flattened_t)


def plot_train_correlations(d):
    all_sigs = []
    all_alphas = []
    for k in os.listdir(d):
        f = load_json(os.path.join(d, k))['parameters']
        all_alphas.extend(f['alpha'])
        all_sigs.extend(f['e'])

    print(np.count_nonzero(np.array(all_alphas) < 0.99))
    all_sigs = np.array(all_sigs)
    matrix_arangment = arange_by_corr(all_sigs)
    corr_mat = np.corrcoef(all_sigs[matrix_arangment])
    plt.imshow(corr_mat)
    plt.show()

    corr_mat = np.corrcoef(all_sigs)
    matrix_arangment = arange_corr_matrix(corr_mat)
    corr_mat = np.corrcoef(all_sigs[matrix_arangment])
    plt.imshow(corr_mat)
    plt.show()

    corr_mat = np.corrcoef(all_sigs)
    matrix_arangment = arange_by_corr2(corr_mat)
    corr_mat = np.corrcoef(all_sigs[matrix_arangment])
    plt.imshow(corr_mat)
    plt.show()


# get_stickiness_loco()
# analyze_loco_socres()
# analyze_sample_cv_scores()
# analyze_number_of_signatures_sample_cv()
# stickiness()

plot_train_correlations('experiments/trained_models/ICGC-BRCA/StickySig-same-allele/11')
plot_train_correlations('experiments/trained_models/ICGC-BRCA/StickySig-same-allele/12')
