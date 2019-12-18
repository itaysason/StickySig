from src.utils import save_json, get_cosmic_signatures, load_json
from src.experiments import split_train_test_loco, split_train_test_sample_cv, train_stickysig, train_test_stickysig,\
    get_data_by_model_name, predict_hidden_variables, prepare_data_to_json
import click
import time
import os


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(chain=False, context_settings=CONTEXT_SETTINGS)
@click.option('--debug/--no-debug', default=False, help="Default is --no-debug.")
@click.option('-v', '--verbosity', default=0, help='Verbosity level (0-3).')
def simple_cli(debug, verbosity):
    return


@simple_cli.command('train_model', short_help='Train and save model')
@click.option('--dataset', type=str)
@click.option('--model_name', type=str)
@click.option('--use_cosmic', type=int)
@click.option('--num_signatures', type=int, default=0)
@click.option('--random_seed', type=int, default=0)
@click.option('--max_iterations', type=int, default=400)
@click.option('--epsilon', type=float, default=1e-15)
@click.option('--out_dir', type=str, default='experiments/trained_models')
def train_model(dataset, model_name, use_cosmic, num_signatures, random_seed, max_iterations, epsilon, out_dir):
    use_cosmic_dir = 'refit' if use_cosmic else 'denovo'
    dataset_name = dataset
    dataset, active_signatures = get_data_by_model_name(dataset, model_name)
    if use_cosmic:
        num_signatures = len(active_signatures)
        signatures = get_cosmic_signatures()[active_signatures]
    elif num_signatures == 0:
        print('use_cosmic is False and num_signatures is 0, using number of active cosmic signatures {}'.format(len(active_signatures)))
        num_signatures = len(active_signatures)
        signatures = None
    else:
        signatures = None

    out_dir = os.path.join(out_dir, dataset_name, use_cosmic_dir, model_name, str(num_signatures))

    try:
        os.makedirs(out_dir)
    except OSError:
        pass

    random_seed = int(time.time()) if random_seed == 0 else random_seed
    out_file = out_dir + "/" + str(random_seed)
    if os.path.isfile(out_file + '.json'):
        print('Experiment with parameters {} {} {} {} {} already exist'.format(
            dataset_name, model_name, use_cosmic, num_signatures, random_seed))
        return

    model, ll = train_stickysig(dataset, num_signatures, signatures, random_seed, epsilon, max_iterations)
    parameters = model.get_params()

    parameters['alpha'] = parameters['alpha'].tolist()
    parameters['e'] = parameters['e'].tolist()
    for sample in parameters['pi']:
        parameters['pi'][sample] = parameters['pi'][sample].tolist()

    out = {'log-likelihood': ll, 'parameters': parameters}
    save_json(out_file, out)


@simple_cli.command('LOCO', short_help='Perform leave one chromosome out (LOCO)')
@click.option('--dataset', type=str)
@click.option('--model_name', type=str)
@click.option('--chromosome', type=int)
@click.option('--use_cosmic', type=int)
@click.option('--num_signatures', type=int, default=0)
@click.option('--random_seed', type=int, default=0)
@click.option('--max_iterations', type=int, default=100)
@click.option('--epsilon', type=float, default=1e-10)
@click.option('--out_dir', type=str, default='experiments/LOCO')
def leave_one_chromosome_out(dataset, model_name, chromosome, use_cosmic, num_signatures, random_seed, max_iterations, epsilon, out_dir):
    use_cosmic_dir = 'refit' if use_cosmic else 'denovo'

    all_chromosomes = [str(i) for i in range(1, 23)]
    all_chromosomes.extend(['X', 'Y'])
    chromosome_name = all_chromosomes[chromosome]

    dataset_name = dataset
    dataset, active_signatures = get_data_by_model_name(dataset, model_name)
    if use_cosmic:
        num_signatures = len(active_signatures)
        signatures = get_cosmic_signatures()[active_signatures]
    elif num_signatures == 0:
        print('use_cosmic is False and num_signatures is 0, using number of active cosmic signatures {}'.format(len(active_signatures)))
        num_signatures = len(active_signatures)
        signatures = None
    else:
        signatures = None

    out_dir = os.path.join(out_dir, dataset_name, use_cosmic_dir, model_name, str(num_signatures), chromosome_name)

    try:
        os.makedirs(out_dir)
    except OSError:
        pass

    random_seed = int(time.time()) if random_seed == 0 else random_seed
    out_file = out_dir + "/" + str(random_seed)
    if os.path.isfile(out_file + '.json'):
        print('Experiment with parameters {} {} {} {} {} {} already exist'.format(
            dataset_name, model_name, chromosome, use_cosmic, num_signatures, random_seed))
        return

    train_data, test_data = split_train_test_loco(dataset, chromosome)

    model, train_ll, test_ll = train_test_stickysig(train_data, test_data, num_signatures, signatures, random_seed, epsilon, max_iterations)
    parameters = model.get_params()

    parameters['alpha'] = parameters['alpha'].tolist()
    parameters['e'] = parameters['e'].tolist()
    for sample in parameters['pi']:
        parameters['pi'][sample] = parameters['pi'][sample].tolist()

    out = {'log-likelihood-train': train_ll, 'log-likelihood-test': test_ll, 'parameters': parameters}
    save_json(out_file, out)


@simple_cli.command('sampleCV', short_help='Cross validate over samples')
@click.option('--dataset', type=str)
@click.option('--model_name', type=str)
@click.option('--num_folds', type=int)
@click.option('--fold', type=int)
@click.option('--use_cosmic', type=int)
@click.option('--num_signatures', type=int, default=0)
@click.option('--shuffle_seed', type=int, default=0)
@click.option('--random_seed', type=int, default=0)
@click.option('--max_iterations', type=int, default=100)
@click.option('--epsilon', type=float, default=1e-10)
@click.option('--out_dir', type=str, default='experiments/sampleCV')
def sample_cv(dataset, model_name, num_folds, fold, use_cosmic, num_signatures, shuffle_seed, random_seed, max_iterations, epsilon, out_dir):

    if fold >= num_folds:
        raise ValueError('num_folds is {} but fold is {}'.format(num_folds, fold))

    dataset_name = dataset
    dataset, active_signatures = get_data_by_model_name(dataset, model_name)
    if use_cosmic:
        num_signatures = len(active_signatures)
        signatures = get_cosmic_signatures()[active_signatures]
    elif num_signatures == 0:
        print('use_cosmic is False and num_signatures is 0, using number of active cosmic signatures {}'.format(len(active_signatures)))
        num_signatures = len(active_signatures)
        signatures = None
    else:
        signatures = None

    use_cosmic_dir = 'refit' if use_cosmic else 'denovo'
    out_dir = os.path.join(out_dir, dataset_name, use_cosmic_dir, model_name, str(num_signatures), str(shuffle_seed), str(num_folds), str(fold))

    try:
        os.makedirs(out_dir)
    except OSError:
        pass

    random_seed = int(time.time()) if random_seed == 0 else random_seed
    out_file = out_dir + "/" + str(random_seed)
    if os.path.isfile(out_file + '.json'):
        print('Experiment with parameters {} {} {} {} {} {} {} {} already exist'.format(
            dataset_name, model_name, num_folds, fold, use_cosmic, num_signatures, shuffle_seed, random_seed))
        return

    train_data, test_data = split_train_test_sample_cv(dataset, num_folds, fold, shuffle_seed)

    model, train_ll, test_ll = train_test_stickysig(train_data, test_data, num_signatures, signatures, random_seed, epsilon, max_iterations)
    parameters = model.get_params()

    parameters['alpha'] = parameters['alpha'].tolist()
    parameters['e'] = parameters['e'].tolist()
    for sample in parameters['pi']:
        parameters['pi'][sample] = parameters['pi'][sample].tolist()

    out = {'log-likelihood-train': train_ll, 'log-likelihood-test': test_ll, 'parameters': parameters}
    save_json(out_file, out)


@simple_cli.command('prepare_prediction_dir', short_help='Predict hidden variables')
@click.option('--trained_models_dir', type=str, default='experiments/trained_models')
@click.option('--prediction_dir', type=str, default='experiments')
def prepare_prediction_dir(trained_models_dir, prediction_dir):
    datasets = os.listdir(trained_models_dir)
    prediction_dir = os.path.join(prediction_dir, 'prediction')
    for dataset in datasets:
        print(dataset)
        dataset_dir = os.path.join(trained_models_dir, dataset)
        for signature_learning in os.listdir(dataset_dir):
            for model in os.listdir(os.path.join(dataset_dir, signature_learning)):
                dataset_path = os.path.join(prediction_dir, dataset, signature_learning, model)
                try:
                    os.makedirs(dataset_path)
                except OSError:
                    pass
                data, _ = get_data_by_model_name(dataset, model)
                json_data = {}
                for sample, sample_data in data.items():
                    json_data[sample] = {}
                    for chrom, chrom_data in sample_data.items():
                        json_data[sample][chrom] = {}
                        json_data[sample][chrom]['Sequence'] = chrom_data['Sequence'].tolist()
                        json_data[sample][chrom]['StrandInfo'] = chrom_data['StrandInfo'].tolist()

                save_json(os.path.join(dataset_path, 'data'), json_data)
                del json_data
                for num_sigs in os.listdir(os.path.join(dataset_dir, signature_learning, model)):
                    num_sig_dir = os.path.join(dataset_path, num_sigs)
                    try:
                        os.makedirs(num_sig_dir)
                    except OSError:
                        pass
                    experiment_dir = os.path.join(dataset_dir, signature_learning, model, num_sigs)
                    runs = os.listdir(experiment_dir)
                    for run in runs:
                        model_parameters = load_json(os.path.join(experiment_dir, run))['parameters']
                        if not model_parameters['e'][0][0] >= 0:
                            print('There was a bug in run {}'.format(os.path.join(experiment_dir, run)))
                        prediction = predict_hidden_variables(data, model_parameters)
                        save_json(os.path.join(num_sig_dir, run), prepare_data_to_json(prediction))

        print('\n')


if __name__ == "__main__":
    simple_cli()
