import os, torch, uuid
from argparse import ArgumentParser
from src.experiment.unsupervised import UnsupervisedClassificationExperiment
from bindsnet.models import DiehlAndCook2015v2
from src.utils.common import max_without_indices, write_args_to_file
from src.utils.dataloaders import load_image_folder_dataloader, load_mnist_dataloader
from torch.utils.tensorboard import SummaryWriter
from src.utils.params import load_parameters

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--params_file", type=str, default='./parameters/mnist.yaml')
    args = parser.parse_args()

    ## ========== PARAMS & SEED ========== ##

    params = load_parameters(args.params_file)

    update_steps = max(250 // params["batch_size"], 1)
    experiment_id = str(uuid.uuid4())[:8]
    experiment_name = f'experiment_{experiment_id}' if not params['name'] else params['name']
    print('\n# Initialization of the experiment protocol {} - {} \n'.format(params['task'], experiment_name))
    log_dir = f"{os.path.join(params['log_dir'], params['task'], experiment_name)}"
    os.makedirs(log_dir, exist_ok=True)
    write_args_to_file(params, log_dir)

    gpu = torch.cuda.is_available()

    if gpu:
        torch.cuda.manual_seed_all(params['seed'])
    else:
        torch.manual_seed(params['seed'])

    device = 'cuda' if gpu else 'cpu'

    if params['reduction'] == "sum":
        reduction = torch.sum
    elif params['reduction'] == "mean":
        reduction = torch.mean
    elif params['reduction'] == "max":
        reduction = max_without_indices
    else:
        raise NotImplementedError

    ## ========== DATASET ========== ##

    if params['task'] == 'MNIST':
        train_dataloader, test_dataloader = load_mnist_dataloader(
            params['data_dir'],
            params['input_size'],
            params["batch_size"],
            params['time'],
            params['dt'],
            params['intensity'],
            gpu
        )
    else:
        train_dataloader, test_dataloader = load_image_folder_dataloader(
            params['data_dir'],
            params['input_size'],
            params["batch_size"],
            params['time'],
            params['dt'],
            params['intensity'],
            gpu
        )

    print('# Dataloaders successfully loaded.\n')

    ## ========== NETWORK ========== ##

    network = DiehlAndCook2015v2(
        n_inpt=params['input_size']**2,
        n_neurons=params['n_neurons'],
        inh=params['inh'],
        dt=params['dt'],
        norm=78.4,
        nu=(1e-4, 1e-2),
        reduction=reduction,
        theta_plus=params['theta_plus'],
        inpt_shape=(1, params['input_size'], params['input_size']),
    )

    print(f'# Network successfully loaded and set to device : {device}\n')

    ## ========== TRAINING ========== ##

    writer = SummaryWriter(log_dir=log_dir, flush_secs=60)
    
    experiment = UnsupervisedClassificationExperiment(
        network,
        writer,
        params['n_neurons'],
        params['n_classes'],
        params['time'],
        device
    )

    print(f"# Tensorboard and Experiment Class loaded.\n")

    experiment.train(
        train_dataloader=train_dataloader, 
        epochs=params['training_epochs'], 
        update_steps=update_steps
    )