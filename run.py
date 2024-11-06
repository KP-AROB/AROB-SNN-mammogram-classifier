import os, torch, uuid
from argparse import ArgumentParser
from src.experiment.unsupervised import UnsupervisedClassificationExperiment
from bindsnet.models import DiehlAndCook2015v2
from src.utils.common import max_without_indices, write_args_to_file
from src.utils.dataloaders import load_image_folder_dataloader, load_mnist_dataloader
from torch.utils.tensorboard import SummaryWriter
from src.utils.params import load_parameters, instanciate_encoder

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--params_file", type=str, default='./parameters/mnist.yaml')
    args = parser.parse_args()

    ## ========== PARAMS & SEED ========== ##

    params = load_parameters(args.params_file)

    update_steps = max(250 // params["training"]["batch_size"], 1)
    experiment_id = str(uuid.uuid4())[:8]
    experiment_name = f'experiment_{experiment_id}' if not params['name'] else params['name']
    print('\n# Initialization of the experiment protocol {} - {} \n'.format(params['training']['task'], experiment_name))
    log_dir = f"{os.path.join(params['training']['log_dir'], params['training']['task'], experiment_name)}"
    os.makedirs(log_dir, exist_ok=True)
    write_args_to_file(params, log_dir)

    gpu = torch.cuda.is_available()

    if gpu:
        torch.cuda.manual_seed_all(params['training']['seed'])
    else:
        torch.manual_seed(params['training']['seed'])

    device = 'cuda' if gpu else 'cpu'

    if params['training']['reduction'] == "sum":
        reduction = torch.sum
    elif params['training']['reduction'] == "mean":
        reduction = torch.mean
    elif params['training']['reduction'] == "max":
        reduction = max_without_indices
    else:
        raise NotImplementedError

    ## ========== DATASET ========== ##

    image_encoder = instanciate_encoder(params['encoder']['module_name'], params['encoder']['class_name'], params['encoder']['params'])
    if params['training']['task'] == 'MNIST':
        train_dataloader, test_dataloader = load_mnist_dataloader(
            data_dir=params['training']['data_dir'],
            image_size=params['training']['input_size'],
            encoder=image_encoder,
            batch_size=params['training']["batch_size"],
            intensity=params['training']['intensity'],
            gpu=gpu
        )
    else:
        train_dataloader, test_dataloader = load_image_folder_dataloader(
            data_dir=params['training']['data_dir'],
            image_size=params['training']['input_size'],
            encoder=image_encoder,
            batch_size=params['training']["batch_size"],
            intensity=params['training']['intensity'],
            gpu=gpu
        )

    print('# Dataloaders successfully loaded.\n')

    ## ========== NETWORK ========== ##

    network = DiehlAndCook2015v2(
        n_inpt=params['training']['input_size']**2,
        n_neurons=params['training']['n_neurons'],
        inh=params['training']['inh'],
        dt=params['dt'],
        norm=78.4,
        nu=(1e-4, 1e-2),
        reduction=reduction,
        theta_plus=params['training']['theta_plus'],
        inpt_shape=(
            1, 
            params['training']['input_size'], 
            params['training']['input_size']
        ),
    )

    print(f'# Network successfully loaded and set to device : {device}\n')

    ## ========== TRAINING ========== ##

    writer = SummaryWriter(log_dir=log_dir, flush_secs=60)
    
    experiment = UnsupervisedClassificationExperiment(
        network,
        writer,
        params['training']['n_neurons'],
        params['training']['n_classes'],
        params['time'],
        device
    )

    print(f"# Tensorboard and Experiment Class loaded.\n")

    experiment.train(
        train_dataloader=train_dataloader, 
        epochs=params['training']['epochs'], 
        update_steps=update_steps
    )