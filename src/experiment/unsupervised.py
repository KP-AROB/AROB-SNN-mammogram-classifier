import torch, math
from torch.utils.tensorboard import SummaryWriter
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from torch.utils.data import DataLoader
from tqdm import tqdm
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from time import time
from bindsnet.utils import get_square_weights
import numpy as np
from src.utils.common import colorize

class UnsupervisedClassificationExperiment(object):

    def __init__(self, 
            network: Network, 
            writer: SummaryWriter, 
            n_neurons: int = 100, 
            n_classes: int = 10, 
            data_sim_time: int = 100, 
            device: str = 'cuda') -> None:
        self.network = network.to(device)
        self.n_classes = n_classes
        self.n_neurons = n_neurons
        self.writer = writer
        self.data_sim_time = data_sim_time
        self.device = device
        self.__init_recorder()
        self.__init_monitor()
    
    def __init_recorder(self):
        self.class_assignments = -torch.ones(self.n_neurons)
        self.weight_proportions = torch.zeros(self.n_neurons, self.n_classes)
        self.neuron_rates = torch.zeros(self.n_neurons, self.n_classes)

    def __init_monitor(self):
        self.spikes_monitor = {}
        for layer in set(self.network.layers):
            self.spikes_monitor[layer] = Monitor(self.network.layers[layer], state_vars=["s"], time=self.data_sim_time)
            self.network.add_monitor(self.spikes_monitor[layer], name="%s_spikes" % layer)

    def train(self, train_dataloader: DataLoader, update_steps, epochs: int = 1):
        print('# Running Unsupervised Classification Experiment \n')

        n_sqrt = int(np.ceil(np.sqrt(self.n_neurons)))
        update_interval = update_steps * train_dataloader.batch_size
        spike_recorder = torch.zeros(update_interval, self.data_sim_time, self.n_neurons)

        for i in range(epochs):
            labels = []
            with tqdm(total=len(train_dataloader), desc=f"Training Epoch {i}") as pbar:
                for step, batch in enumerate(train_dataloader):
                    global_step = (len(train_dataloader) * train_dataloader.batch_size) * i + train_dataloader.batch_size * step
                    if step % update_steps == 0 and step > 0:
                        # Convert the array of labels into a tensor
                        label_tensor = torch.tensor(labels)

                        # Get network predictions.
                        all_activity_pred = all_activity(
                            spikes=spike_recorder, assignments=self.class_assignments, n_labels=self.n_classes
                        )
                        
                        self.writer.add_scalar(
                            tag="accuracy/all vote",
                            scalar_value=torch.mean(
                                (label_tensor.long() == all_activity_pred).float()
                            ),
                            global_step=global_step,
                        )
                        square_weights = get_square_weights(
                            self.network.connections["X", "Y"].w.view(self.network.n_inpt, self.n_neurons),
                            n_sqrt,
                            int(math.sqrt(self.network.n_inpt)),
                        )
                        img_tensor = colorize(square_weights, cmap="hot_r")

                        self.writer.add_image(
                            tag="weights",
                            img_tensor=img_tensor,
                            global_step=global_step,
                            dataformats="HWC",
                        )

                        self.class_assignments, _, self.neuron_rates = assign_labels(
                            spikes=spike_recorder,
                            labels=label_tensor,
                            n_labels=self.n_classes,
                            rates=self.neuron_rates,
                        )

                        labels = [] 
                    labels.extend(batch["label"].tolist())
                    inpts = {"X": batch["encoded_image"]}
                    if self.device == 'cuda':
                        inpts = {k: v.cuda() for k, v in inpts.items()}
                    t0 = time()
                    self.network.run(inputs=inpts, time=self.data_sim_time, one_step=True)
                    t1 = time() - t0
                    s = self.spikes_monitor["Y"].get("s").permute((1, 0, 2))
                    spike_recorder[
                        (step * train_dataloader.batch_size)
                        % update_interval : (step * train_dataloader.batch_size % update_interval)
                        + s.size(0)
                    ] = s
                    self.writer.add_scalar(
                        tag="time/simulation", scalar_value=t1, global_step=global_step
                    )
                    self.network.reset_state_variables()
                    pbar.update()