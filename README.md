# BindsNet implementation of a two-layer SNN for image classification


## Installation

```bash
conda env create -f environment.yml
```

Alternatively you can install the dependencies in your own Python setup with : 

```bash
pip install -r requirements.txt
```

## Usage

All flag variables are set by default so running the following command will launch the SNN training :

```bash
python run.py --params_file ./parameters/mnist.yaml
```

| Parameters     | Description                                                                               | Default Value |
|----------------|-------------------------------------------------------------------------------------------|---------------|
| task         | The classification task to perform. "MNIST" or "MAMMO"                                      | "MNIST"       |
| log_dir      | The directory to save the log files for tensorboard                                         | "./logs"      |
| time         | The simulation time for one input                                                           | 100           |
| intensity    | The intensity multiplier for image pixels                                                   | 128           |
| n_classes    | The number of class on which to perform classification                                      | 10            |
| batch_size   | The batch size for both training and validation sets                                        | 16            |
| seed         | The seed used for reproductibility                                                          | 1234          |
| n_neurons    | The amount of neurons for both Excitatory and Inhibitory layers                             | 100           |
| input_size   | The size of the input image                                                                 | 28            |
| reduction    | The value defining how neuron-level outputs should be aggregated. ("max", "sum", "mean")    | "sum"         |
| theta_plus   | The magnitude of weight changes                                                             | 0.05          |
| dt           | The simulation time step                                                                    | 1.0           |
| inh          | The strength of inhibition weights                                                          | 120           |
| exc          | The strength of excitation weights                                                          | 22.5          |


## Logs

Some scalars are gathered during training and can be visualized using Tensorboard.
The Tensorboard logs are stored automatically in your ```log_dir``` for each experiment you run.
To launch the board, run : 

```bash
tensorboard  --logdir ./logs
```

The command will find every log file and display the scalars inside the GUI.
For reproductibility, the parameters of your experiments are also saved as a .txt file in the same directory as the experiments' logs.