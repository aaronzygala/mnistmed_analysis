# mnistmed_analysis

## Setup Instructions

1. Ensure you have Python 3.12 installed on your system
2. Create a virtual environment using
```
python3 -m venv .venv
```
3. Now activate that venv with this command for Windows:
```
.\.venv\Scripts\activate
```
or use this command for macOS:
```
source .venv/bin/activate
```
4. Now that the virtual env is activated, run this command to install all of the necessary packages:
```
pip install -r requirements.txt
```
5. Your environment should now be set up, and you can run the main training/testing/validation script by running:
```
cd src
python main.py
```


## Usage examples

1. You can configure parts of the model and main program for testing in the **config** directory. Here you can edit the learning rate, and the gamma value, to try and test for different results
2. You can also edit the default values for the command line arguments (or you can just enter different values in the command line). Below is a list of command line values with a short description of what they do.

| Argument Name    | Command Line Syntax     | Description                                                                                    |
|------------------|-------------------------|------------------------------------------------------------------------------------------------|
| Data Flag        | --data_flag {STRING}    | The name of the **2D dataset** to use in the experiment. Full list of names found in MedMNIST github. |
| Output Root      | --output_root {STRING}  | Where to save the models and results                                                           |
| Number of Epochs | --num_epochs {INTEGER}      | Number of epochs for training, script will only test the model if it is set to 0               |
| Size             | --size {INTEGER}            | The image size of the dataset. Can be 28, 64, 128, or 224. (default = 28)                      |
| GPU Ids          | --gpu_ids {STRING}         | The ids of the gpus you would like to use for training                                         |
| Batch Size       | --batch_size {INTEGER}      | The size of training batches                                                                   |
| Download         | --download              | Use this boolean flag if you want to download the dataset                                      |
| Resize           | --resize                | Use this boolean flag if you want to resize images from 28x28 to 224x224                       |
| As RGB           | --as_rgb                | This boolean flag converts grayscale images to rgb                                             |
| Model Path       | --model_path {STRING}      | Defines a model path of pretrained model to use for testing                                    |
| Model Flag       | --model_flag {STRING}      | Choose between Resnet18 or Resnet50                                                            |
| Run              | --run {STRING}             | Used to name the output csv eval file                                                          |


## Reproduction Steps & Results

1. To obtain the same results as myself while running this, please use the default configurations that exist in the config.py file and run
```
cd src
python main.py
```
**Important Note**: You may need to run this command with the --download flag if you have not downloaded the dataset before.

# Results

**pathmnist**
-------------
|       | auc     | acc     |
|-------|---------|---------|
| train | 1.0     | 0.99898 |
| val   | 0.99985 | 0.98800 |
| test  | 0.98326 | 0.88607 |

**chestmnist**
|       | auc     | acc     |
|-------|---------|---------|
| train | 0.85017 | 0.95105 |
| val   | 0.77440 | 0.94908 |
| test  | 0.76910 | 0.94746 |

**bloodmnist**
|       | auc     | acc     |
|-------|---------|---------|
| train | 1.00000 | 0.99925 |
| val   | 0.99852 | 0.96846 |
| test  | 0.99738 | 0.95937 |

**dermamnist**
|       | auc     | acc     |
|-------|---------|---------|
| train | 0.95100 | 0.80063 |
| val   | 0.92780 | 0.75972 |
| test  | 0.92401 | 0.75711 |

**octmnist**
|       | auc     | acc     |
|-------|---------|---------|
| train | 0.99872 | 0.98744 |
| val   | 0.98223 | 0.93390 |
| test  | 0.95481 | 0.75800 |

**pneumoniamnist**
|       | auc     | acc     |
|-------|---------|---------|
| train | 1.00000 | 1.00000 |
| val   | 0.99631 | 0.97519 |
| test  | 0.94237 | 0.84776 |

**retinamnist**
|       | auc     | acc     |
|-------|---------|---------|
| train | 0.77690 | 0.52315 |
| val   | 0.79272 | 0.60833 |
| test  | 0.73599 | 0.50250 |

**breastmnist**
|       | auc     | acc     |
|-------|---------|---------|
| train | 0.87274 | 0.71245 |
| val   | 0.87051 | 0.69231 |
| test  | 0.84607 | 0.73718 |

**tissuemnist**
|       | auc     | acc     |
|-------|---------|---------|
| train | 0.94424 | 0.70617 |
| val   | 0.92757 | 0.67081 |
| test  | 0.92766 | 0.66853 |
