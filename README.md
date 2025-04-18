# CNNs-for-Satellite-Imagery
## Abstract
Satellite imagery is highly beneficial for analyzing and monitoring Earth-related tasks, especially those related to climate change. When analyzing satellite imagery with machine learning models, it is common to transfer existing methods to satellite imagery without taking into consideration the uniqueness of its domain. The differences between satellite imagery and other data modalities in which existing methods were developed for likely hinders the performance of these methods on satellite data. In this work, we evaluate the impact of convolutional neural network (CNN) depth and determine a default ResNet18 architecture is not optimal across several satellite imagery tasks. We show that in all six tasks a CNN with fewer layers than ResNet18 outperforms the full ResNet18 model.  


## Data
[UC Merced Land Use](http://weegee.vision.ucmerced.edu/datasets/landuse.html)

[Brazilian Coffee Scenes](https://patreo.dcc.ufmg.br/2017/11/12/brazilian-coffee-scenes-dataset/)

[EuroSAT](https://github.com/phelber/EuroSAT)

[Forest cover, Nighttime lights, Elevation](https://www.nature.com/articles/s41467-021-24638-z)

## Instructions
Use the config files to specify run parameters and paths.py to specify the data directories. Example config files can be found in the config directory. Paper results were achieved with the settings specified in the config files. 

To train a model use: ```python run.py --c <path_to_config_file>```

