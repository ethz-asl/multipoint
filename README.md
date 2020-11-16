
# MultiPoint
This is a PyTorch implementation of "MultiPoint: Cross-spectral registration of
thermal and optical aerial imagery"

## Installation
This software requires a base python install of 3.6 or higher.
Requirements can be installed with:
```
pip3 install -r requirements.txt
```

The repository provides models with pretrained weights for MultiPoint but for training the dataset needs to be downloaded separately (see [Dataset](#dataset)).

The multipoint python package can be locally installed by executing:
```
pip3 install -e . --user
```
(You can remove the `--user` flag if operating in a virtual environment)

## Dataset
TODO

## Usage
In the following section the scripts to train and visualize the results of MultiPoint are explained. For each script, additional help on the input paramaters and flags can be found using the `-h` flag (e.g. `python show_keypoints.py -h`).

#### Generating Keypoint Labels
Keypoint labels for a given set of image pairs can be generated using:

```
python3 export_keypoints.py -o /tmp/labels.hdf5
```

where the `-o` flag defines the output filename. The base detector and the export settings can be modified by making a copy of the `configs/config_export_keypoints.yaml` config file, editing the desired parameters, and specifying your new config file with the `-y` flag.
```
python3 export_keypoints.py -y configs/custom_export_keypoints.yaml -o /tmp/labels.hdf5
```


#### Visualizing Keypoint Labels
The generated keypoint labels can be inspected by executing the `show_keypoints.py` script:

```
python3 show_keypoints.py -d /tmp/training.hdf5 -k /tmp/labels.hdf5 -n 100
```

The `-d` flag specifies the dataset file, the `-k` flag the labels file, and the `-n` flag the index of the sample which is shown.

#### Visualizing Synthetic Shapes Samples
A randomly generated sample of the synthetic shapes dataset can be visualized by executing the `show_synthetic_images.py` script (any key to exit).

#### Visualizing Samples from Datasets
By executing the following command:
```
python3 show_image_pair_sample.py -i /tmp/test.hdf5 -n 100
```

the 100th image pair of the `/tmp/test.hdf5` dataset is shown.

#### Training MultiPoint
MultiPoint can be trained by executing the `train.py` script. All that script requires is a path to a yaml file with the training parameters:

```
python3 train.py -y configs/config_multipoint_training.yaml
```

The hyperparameter for the training, e.g. learning rate, can be modified in the yaml file.

#### Predicting Keypoints
Predicting only keypoints can be done executing the `predict_keypoints.py` script.
The results are plotted by adding the `-p` flags and the metrics for the whole dataset are computed by adding the `-e` flag.

#### Predicting the Homography
Predicting the alignment of an image pair can be done using the `predict_align_image_pair.py` script.
The resulting keypoints and matches can be visualized by adding the `-p` flag.
The metrics over the full dataset are computed when adding the `-e` flag.

## Credits
This code is based on previous implementations of SuperPoint by [Rémi Pautrat](https://github.com/rpautrat/SuperPoint) and  [You-Yi Jau](https://github.com/eric-yyjau/pytorch-superpoint)
