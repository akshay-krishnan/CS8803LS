# Unsupervised Generative Disentangled Action Embedding Extraction

**Unsupervised**: without action supervision

**Generative**: We use a generrative model

**Disentangled**: We wish to extract the action embedding only, which is object-agnostic.

## Model
![Model](Framework.png)

## Slides
[Proposal](https://docs.google.com/presentation/d/1tsfbN8aLZl1RS3epBdQ2Jkkthdxr8mmmnycfGUP0ld8/edit?usp=sharing)

## Training and testing

Our framework can be used in several modes. In the motion transfer mode, a static image will be animated using a driving video. In the image-to-video translation mode, given a static image, the framework will predict future frames.

### Installation

We support ```python3```. To install the dependencies run:
```
pip3 install -r requirements.txt
```

### YAML configs

There are several configuration (```config/dataset_name.yaml```) files one for each `dataset`. See ```config/actions.yaml``` to get description of each parameter.

<!-- ### Motion Transfer Demo 

To run a demo, download a [checkpoint](https://yadi.sk/d/BX-hwuPEVm6iNw) and run the following command:
```
python demo.py --config  config/moving-gif.yaml --driving_video sup-mat/driving.png --source_image sup-mat/source.png --checkpoint path/to/checkpoint
```
The result will be stored in ```demo.gif```. -->

### Training

Training the model : 
```
python3 run_update1.py --config config/actions.yaml --mode train --verbose
```
To train a model on specific dataset run:
```
CUDA_VISIBLE_DEVICES=0 python3 run.py --config config/dataset_name.yaml
```
The code will create a folder in the log directory (each run will create a time-stamped new directory).
Checkpoints will be saved to this folder.
To check the loss values during training in see ```log.txt```.
You can also check training data reconstructions in the ```train-vis``` subfolder.

<!-- ### Reconstruction

To evaluate the reconstruction performance run:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config config/dataset_name.yaml --mode reconstruction --checkpoint path/to/checkpoint
```
You will need to specify the path to the checkpoint,
the ```reconstruction``` subfolder will be created in the checkpoint folder.
The generated video will be stored to this folderenerated video there and in ```png``` subfolder loss-less verstion in '.png' format.

### Motion transfer

In order to perform motion transfer run:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config config/dataset_name.yaml --mode transfer --checkpoint path/to/checkpoint
```
You will need to specify the path to the checkpoint,
the ```transfer``` subfolder will be created in the same folder as the checkpoint.
You can find the generated video there and its loss-less version in the ```png``` subfolder.

There are 2 different ways of performing transfer:
by using **absolute** keypoint locations or by using **relative** keypoint locations.

1) Absolute Transfer: the transfer is performed using the absolute postions of the driving video and appearance of the source image.
In this way there are no specific requirements for the driving video and source appearance that is used.
However this usually leads to poor performance since unrelevant details such as shape is transfered.
Check transfer parameters in ```shapes.yaml``` to enable this mode.

2) Realtive Transfer: from the driving video we first estimate the relative movement of each keypoint,
then we add this movement to the absolute position of keypoints in the source image.
This keypoint along with source image is used for transfer. This usually leads to better performance, however this requires
that the object in the first frame of the video and in the source image have the same pose.

The approximately aligned pairs of videos are given in the data folder. (e.g  ```data/taichi.csv```). -->

### Image-to-video translation

In order to perform image-to-video translation run:
```
CUDA_VISIBLE_DEVICES=0 python3 run.py --config config/dataset_name.yaml --mode prediction --checkpoint path/to/checkpoint
```

### Datasets

1) **Shapes**. This dataset is saved along with repository.
Training takes about 1 hour.

2) **Actions**. This dataset is also saved along with repository.
 And training takes about 4 hours.

<!-- ### Training on your own dataset
1) Resize all the videos to the same size e.g 128x128, the videos can be in '.gif' or '.mp4' format. But we recommend to make them stacked '.png' (see data/shapes), because this format is lossless.

2) Create a folder ```data/dataset_name``` with 2 subfolders ```train``` and ```test```, put training videos in the ```train``` and testing in the ```test```.

3) Create a config ```config/dataset_name.yaml``` (it is better to start from one of the existing configs, for 64x64 videos ```config/nemo.yaml```, for 128x128 ```config\moving-gif.yaml```, for 256x256 ```config\vox.yaml```), in dataset_params specify the root dir the ```root_dir:  data/dataset_name```. Also adjust the number of epoch in train_params. -->

## Reference
[1][Animating Arbitrary Objects via Deep Motion Transfer](https://arxiv.org/abs/1812.08861) by Aliaksandr Siarohin, Stéphane Lathuilière, [Sergey Tulyakov](http://stulyakov.com), [Elisa Ricci](http://elisaricci.eu/) and [Nicu Sebe](http://disi.unitn.it/~sebe/).
