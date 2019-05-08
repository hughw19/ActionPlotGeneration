# Learning a Generative Model for Multi-Step Human-Object Interactions from Videos
This is a tensorflow implementation of Action Plot RNN in the Eurographics 2019 paper, Learning a Generative Model for Multi-Step Human-Object Interactions from Videos. The model generates action plots. 

![Teaser](assets/teaser.jpg)

The repository includes:

* Source code of Action Plot RNN.
* Training code
* Pre-trained weights
* Sampling code for generating action plots

# Requirements
* Python 3.5
* Tensorflow 1.3.0
* tflearn
* cPickle

# Video Dataset
For those who are interested in the interaction videos, one can download our dataset via https://drive.google.com/open?id=1Ggpxm1kyQ9Lm3itcvFrIGuWhNRml4o28.


# Training
```
# Train a new Action Plot model from scratch
python3 train.py
```

# Generation
```
# Sampling action plots using a checkpoint
python3 sample.py --save_dir=/ckpts/ckpts_dir --obj_list="book phone bowl bottle cup orange"
```



