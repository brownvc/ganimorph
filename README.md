# GANimorph: Improving Shape Deformation in Unsupervised Image to Image Translation

This Git repository hosts the official code for 'Improving Shape Deformation in Unsupervised Image to Image Translation', by [Aaron Gokaslan](https://skylion007.github.io/), Vivek Ramanujan, [Daniel Ritchie](https://dritchie.github.io/), Kwang In Kim, and [James Tompkin](www.jamestompkin.com), to be published at [ECCV 2018](https://eccv2018.org/).

[arXiv preprint](http://arxiv.org/abs/1808.04325)

# Dataset and Pretrained Models
The datasets used in the paper and  pretrained models can be [downloaded here](https://drive.google.com/drive/u/1/folders/1xhOp43mmPSmL1_P6oYVzGNBgmAWeDYMO).

# Bibtex

```
@inproceedings{Gokaslan2018,
  title={Improving Shape Deformation in Unsupervised Image to Image Translation},
  author={Aaron Gokaslan and Vivek Ramanujan and Daniel Ritchie and Kwang In Kim and James Tompkin},
  booktitle={European Conference on Computer Vision},
  year={2018}
}
```

# Dataset
The Microsoft Research Kaggle Cat vs. Dog Dataset serves as an excellent demo. Simply download the dataset and remove 128 examples from the holdout list.

# Dependencies

Tensorflow > 1.2 (GPU strongly recomended)
This repo uses the Tensorpack library and this particular has been updated to work with Tensorpack 0.8.9.
See the other branch to see the original version that worked only with Tensorflow version <=1.3.
