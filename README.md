# Estimating High Order Gradients of the Data Distribution by Denoising (NeurIPS 2021)

Paper: https://arxiv.org/abs/2111.04726

[Chenlin Meng](https://cs.stanford.edu/~chenlin/), [Yang Song](https://yang-song.github.io/), Wenzhe Li, [Stefano Ermon](https://cs.stanford.edu/~ermon/)

**Stanford University** and **Tsinghua University**

Abstract: The first order derivative of a data density can be estimated efficiently by denoising score matching, and has become an important component in many applications, such as image generation and audio synthesis. Higher order derivatives provide additional local information about the data distribution and enable new applications. Although they can be estimated via automatic differentiation of a learned density model, this can amplify estimation errors and is expensive in high dimensional settings. To overcome these limitations, we propose a method to directly estimate high order derivatives (scores) of a data density from samples. We first show that denoising score matching can be interpreted as a particular case of Tweedie's formula. By leveraging Tweedie's formula on higher order moments, we generalize denoising score matching to estimate higher order derivatives. We demonstrate empirically that models trained with the proposed method can approximate second order derivatives more efficiently and accurately than via automatic differentiation. We show that our models can be used to quantify uncertainty in denoising and to improve the mixing speed of Langevin dynamics via Ozaki discretization for sampling synthetic data and natural images.

## Requirements

The code has been tested on PyTorch 1.10.1 (CUDA 10.2).

To install necessary packages and dependencies, run
```
pip install -r requirements.txt
```

## Training the model
To train the model using denoising score matching on MNIST with noise scale 0.5, run

```
python main.py --runner UnetDSMRunner --config dsm_mnist.yml --doc dsm_mnist_0.5  --noise 0.5 --ni
```

To train the model with second order denoising score matching on MNIST with noise scale 0.5, run

```
python main.py --runner LowRankHybridRunner --config unet_mnist.yml --doc low_rank_mnist_0.5  --noise 0.5 --ni
```

To train the model with diagonal second order denoising score matching on CIFAR-10 with noise scale 0.5, run

```
python main.py --runner DoubleDiagonalHybridRunner --config unet_cifar10.yml --doc diagonal_cifar10_0.5  --noise 0.5 --ni
```

## Visualizing the results (coming soon)
To visualize the results, run the notebook in the  `notebooks` folder. 

## Citation

```tex
@article{meng2021estimating,
  title={Estimating High Order Gradients of the Data Distribution by Denoising},
  author={Meng, Chenlin and Song, Yang and Li, Wenzhe and Ermon, Stefano},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

This implementation is based on / inspired by:

- [https://github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion) (the DDPM TensorFlow repo), 
- [https://github.com/pesser/pytorch_diffusion](https://github.com/pesser/pytorch_diffusion) (PyTorch helper that loads the DDPM model), and
- [https://github.com/ermongroup/ncsnv2](https://github.com/ermongroup/ncsnv2) (code structure).
