python main.py --runner UnetDSMRunner --config dsm_mnist.yml --doc dsm_mnist_0.5  --noise 0.5 --ni
python main.py --runner LowRankHybridRunner --config unet_mnist.yml --doc unet_low_rank_mnist_0.5_with_dsm  --noise 0.5 --ni
python main.py --runner DoubleDiagonalHybridRunner --config unet_cifar10.yml --doc diagonal_cifar10_0.5  --noise 0.5 --ni
