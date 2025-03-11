import torchvision

def get_mnist_dataset():
    torchvision.datasets.MNIST(
        root='../../data/mnist/',
        train=True,
        download=True
    )

    torchvision.datasets.MNIST(
        root='../../data/mnist/',
        train=False,
        download=True
    )

if __name__ == '__main__':
    get_mnist_dataset()