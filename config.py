config_cifar10 = {
    'dataset': 'CIFAR10',
    'dataset_root_path': './dataset/',
    'model_weights_root_path': './model_weights/',
    'device': 'cpu',
    'n_epochs': 2,
    'lr': 0.002,
    'verbose': False,
    'batch_size': 64,
    'random_seed': 42,
    'image_size': 64,
    'loss_alpha': 0.5,
    'reduction': True
}

config_mnist = {
    'dataset': 'MNIST',
    'dataset_root_path': './dataset/',
    'model_weights_root_path': './model_weights/',
    'device': 'cpu',
    'n_epochs': 2,
    'lr': 0.002,
    'verbose': False,
    'batch_size': 64,
    'random_seed': 42,
    'image_size': 64,
    'loss_alpha': 0.5,
    'reduction': True
}