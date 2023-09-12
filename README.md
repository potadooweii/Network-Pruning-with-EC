[model weights](https://drive.google.com/drive/folders/1zm2v6JbL3GACy7GA7qeZppb0uq_tpsDd?usp=sharing)

See main.py for example of usage.

Default resnet18 with input image of (64,64)  
Default vgg11 with input image of (224,224) <- the (64,64) also available now.

The automatically loaded model would be the one without postfix (i.e. vgg11_MNIST.pt, resnet18_MNIST.pt etc.),
which means if you want to use (64,64)-vgg11, you should modify the file name.


File Structure:
```
project
│   README.md
|   *.py
|   ...
└───model_weights
│   │   vgg11_MNIST.pt
│   │   resnet18_MNIST.pt
│   │   ...
│   └───pruned
│       │   pruned_resnet18_MNIST.model
│       │   ...
│   
└───dataset (generated automatically)
    │   ...
```