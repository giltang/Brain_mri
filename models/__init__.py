from .resnet_weight_norm import ResNet18, ResNet34, ResNet50, ResNet101
from .densenet_wn import densenet121, densenet161
from .vgg_wn import vgg16_bn, vgg19_bn
#from .efficientnet import efficientnet_b6, efficientnet_b7, efficientnet_v2_s, efficientnet_v2_l
from .googlenet import GoogLeNet
from .simple_cnn import CNN
from .MLP import MLP

model_dict = {
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'DenseNet121': densenet121,
    'DenseNet161': densenet161,
    'VGG16': vgg16_bn,
    'VGG19': vgg19_bn,
#    'EfficientNet_B6': efficientnet_b6,
 #   'EfficientNet_B7': efficientnet_b7,
  #  'EfficientNet_V2_S': efficientnet_v2_s,
  #  'EfficientNet_V2_L': efficientnet_v2_l,
    'GoogleNet': GoogLeNet,
    'CNN': CNN,
    'MLP': MLP,
}

