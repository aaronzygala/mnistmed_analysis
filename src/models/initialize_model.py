from models import ResNet18, ResNet50
from torchvision.models import resnet18, resnet50
import torch, os
from medmnist import INFO


def choose_device(gpu_ids):
    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids[0])

    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu') 
    return device

def initialize_model(data_flag: str, model_flag: str, as_rgb: bool, resize: bool, gpu_ids):

    info = INFO[data_flag]
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])

    print('==> Building and training model...')
    
    if model_flag == 'resnet18':
        model =  resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        model =  resnet50(pretrained=False, num_classes=n_classes) if resize else ResNet50(in_channels=n_channels, num_classes=n_classes)
    else:
        raise NotImplementedError

    device = choose_device(gpu_ids)
    model = model.to(device)
    return (model, device)
