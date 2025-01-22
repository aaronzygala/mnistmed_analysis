import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy
import medmnist
import torch
import torch.nn as nn
from medmnist import INFO
from tensorboardX import SummaryWriter
from tqdm import trange
import data.data as data
import models.initialize_model as init
from eval.test import test
from train.train import train
from config.config import Config

def main(data_flag, output_root, gpu_ids, num_epochs, size, model_path, run):

    lr = Config.LR
    gamma=Config.GAMMA
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]

    info = INFO[data_flag]
    task = info['task']

    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    [ 
        train_loader,
        train_loader_at_eval,
        val_loader,
        test_loader 
    ] = data.initialize_dataset(data_flag, download, resize, as_rgb, size, batch_size)

    (model, device) = init.initialize_model(data_flag, model_flag, as_rgb, resize, gpu_ids)

    train_evaluator = medmnist.Evaluator(data_flag, 'train', size=size)
    val_evaluator = medmnist.Evaluator(data_flag, 'val', size=size)
    test_evaluator = medmnist.Evaluator(data_flag, 'test', size=size)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run, output_root)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run, output_root)

        print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
              'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
              'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))

    if num_epochs == 0:
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)
    
    writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model)

    iteration = 0
    
    for epoch in trange(num_epochs):        
        (train_loss, iteration) = train(model, train_loader, task, criterion, optimizer, device, writer, iteration)
        
        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run)
        
        scheduler.step()
        
        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)
            
        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)
            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)

    state = {
        'net': best_model.state_dict(),
    }

    path = os.path.join(output_root, 'best_model.pth')
    torch.save(state, path)

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
    val_metrics = test(best_model, val_evaluator, val_loader, task, criterion, device, run, output_root)
    test_metrics = test(best_model, test_evaluator, test_loader, task, criterion, device, run, output_root)

    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

    log = '%s\n' % (data_flag) + train_log + val_log + test_log
    print(log)
            
    with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
        f.write(log)  
            
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST2D')

    parser.add_argument('--data_flag',
                        default=Config.DEFAULT_DATA_FLAG,
                        type=str)
    parser.add_argument('--output_root',
                        default=Config.DEFAULT_OUTPUT_ROOT,
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=Config.DEFAULT_NUM_EPOCHS,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--size',
                        default=Config.DEFAULT_IMAGE_SIZE,
                        help='the image size of the dataset, 28 or 64 or 128 or 224, default=28',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default=Config.DEFAULT_GPU_IDS,
                        type=str)
    parser.add_argument('--batch_size',
                        default=Config.DEFAULT_BATCH_SIZE,
                        type=int)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--resize',
                        help='resize images of size 28x28 to 224x224',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        help='convert the grayscale image to RGB',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default=Config.DEFAULT_MODEL_FLAG,
                        help='choose backbone from resnet18, resnet50',
                        type=str)
    parser.add_argument('--run',
                        default=Config.DEFAULT_MODEL_NAME,
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)


    args = parser.parse_args()
    data_flag = args.data_flag
    output_root = args.output_root
    num_epochs = args.num_epochs
    size = args.size
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    download = args.download
    model_flag = args.model_flag
    resize = args.resize
    as_rgb = args.as_rgb
    model_path = args.model_path
    run = args.run
    
    main(data_flag, output_root, gpu_ids, num_epochs, size, model_path, run)