import argparse
import logging
import time
# select GPU on the server
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# pytorch related package 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

print('pytorch version: ' + torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ",device)
# NVIDIA apex
from apex import amp
# math and showcase
import matplotlib.pyplot as plt
import numpy as np
# pre-define model: https://github.com/lukemelas/EfficientNet-PyTorch
from efficientnet.model import EfficientNet
from utils import *

parser = argparse.ArgumentParser( description='PyTorch efficientnet model playground')
parser.add_argument('--resume', '-r',       default=False,        type=bool,  help='resume from checkpoint')
parser.add_argument('--sess',               default='default',    type=str,   help='session id')
parser.add_argument('--seed',               default=70082353,     type=int,   help='random seed')
parser.add_argument('--lr', '-lr',          default=0.01,        type=float, help='learning rate')
parser.add_argument('--batch_size', '-b',   default=32,           type=int,   help='mini-batch size (default: 120)')
parser.add_argument('--epochs', '-e',       default=1000,         type=int,   help='number of total epochs to run')
parser.add_argument('--early_stop', '-es',  default=30,           type=int,   help='early stop')
parser.add_argument('--image_size', '--is', default=380,          type=int,   help='resize input image (default: 256)')
parser.add_argument('--image_crop', '--ic', default=380,          type=int,   help='centercrop input image after resize (default: 128 for food-11)')
parser.add_argument('--data-directory',     default='../food-11', type=str,   help='dataset inputs root directory')
parser.add_argument('--opt-level', '-o',    default='O1',         type=str,   help='Nvidia apex optimation level (default: O1)')
parser.add_argument('--model_name', '-m',   default='efficientnet-b4', type=str, help='Specify the varient of the model ')
args = parser.parse_args()

def main():
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # load dataset (Imagenet)
    train_loader, test_loader, val_loader = get_loaders(args.data_directory, args.batch_size, args.image_size, args.image_crop)


    # Load model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        model = EfficientNet.from_name(args.model_name).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

        checkpoint = torch.load('./checkpoint/' + args.sess + '_' + str(args.seed) + '.pth')
        prev_acc = checkpoint['acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        amp.load_state_dict(checkpoint['amp_state_dict'])
        epoch_start = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    else:
        print('==> Building model..')
        epoch_start = 0
        prev_acc = 0.0
        model = EfficientNet.from_name(args.model_name).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # Logger
    result_folder = './results/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    logger = logging.getLogger(__name__)
    logname = args.model_name + '_' + args.sess + '_' + args.opt_level + '_' + str(args.seed) + '_bs=64_lr=0.01_RMSprop_noApex' + '.log'
    logfile = os.path.join(result_folder, logname)
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile
    )
    logger.info(args)

    # Training
    def train(epoch):
        print('\nEpoch: {:04}'.format(epoch))
        train_loss, correct, total = 0, 0, 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output_logit = model(data)
            loss = F.cross_entropy(output_logit, target)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            
            # backward only for apex
            #loss.backward()
            optimizer.step()
            preds = F.softmax(output_logit, dim=1)
            preds_top_p, preds_top_class = preds.topk(1, dim=1)

            train_loss += loss.item()
            total += target.size(0)
            correct += (preds_top_class.view(target.shape) == target).sum().item()

        return (train_loss / total, 100. * correct / total)

    # Test
    def test(epoch, loader):
        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
    
                optimizer.zero_grad()
                output_logit = model(data)
                loss = F.cross_entropy(output_logit, target)
                preds = F.softmax(output_logit, dim=1)
                preds_top_p, preds_top_class = preds.topk(1, dim=1)
    
                test_loss += loss.item()
                total += target.size(0)
                correct += (preds_top_class.view(target.shape) == target).sum().item()
        
        return (test_loss / total, 100. * correct / total)
            
    # Save checkpoint
    def checkpoint(acc, epoch):
        print('==> Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_path = './checkpoint/' + args.sess + '_' + str(args.seed) + '.pth'
        torch.save({
            'epoch': epoch,
            'acc': acc,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'amp_state_dict': amp.state_dict(),
            'rng_state': torch.get_rng_state(),
            }, save_path)
    
    # Run
    max_acc = 0.0
    early_stop_cnt = 0
    logger.info('Epoch \t Seconds \t \t Train Loss \t Train Acc')
    start_train_time = time.time()
    for epoch in range(epoch_start, args.epochs):
        start_epoch_time = time.time()
        
        train_loss, train_acc = train(epoch)
        print('train_loss = {:.4f}, train_acc = {:.4f}'.format(train_loss, train_acc))
        epoch_time = time.time()
        logger.info('%5d \t %7.1f \t \t %10.4f \t %9.4f', epoch, epoch_time - start_epoch_time, train_loss, train_acc)
        
        # validation
        val_loss, val_acc = test(epoch, val_loader)
        print('val_loss = {:.4f}, val_acc = {:.4f}'.format(val_loss, val_acc))
        
        if val_acc > max_acc:
            max_acc = val_acc
            # Save checkpoint
            checkpoint(train_acc, epoch)
            test_loss, test_acc = test(epoch, test_loader)
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        if early_stop_cnt > args.early_stop:
            break
    train_time = time.time()
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    logger.info('Test Loss \t Test Acc')
    logger.info('%9.4f \t %8.4f', test_loss, test_acc)
    print('test_loss = {:.4f}, test_acc = {:.4f}'.format(test_loss, test_acc))


if __name__ == "__main__":
    main()
