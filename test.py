import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import os
import time
import torch.nn.functional as F
import torch.nn as nn

from utils.dataset.isic import isic2018_dataset, augmentation_rand, augmentation_sim,augmentation_test
from utils.eval_metrics import ConfusionMatrix, Auc
from models.ecl import ECL_model,balanced_proxies
from models.loss import CE_weight,BHP

'''function for getting proxies number'''
def get_proxies_num(cls_num_list):
    ratios = [max(np.array(cls_num_list)) / num for num in cls_num_list]
    prototype_num_list = []
    for ratio in ratios:
        if ratio == 1:
            prototype_num = 1
        else:
            prototype_num = int(ratio // 10) + 2
        prototype_num_list.append(prototype_num)
    assert len(prototype_num_list) == len(cls_num_list)
    return prototype_num_list

def main(args):
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    log_file = open(os.path.join(args.log_path,'test_log.txt'), 'w')

    '''print args'''
    for arg in vars(args):
        print(arg, getattr(args, arg))
        print(arg, getattr(args, arg),file=log_file)


    '''load models'''
    model = ECL_model(num_classes=args.num_classes,feat_dim=args.feat_dim)
    proxy_num_list = get_proxies_num(args.cls_num_list)
    model_proxy = balanced_proxies(dim=args.feat_dim,proxy_num=sum(proxy_num_list))

    if args.cuda:
        model.cuda()
        model_proxy.cuda()
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0),file=log_file)
    print("Model_proxy size: {:.5f}M".format(sum(p.numel() for p in model_proxy.parameters())/1000000.0))
    print("Model_proxy size: {:.5f}M".format(sum(p.numel() for p in model_proxy.parameters())/1000000.0),file=log_file)
    print("=============model init done=============")
    print("=============model init done=============",file=log_file)


    '''load dataset'''
    if args.dataset == 'ISIC2018':
        test_iterator = DataLoader(isic2018_dataset(path=args.data_path, transform=augmentation_test, mode='test'),
                                   batch_size=1, shuffle=False, num_workers=2)
    else:
        raise ValueError("dataset error")

    try:
        model.load_state_dict(torch.load(args.model_path),strict=True)
        model.eval()

        pro_diag, lab_diag = [], []
        confusion_diag = ConfusionMatrix(num_classes=args.num_classes, labels=list(range(args.num_classes)))
        with torch.no_grad():
            for batch_index, (data, label) in enumerate(test_iterator):
                if args.cuda:
                    data = data.cuda()
                    label = label.cuda()
                diagnosis_label = label.squeeze(1)

                output = model(data)
                predicted_results = torch.argmax(output, dim=1)
                pro_diag.extend(output.detach().cpu().numpy())
                lab_diag.extend(diagnosis_label.cpu().numpy())

                confusion_diag.update(predicted_results.cpu().numpy(), diagnosis_label.cpu().numpy())

            print("Test confusion matrix:")
            print("Test confusion matrix:",file=log_file)
            confusion_diag.summary(log_file)
            print("Test AUC:")
            print("Test AUC:",file=log_file)
            Auc(pro_diag, lab_diag, args.num_classes, log_file, task='test')

    except Exception:
        import traceback
        traceback.print_exc()

    finally:
        log_file.close()


parser = argparse.ArgumentParser(description='Training for the classification task')

#dataset
parser.add_argument('--data_path', type=str, default='./data/ISIC2018/', help='the path of the data')
parser.add_argument('--dataset', type=str, default='ISIC2018', help='the name of the dataset')
parser.add_argument('--model_path', type=str, default = './results/ISIC2018/model_100.pth', help='the path of the model')
parser.add_argument('--log_path', type=str, default = './logs/ISIC2018', help='the path of the log')


# training parameters
parser.add_argument('--cuda', type=bool, default=True, help='whether to use cuda')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='gpu device ids for CUDA_VISIBLE_DEVICES')


# hyperparameters for model
parser.add_argument('--feat_dim', dest='feat_dim', type=int, default=128)


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.dataset == 'ISIC2018':
        args.cls_num_list = [84, 195, 69, 4023, 308, 659, 667]
        args.num_classes = 7
    else:
        raise Exception("Invalid dataset name!")
    if args.log_path is None:
        args.log_path = args.model_path
    main(args)
    print("Done!")
