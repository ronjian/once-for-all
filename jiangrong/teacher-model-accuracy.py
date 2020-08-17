import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import sys; sys.path.append('/workspace/once-for-all')
from ofa.elastic_nn.networks import OFAMobileNetV3
from PIL import Image
import torchvision.transforms as transforms
import math
import numpy as np
import torch.nn.functional as F

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def load_weights_from_net(model, src_model_dict):
    model_dict = model.state_dict()
    for key in src_model_dict:
        if key in model_dict:
            new_key = key
        elif '.bn.bn.' in key:
            new_key = key.replace('.bn.bn.', '.bn.')
        elif '.conv.conv.weight' in key:
            new_key = key.replace('.conv.conv.weight', '.conv.weight')
        elif '.linear.linear.' in key:
            new_key = key.replace('.linear.linear.', '.linear.')
        elif '.linear.' in key:
            new_key = key.replace('.linear.', '.linear.linear.')
        elif 'bn.' in key:
            new_key = key.replace('bn.', 'bn.bn.')
        elif 'conv.weight' in key:
            new_key = key.replace('conv.weight', 'conv.conv.weight')
        else:
            raise ValueError(key)
        assert new_key in model_dict, '%s' % new_key
        model_dict[new_key] = src_model_dict[key]
    model.load_state_dict(model_dict)

def preprocess(img_path):
    img = pil_loader(img_path)
    image_size = 224
    trans = transforms.Compose([
                    transforms.Resize(int(math.ceil(image_size / 0.875))),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[
                            0.485,
                            0.456,
                            0.406],
                        std=[
                            0.229,
                            0.224,
                            0.225]),])
    img = trans(img)
    img = img.unsqueeze(0)
    return img

def get_teacher_model():
    teacher_path = '/workspace/once-for-all/.torch/ofa_checkpoints/0/ofa_D4_E6_K7'
    bn_momentum = 0.1
    bn_eps = 1e-5
    teacher_model = OFAMobileNetV3(
        n_classes=1000, bn_param=(bn_momentum, bn_eps),
        dropout_rate=0, width_mult_list=1.0, ks_list=7, expand_ratio_list=6, depth_list=4,
    )
    init = torch.load(teacher_path, map_location='cpu')['state_dict']
    load_weights_from_net(teacher_model, init)
    teacher_model.eval()
    return teacher_model

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_class2idx():
    res = {}
    with open('/dataset/ILSVRC2012/labels.txt', 'r') as rf:
        class_id = 0
        for line in rf:
            classname = line.split(',')[0]
            res[classname] = class_id
            class_id += 1
    return res

top1 = AverageMeter('Acc@1', ':6.2f')
top5 = AverageMeter('Acc@5', ':6.2f')
teacher_model = get_teacher_model()
class2idx = get_class2idx()

csum = 0
for cur, _, files in os.walk('/dataset/ILSVRC2012/val'):
    for file in files:
        if file.endswith('JPEG'):
            clsname = cur.split('/')[-1]
            # sample_path = "/dataset/ILSVRC2012/val/n03085013/ILSVRC2012_val_00035446.JPEG" # keyboard,  509 n03085013 computer keyboard, keypad
            sample_path = os.path.join(cur, file)
            input_tensor = preprocess(sample_path)
            soft_logits = teacher_model(input_tensor).detach()
            output = F.softmax(soft_logits, dim=0).unsqueeze(0)
            target = torch.Tensor(np.array([class2idx[clsname]])).unsqueeze(0)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # print(np.argsort(soft_label.numpy())[-5:])
            top1.update(acc1[0], 1)
            top5.update(acc5[0], 1)
            csum += 1
            if csum % 1000 == 0:
                print(csum, top1.avg, top5.avg)
            
print('finally:', top1.avg, top5.avg)
# finally: tensor(77.4220) tensor(93.5820)