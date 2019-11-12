from models.PFLD_Net import PFLD_Net
import sys
from models.mobilenetv1 import mobilenet_v1
# sys.path.insert(0,'/home/zp/paper_model/RGB_liveness')
from models.FeathernetB_R import FeatherNetB
import pytorch_to_caffe
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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

if __name__ == '__main__':
    name=sys.argv[1]
#    net=mobilenet_v1()
    net=FeatherNetB()
    #net=net.to('cuda')
    path= sys.argv[2]
#     model_CKPT = torch.load(path, map_location=lambda storage, loc: storage)
#     net.load_state_dict(model_CKPT['state_dict'])
#     net.eval()
#     input = torch.ones([1, 3, 224, 224])
#     pytorch_to_caffe.trans_net(net, input, name)
#     pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
#     pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
    
    
    checkpoint = torch.load(path,map_location = 'cpu')
    print('load model:',path)
    model_dict = {}
    state_dict = net.state_dict()
    #print(checkpoint)
    for (k,v) in checkpoint['state_dict'].items():
        print(k)
        if k[7:] in state_dict:
            model_dict[k[7:]] = v
    state_dict.update(model_dict)
    net.load_state_dict(state_dict)
    net.eval()
    input = torch.ones([1, 3, 224, 224])
    pytorch_to_caffe.trans_net(net, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))

#