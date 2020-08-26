
import torch
import os; os.environ["CUDA_VISIBLE_DEVICES"] = ''

import sys; sys.path.append("/workspace/once-for-all")

from mbv3 import MobileNetV3_Large
net = MobileNetV3_Large()

# from ofa.imagenet_codebase.networks.mobilenet_v3 import MobileNetV3Large
# net = MobileNetV3Large()

net.eval()
x = torch.Tensor(1, 3, 224, 224)
y = net(x)

from torchsummary import summary
print('model summarization as: ')
summary(net, (3, 224, 224))

trace_model = torch.jit.trace(net, (x, ))
trace_model.save("assets/mbv3.jit")

print('done')
