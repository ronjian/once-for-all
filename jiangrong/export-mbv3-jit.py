from mbv3 import MobileNetV3_Large
import torch
import os; os.environ["CUDA_VISIBLE_DEVICES"] = ''

net = MobileNetV3_Large()
x = torch.randn(2,3,224,224)
y = net(x)

trace_model = torch.jit.trace(net, (x, ))
trace_model.save("assets/mbv3.jit")

print('done')