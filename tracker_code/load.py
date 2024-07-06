import torch
import sys

checkpoint = torch.load('/root/autodl-tmp/STARK/checkpoints/train/stark_s/baseline/20240515/STARKS_ep0600.pth.tar', map_location='cpu')
# checkpoint = torch.load('/root/autodl-tmp/STARK/checkpoints/train/stark_s/baseline/20240523_decoder_0/STARKS_ep0500.pth.tar', map_location='cpu')
# checkpoint = torch.load('/root/autodl-tmp/Stark/checkpoints/train/stark_s/baseline/STARKS_ep0500.pth.tar', map_location='cpu')


# file = open('output.txt', 'w')
# sys.stdout = file
# print(checkpoint['net'])
# file.close


net = torch.load('/root/autodl-tmp/STARK/checkpoints/train/stark_s/baseline_step2/STARKS_ep0001.pth.tar', map_location='cpu') #encoder=0
# net = torch.load('/root/autodl-tmp/STARK/checkpoints/train/stark_s/baseline_step1/STARKS_ep0001.pth.tar', map_location='cpu')

# file1 = open('output1.txt', 'w')
# sys.stdout = file1
# # print(net['net'].values())
# print(net['net'])
# file1.close

pretrained_dict = checkpoint['net']
net_dict = net['net']
pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in net_dict}
net['net'].update(pretrained_dict)

# file2 = open('output2.txt', 'w')
# sys.stdout = file2
# print(net['net'])
# file2.close

torch.save(net, 'STARKS_ep0001.pth.tar')


# net = torch.load('/root/autodl-tmp/STARK/STARKS_ep0001.pth.tar', map_location='cpu')

# file1 = open('output3.txt', 'w')
# sys.stdout = file1
# # print(net['net'].values())
# print(net)
# file1.close


# net = torch.load('/root/autodl-tmp/STARK/checkpoints/train/stark_s/baseline_step2/STARKS_ep0003.pth.tar', map_location='cpu')

# net['epoch'] = 3

# # print(net['epoch'])

# torch.save(net, 'STARKS_ep0003.pth.tar')
# net.load_state_dict(net_dict)
# checkpoint_dict = {k:v for k, v in }

# print(checkpoint['net'].items())
# print(checkpoint['net'].keys())
# print(checkpoint['net'].values())
# print(net.keys())
# print(checkpoint['net'])