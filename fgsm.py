'''fgsm攻击算法'''
import torch
from torch.autograd import Variable
from torchvision import models
import numpy as np
from torchvision import transforms,datasets
import torchvision.transforms as transforms
import os
from PIL import Image

modelName='resnet50'

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """ 
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


# 获取计算设备 默认是CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_transform =  transforms.Compose([
    # transforms.CenterCrop(180),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])
batch_size =1

image_path = '../data/Mstar/'
val_dataset = ImageFolderWithPaths(root=image_path + "test",
                                            transform=test_transform)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
class_names = val_dataset.classes
number = 0
a=0

checkpoint = torch.load('./checkpoint/%s.t7'%(modelName))
model = checkpoint['net']

transf = transforms.ToTensor()

for img, label,path in val_dataloader:
    orig_label = label
    a += 1
    orig = img.squeeze(0).squeeze(0).cpu().numpy()
    img = Variable(img.float(),requires_grad = False).cuda()
    orig_label0 = np.argmax(model(img).data.cpu().numpy())
    # 图像数据梯度可以获取
    img.requires_grad = True

    # 设置为不保存梯度值 自然也无法修改
    for param in model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam([img])
    loss_func = torch.nn.CrossEntropyLoss()
    '''设置迭代次数和步长'''
    epochs = 100
    e = 2/255.2
    target = Variable(torch.Tensor([float(orig_label0)]).to(device).long())

    for epoch in range(epochs):
        # 梯度清零
        optimizer.zero_grad()
        # forward + backward
        output = model(img)
        output = output.cuda()
        loss = -loss_func(output, target)
        label = np.argmax(output.data.cpu().numpy()[0])
        # 如果定向攻击成功
        if label != target :
            adv = img.data.cpu().numpy()[0]
            adv = (adv *0.5 ) + 0.5
            adv = adv * 255.0
            adv = np.clip(adv, 0, 255).astype(np.uint8)
            adv = adv.astype("float32")
            adv = adv.transpose(1, 2, 0)
            adv2=transf(adv)
            adv2 = (adv2 / 255.0 - 0.5) / 0.5
            adv2 = adv2.unsqueeze(0)
            adv2=adv2.cuda()
            output = model(adv2)
            label_t = np.argmax(output.data.cpu().numpy()[0])
            if target != label_t:
                break
            else:
                img.data = adv2.data
                continue
            
        # 反向传递 计算梯度
        loss.backward()
        img.data = torch.clamp(img.data - e * torch.sign(img.grad.data),-1,1)

    adv = img.data.cpu().numpy()[0]
    adv1 = adv.transpose(1, 2, 0)
    adv1 = transf(adv1).cuda()
    adv1 = adv1.unsqueeze(0)
    output = model(adv1)
    output = output.cuda()
    label1 = np.argmax(output.data.cpu().numpy()[0])
    adv = (adv *0.5 ) + 0.5
    adv = adv * 255.0  
    adv = np.reshape(adv,(100,100))
    im = Image.fromarray(adv).convert('L')
    adv = np.asarray(im).astype("float32")
    adv2 = transf(adv)
    adv2 = (adv2 / 255.0 - 0.5) / 0.5
    adv2 = adv2.unsqueeze(0)
    output = model(adv2.cuda())
    output = output.cuda()
    label2 = np.argmax(output.data.cpu().numpy()[0])
    if orig_label0 != label2:
        number = number + 1
        print('已攻击成功：{}张图片,攻击成功率：{}%'.format(number,100*number/a))
        new_path = path[0].replace("test", "fgsm\\%s"%(modelName))
        new_path2 = new_path.rsplit("\\", 1)[0]
        if not os.path.exists(new_path2):
            os.makedirs(new_path2)
        im.save(new_path)
exit()
        
