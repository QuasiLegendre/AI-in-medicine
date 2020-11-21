import torch
import torchvision

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

class DNN(nn.Module):
	def __init__(self, image_w, image_h):
		super(DNN, self).__init__()
		self.w = image_w
		self.h = image_h
		self.h_ln = nn.Linear(self.w * self.h, 128)
		self.out_ln = nn.Linear(128, 10)

	def forward(self, x):
		x = x.view(-1, self.w * self.h)
		x = F.relu(self.h_ln(x))
		x = self.out_ln(x)
		return x

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(15, 6))
    for idx in np.arange(10):
        ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],\
            probs[idx] * 100.0,\
            classes[labels[idx]]),\
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

if __name__ == '__main__':
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	data_file = '../../Datasets'
#matplotlib.use('Qt5Agg')

	transform = torchvision.transforms.Compose(\
				[torchvision.transforms.ToTensor(),\
				torchvision.transforms.Normalize((0.5,), (0.5,))])

	train_set = torchvision.datasets.FashionMNIST(data_file,\
				download=True,\
				train=True,\
				transform=transform)
	test_set = torchvision.datasets.FashionMNIST(data_file,\
				download=True,\
				train=False,\
				transform=transform)

	train_loader = torch.utils.data.DataLoader(train_set,\
				batch_size=64, shuffle=True, num_workers=10)

	test_loader =  torch.utils.data.DataLoader(test_set,\
				batch_size=10, shuffle=True, num_workers=10)
	 
	classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',\
			'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

	net = DNN(28, 28)
	net.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	
	running_loss = 0.0
	for epoch in range(50):

		for i, data in enumerate(train_loader, 0):
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			if i % 1000 == 0:
				print('training loss: '\
					+ str(running_loss/1000) + ' --- '\
					+ str(epoch * len(train_loader) + i))
			running_loss = 0.0
	
	print('Finished Training!')
	torch.save(net, 'DNN.pt')
	
	net.to(torch.device('cpu'))
	tl = iter(test_loader)
	test_inputs, test_labels = next(tl)	
	fig = plot_classes_preds(net, test_inputs, test_labels)
	fig.savefig('Res.png')
