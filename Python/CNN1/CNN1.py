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

from torch.utils.tensorboard import SummaryWriter

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

class CNN(nn.Module):
	def __init__(self, image_w, image_h):
		super(CNN, self).__init__()
		self.w = image_w
		self.h = image_h
		self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.pool = nn.AvgPool2d(2)
		self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(8)
		self.conv3 = nn.Conv2d(8, 16, 3, stride=2)
		self.bn3 = nn.BatchNorm2d(16)
		self.fc1 = nn.Linear(16 * (self.w//8) * (self.h//8), 72)
		self.fc2 = nn.Linear(72, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.bn1(self.conv1(x))))
		x = self.pool(F.relu(self.bn2(self.conv2(x))))
		x = F.relu(self.bn3(self.conv3(x)))
		x = x.view(-1, 16 * (self.w//8) * (self.h//8))
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 0, 0.02)
		nn.init.constant_(m.bias.data, 0)
	elif classname.find('Linear') != -1:
		nn.init.normal_(m.weight.data, 0, 0.02)
		nn.init.constant_(m.bias.data, 0)


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images).to(torch.device('cpu'))
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
    images = images.to(torch.device('cpu'))
    labels = labels.to(torch.device('cpu')) 
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

	net = CNN(28, 28)
	net.to(device)
	net.apply(weights_init)

	writer = SummaryWriter('runs/fashion_mnist_experiment')
	dataiter = iter(train_loader)
	images, labels = dataiter.next()

	img_grid = torchvision.utils.make_grid(images)

	matplotlib_imshow(img_grid, one_channel=True)

	writer.add_image('four_fashion_mnist_images', img_grid)
#writer.add_graph(net, images)
#writer.close()

	beta1 = 0.5
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.001, betas = (beta1, 0.999))
	
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
				writer.add_scalar('training loss:',\
					running_loss/1000,\
					epoch * len(train_loader) + i)
				writer.add_figure('predictions vs. actuals',
							plot_classes_preds(net, inputs, labels),
							global_step=epoch * len(train_loader) + i)

			running_loss = 0.0	
	print('Finished Training!')
	torch.save(net, 'CNN.pt')
	
	net.to(torch.device('cpu'))
	tl = iter(test_loader)
	test_inputs, test_labels = next(tl)	
	fig = plot_classes_preds(net, test_inputs, test_labels)
	fig.savefig('Res.png')

	class_probs = []
	class_preds = []
	with torch.no_grad():
		for data in test_loader:
			images, labels = data
			output = net(images)
			class_probs_batch = [F.softmax(el, dim=0) for el in output]
			_, class_preds_batch = torch.max(output, 1)
			
			class_probs.append(class_probs_batch)
			class_preds.append(class_preds_batch)

	test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
	test_preds = torch.cat(class_preds)

	def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
		tensorboard_preds = test_preds == class_index
		tensorboard_probs = test_probs[:, class_index]

		writer.add_pr_curve(classes[class_index],
					tensorboard_preds,
					tensorboard_probs,
					global_step=global_step)
		writer.close()

	for i in range(len(classes)):
		add_pr_curve_tensorboard(i, test_probs, test_preds)
