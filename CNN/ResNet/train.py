import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets
import resnet18
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ModelTraining:
    def __init__(self, transform_train, transform_test, device, classes, model):
        self.classes = classes
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.device = device
        self.model = model
        self.classes = classes    

        if not self.device:
            print("CUDA is not vailable for traiaing and model will be trained on CPU")
        else:
            print("CUDA is available")    

    def datalaoder(self):

        valid_size = 10000
        batch_size = 8
       
        self.train_dataset = datasets.CIFAR10('data', train=True, download=True, transform = self.transform_train)
        self.test_dataset = datasets.CIFAR10('data', train=False, download=True, transform = self.transform_test)
        # splitting traina dn valid dataset     

        train_size = len(self.train_dataset) - valid_size
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.train_dataset, [train_size, valid_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size = batch_size, shuffle = True)
        self.test_loader = DataLoader(self.test_dataset, batch_size = batch_size, shuffle = False)
        #  Print dataset statistics
        print(f'Training samples: {len(self.train_dataset)}')
        print(f'Validation samples: {len(self.valid_dataset)}')
        print(f'Test samples: {len(self.test_dataset)}')
        print(f'Image shape: {self.train_dataset[0][0].shape}')
        print(f'Number of classes: {len(self.classes)}')

        # Print ResNet model summary
        print(self.model)

        

    def display_images(self):
        num_images = 10
        class_indices = {class_name: [] for class_name in self.classes}
        # Collect indices of each class
        for i, (_, label) in enumerate(self.train_dataset):
            class_indices[self.classes[label]].append(i)

        # Display images for each class
        fig, axs = plt.subplots(len(self.classes), num_images, figsize=(10, 10))
        for i, class_name in enumerate(self.classes):
            np.random.shuffle(class_indices[class_name])
            for j in range(num_images):
                idx = class_indices[class_name][j]
                img, _ = self.train_dataset[idx]
                axs[i, j].imshow(img.permute(1, 2, 0))
                axs[i, j].set_title(class_name)
                axs[i, j].axis('off')
        plt.tight_layout()
        plt.show()


    def train_model(self,):

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.001, weight_decay = 0.001, momentum= 0.999)
        n_epochs = 10
        valid_loss_min = np.Inf
        print("[INFO] training started..........")

        for epoch in range(n_epochs):
            train_loss = 0.0
            valid_loss = 0.0
            # print(self.model)

            self.model.train()

            for data, labels in self.train_loader:
                if self.device:
                    data, labels = data.cuda(), labels.cuda()
                # output = model.forward(data)
                # print(output)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()*data.size(0)

            self.model.eval()
            for data, labels in self.valid_loader:
                if self.device:
                    data, labels = data.cuda(), labels.cuda()
                output = self.model(data)
                loss = self.criterion(output, labels)
                valid_loss += loss.item()*data.size(0)

            train_loss = train_loss/len(self.train_loader.dataset)
            valid_loss = valid_loss/len(self.valid_loader.dataset)

            print('Epoch: {} \t Training Loss: {} \t Validation Loss: {}'.format(epoch, train_loss, valid_loss))

            if valid_loss <= valid_loss_min:
                print('Validation Loss decreased, Saving Model'.format(valid_loss_min, valid_loss))

                torch.save(self.model.state_dict(), 'CIFAR10.pt')
                valid_loss_min = valid_loss
                self.model.load_state_dict(torch.load('CIFAR10.pt'))

    def test(self):
        test_loss = 0.0
        correct_class = 0
        total_class = 0
        for data, labels in self.train_loader:
            if self.device:
                data, labels = data.cuda(), labels.cuda()
            output = self.model(data)
            loss = self.criterion(output, labels)
            test_loss += loss.item()*data.size(0)
            _,predict = torch.max(output, 1)
            total_class += labels.size(0)
            correct_class += (predict == labels).sum().item()
        test_loss = test_loss/len(self.test_loader.dataset)
        accuracy = 100*correct_class/total_class

        print('Test Loss: {:.6f}\nAccuracy: {:.2f}%'.format(test_loss, accuracy))          
                


if __name__ == "__main__":

    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    model = resnet18.ResNet18(10)
    device = torch.cuda.is_available()


    train = ModelTraining(transform_train, transform_test, device, classes, model)
    train.datalaoder()
    train.display_images()
    train.train_model()
    train.test()