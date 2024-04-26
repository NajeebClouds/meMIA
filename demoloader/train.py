import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

np.set_printoptions(threshold=np.inf)

from opacus import PrivacyEngine
from torch.optim import lr_scheduler

def GAN_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class model_training_location():
    def __init__(self, trainloader, testloader, model, device, use_DP, noise, norm, delta):
        self.use_DP = use_DP
        self.device = device
        self.delta = delta
        self.net = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader

        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        self.noise_multiplier, self.max_grad_norm = noise, norm
        
        if self.use_DP:
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                module=model,
                optimizer=self.optimizer,
                data_loader=self.trainloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
       
            print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))
            # privacy_engine.attach(self.optimizer)

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 75], 0.1)

    # Training
    def train(self):
        self.net.train()
        
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if isinstance(targets, list):
                targets = targets[0]

            if str(self.criterion) != "CrossEntropyLoss()":
                targets = torch.from_numpy(np.eye(self.num_classes)[targets]).float()
             
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # print(f" inputs size: {inputs.size()}, targets size: {targets.size()}")
          
            
            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if str(self.criterion) != "CrossEntropyLoss()":
                _, targets= targets.max(1)

            correct += predicted.eq(targets).sum().item()

        if self.use_DP:
            epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
            # epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(1e-5)
            print("\u03B5: %.3f \u03B4: 1e-5" % (epsilon))
                
        self.scheduler.step()

        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return 1.*correct/total


    def saveModel(self, path):
        torch.save(self.net.state_dict(), path)

    def get_noise_norm(self):
        return self.noise_multiplier, self.max_grad_norm

    def test(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                if isinstance(targets, list):
                    targets = targets[0]
                if str(self.criterion) != "CrossEntropyLoss()":
                    targets = torch.from_numpy(np.eye(self.num_classes)[targets]).float()

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                if str(self.criterion) != "CrossEntropyLoss()":
                    _, targets= targets.max(1)

                correct += predicted.eq(targets).sum().item()

            print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

        return 1.*correct/total




class model_training():
    def __init__(self, trainloader, testloader, model, device, use_DP, noise, norm, delta):
        self.use_DP = use_DP
        self.device = device
        self.delta = delta
        self.net = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader

        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        self.noise_multiplier, self.max_grad_norm = noise, norm
        
        if self.use_DP:
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                module=model,
                optimizer=self.optimizer,
                data_loader=self.trainloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
        
            print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))
            # privacy_engine.attach(self.optimizer)

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)

    # Training
    def train(self):
        self.net.train()
        
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if isinstance(targets, list):
                targets = targets[0]

            if str(self.criterion) != "CrossEntropyLoss()":
                targets = torch.from_numpy(np.eye(self.num_classes)[targets]).float()
             
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if str(self.criterion) != "CrossEntropyLoss()":
                _, targets= targets.max(1)

            correct += predicted.eq(targets).sum().item()

        if self.use_DP:
            epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
            # epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(1e-5)
            print("\u03B5: %.3f \u03B4: 1e-5" % (epsilon))
                
        self.scheduler.step()

        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return 1.*correct/total


    def saveModel(self, path):
        torch.save(self.net.state_dict(), path)

    def get_noise_norm(self):
        return self.noise_multiplier, self.max_grad_norm

    def test(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                if isinstance(targets, list):
                    targets = targets[0]
                if str(self.criterion) != "CrossEntropyLoss()":
                    targets = torch.from_numpy(np.eye(self.num_classes)[targets]).float()

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                if str(self.criterion) != "CrossEntropyLoss()":
                    _, targets= targets.max(1)

                correct += predicted.eq(targets).sum().item()

            print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

        return 1.*correct/total


    def saveModel(self, path_d, path_g):
        torch.save(self.model_discriminator.state_dict(), path_d)
        torch.save(self.model_generator.state_dict(), path_g)

