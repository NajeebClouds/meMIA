import os
import glob
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sklearn.preprocessing import MinMaxScaler
from torch.nn.functional import normalize
import base64
from torchmetrics.classification import BinaryConfusionMatrix
import pandas as pd
import matplotlib.pyplot as plt
import time
# from demoloader.dataloader import *


np.set_printoptions(threshold=np.inf)

from opacus import PrivacyEngine
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, roc_auc_score


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti  # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    # if tempBool:
        # print("Elapsed time: %f seconds." % tempTimeInterval)
    return tempTimeInterval

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)  # The first call to toc() after this will measure from here


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

# class shadow():
#     def __init__(self, trainloader, testloader, model, device, use_DP, noise, norm, loss, optimizer, delta):
#         self.delta = delta
#         self.use_DP = use_DP
#         self.device = device
#         self.model = model.to(self.device)
#         self.trainloader = trainloader
#         self.testloader = testloader

#         self.criterion = loss
#         self.optimizer = optimizer

#         self.noise_multiplier, self.max_grad_norm = noise, norm
        
#         if self.use_DP:
#             self.privacy_engine = PrivacyEngine()
#             self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
#                 module=self.model,
#                 optimizer=self.optimizer,
#                 data_loader=self.trainloader,
#                 noise_multiplier=self.noise_multiplier,
#                 max_grad_norm=self.max_grad_norm,
#             )
#             # self.model = module_modification.convert_batchnorm_modules(self.model)
#             # inspector = DPModelInspector()
#             # inspector.validate(self.model)
#             # privacy_engine = PrivacyEngine(
#             #     self.model,
#             #     batch_size=batch_size,
#             #     sample_size=len(self.trainloader.dataset),
#             #     alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
#             #     noise_multiplier=self.noise_multiplier,
#             #     max_grad_norm=self.max_grad_norm,
#             #     secure_rng=False,
#             # )
#             print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))
#             # privacy_engine.attach(self.optimizer)

#         self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)

#     # Training
#     def train(self):
#         # this line is used to set the mode of the mode of the mode for training
#         # means during the backprop process the gradient will be calculated and
#         # then the optimizer will update the weight based on the gradients calculated previously
#         # by the back prop
#         # Else if set to eval() mode then no gradients will be calculated and no wieghts will be
#         # updated
#         self.model.train()
        
#         train_loss = 0
#         correct = 0
#         total = 0
#         # In each epoch the mode looks at all the training samples
#         # before this, the training samples are divided into mini-batches of size for example 64
#         # dataloader is used to turn the training samples into iterable mini-batches
#         # for example if train size is (105,4) and batch size is 10, then mini_batches = 11
#         # the first batch inputs (10,4) is taken and fed to the model, get corresponding 
#         # predictions, compare it with targets, calculate the loss per batch,
#         # propagare the loss per batch and update the wieghts using optim.step
#         # pick the next batch, clear prevoous gradient info and repeat the process for 11 times
#         # once  all the bathces are shown to the model then start the next epoch
 
#         for batch_idx, (inputs, targets) in enumerate(self.trainloader):

#             inputs, targets = inputs.to(self.device), targets.to(self.device)

#             self.optimizer.zero_grad()
#             outputs = self.model(inputs)

#             loss = self.criterion(outputs, targets)
#             loss.backward()
#             self.optimizer.step()

#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
           

#         if self.use_DP:
#             epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
#             # epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(1e-5)
#             print("\u03B5: %.3f \u03B4: 1e-5" % (epsilon))
            
#         # after the end of each epoch, update the learning rate     
#         self.scheduler.step()

#         print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

#         return 1.*correct/total


#     def saveModel(self, path):
#         torch.save(self.model.state_dict(), path)

#     def get_noise_norm(self):
#         return self.noise_multiplier, self.max_grad_norm

#     def test(self):
#         self.model.eval()
#         test_loss = 0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for inputs, targets in self.testloader:
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 outputs = self.model(inputs) # outputs = [64, 10], targets = [64]

#                 loss = self.criterion(outputs, targets)

#                 test_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets).sum().item()

#             print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

#         return 1.*correct/total

class shadow():
    def __init__(self, trainloader, testloader, model, device, use_DP, noise, norm, loss, optimizer, delta):
        self.delta = delta
        self.use_DP = use_DP
        self.device = device
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader

        self.criterion = loss
        self.optimizer = optimizer

        self.noise_multiplier, self.max_grad_norm = noise, norm
        
        if self.use_DP:
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.trainloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
            # self.model = module_modification.convert_batchnorm_modules(self.model)
            # inspector = DPModelInspector()
            # inspector.validate(self.model)
            # privacy_engine = PrivacyEngine(
            #     self.model,
            #     batch_size=batch_size,
            #     sample_size=len(self.trainloader.dataset),
            #     alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            #     noise_multiplier=self.noise_multiplier,
            #     max_grad_norm=self.max_grad_norm,
            #     secure_rng=False,
            # )
            print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))
            # privacy_engine.attach(self.optimizer)

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)
        # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)
        

    # Training
    def train(self):
        self.model.train()
        
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if self.use_DP:
            epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
            # epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(1e-5)
            print("\u03B5: %.3f \u03B4: 1e-5" % (epsilon))
                
        self.scheduler.step()

        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return 1.*correct/total


    def saveModel(self, path):
        torch.save(self.model.state_dict(), path)

    def get_noise_norm(self):
        return self.noise_multiplier, self.max_grad_norm

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

        return 1.*correct/total

    
# class model_training():
#     def __init__(self, trainloader, testloader, model, device, use_DP, noise, norm, delta):
#         self.use_DP = use_DP
#         self.device = device
#         self.delta = delta
#         self.net = model.to(self.device)
#         self.trainloader = trainloader
#         self.testloader = testloader

#         if self.device == 'cuda':
#             self.net = torch.nn.DataParallel(self.net)
#             cudnn.benchmark = True

#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.SGD(self.net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-7)

#         self.noise_multiplier, self.max_grad_norm = noise, norm
        
#         if self.use_DP:
#             self.privacy_engine = PrivacyEngine()
#             self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
#                 module=model,
#                 optimizer=self.optimizer,
#                 data_loader=self.trainloader,
#                 noise_multiplier=self.noise_multiplier,
#                 max_grad_norm=self.max_grad_norm,
#             )
#             # self.net = module_modification.convert_batchnorm_modules(self.net)
#             # inspector = DPModelInspector()
#             # inspector.validate(self.net)
#             # privacy_engine = PrivacyEngine(
#             #     self.net,
#             #     batch_size=64,
#             #     sample_size=len(self.trainloader.dataset),
#             #     alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
#             #     noise_multiplier=self.noise_multiplier,
#             #     max_grad_norm=self.max_grad_norm,
#             #     secure_rng=False,
#             # )
#             print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))
#             # privacy_engine.attach(self.optimizer)

#         # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)

#     # Training
#     def train(self):
#         self.net.train()
        
#         train_loss = 0
#         correct = 0
#         total = 0
        
#         for batch_idx, (inputs, targets) in enumerate(self.trainloader):
#             if isinstance(targets, list):
#                 targets = targets[0]

#             if str(self.criterion) != "CrossEntropyLoss()":
#                 targets = torch.from_numpy(np.eye(self.num_classes)[targets]).float()
             
#             inputs, targets = inputs.to(self.device), targets.to(self.device)
#             self.optimizer.zero_grad()
#             outputs = self.net(inputs)

#             loss = self.criterion(outputs, targets)
#             loss.backward()
#             self.optimizer.step()

#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             if str(self.criterion) != "CrossEntropyLoss()":
#                 _, targets= targets.max(1)

#             correct += predicted.eq(targets).sum().item()

#         if self.use_DP:
#             epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
#             # epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(1e-5)
#             print("\u03B5: %.3f \u03B4: 1e-5" % (epsilon))
                
#         # self.scheduler.step()

#         print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

#         return 1.*correct/total


#     def saveModel(self, path):
#         torch.save(self.net.state_dict(), path)

#     def get_noise_norm(self):
#         return self.noise_multiplier, self.max_grad_norm

#     def test(self):
#         self.net.eval()
#         test_loss = 0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for inputs, targets in self.testloader:
#                 if isinstance(targets, list):
#                     targets = targets[0]
#                 if str(self.criterion) != "CrossEntropyLoss()":
#                     targets = torch.from_numpy(np.eye(self.num_classes)[targets]).float()

#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 outputs = self.net(inputs)

#                 loss = self.criterion(outputs, targets)

#                 test_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 total += targets.size(0)
#                 if str(self.criterion) != "CrossEntropyLoss()":
#                     _, targets= targets.max(1)

#                 correct += predicted.eq(targets).sum().item()

#             print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

#         return 1.*correct/total
     
     
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
            # self.net = module_modification.convert_batchnorm_modules(self.net)
            # inspector = DPModelInspector()
            # inspector.validate(self.net)
            # privacy_engine = PrivacyEngine(
            #     self.net,
            #     batch_size=64,
            #     sample_size=len(self.trainloader.dataset),
            #     alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            #     noise_multiplier=self.noise_multiplier,
            #     max_grad_norm=self.max_grad_norm,
            #     secure_rng=False,
            # )
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

class model_training_shadow():
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
            # self.net = module_modification.convert_batchnorm_modules(self.net)
            # inspector = DPModelInspector()
            # inspector.validate(self.net)
            # privacy_engine = PrivacyEngine(
            #     self.net,
            #     batch_size=64,
            #     sample_size=len(self.trainloader.dataset),
            #     alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            #     noise_multiplier=self.noise_multiplier,
            #     max_grad_norm=self.max_grad_norm,
            #     secure_rng=False,
            # )
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
    
class distillation_training():
    def __init__(self, PATH, trainloader, testloader, model, teacher, device, optimizer, T, alpha):
        self.device = device
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader

        self.PATH = PATH
        self.teacher = teacher.to(self.device)
        self.teacher.load_state_dict(torch.load(self.PATH))
        self.teacher.eval()

        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.optimizer = optimizer

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)

        self.T = T
        self.alpha = alpha

    def distillation_loss(self, y, labels, teacher_scores, T, alpha):
        loss = self.criterion(F.log_softmax(y/T, dim=1), F.softmax(teacher_scores/T, dim=1))
        loss = loss * (T*T * alpha) + F.cross_entropy(y, labels) * (1. - alpha)
        return loss

    def train(self):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, [targets, _]) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            teacher_output = self.teacher(inputs)
            teacher_output = teacher_output.detach()
    
            loss = self.distillation_loss(outputs, targets, teacher_output, T=self.T, alpha=self.alpha)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

        self.scheduler.step()
        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return 1.*correct/total

    def saveModel(self, path):
        torch.save(self.model.state_dict(), path)

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, [targets, _] in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

        return 1.*correct/total
    
class RNN_attack_for_blackbox():
    # attack_model need to pass RNN mode here
    def __init__(self, SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, target_model, shadow_model, rnn_model, device):
        self.device = device

        self.TARGET_PATH = TARGET_PATH
        self.SHADOW_PATH = SHADOW_PATH
        self.ATTACK_SETS = ATTACK_SETS

        self.target_model = target_model.to(self.device)
        self.shadow_model = shadow_model.to(self.device)

        
        print( 'self.TARGET_PATH: %s' % self.TARGET_PATH)
        # exit()
        
        self.target_model.load_state_dict(torch.load(self.TARGET_PATH)) # load TM here
        self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH)) # load SM here

        self.target_model.eval()
        self.shadow_model.eval()

        self.attack_train_loader = attack_train_loader # batch of 64
        self.attack_test_loader = attack_test_loader

        self.rnn_model = rnn_model.to(self.device)
        torch.manual_seed(0)
        self.rnn_model.apply(weights_init)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.rnn_model.parameters(), lr=1e-2)
        # self.optimizer = optim.Adam(self.rnn_model.parameters(), lr=1e-2) workes for 256,128,64
        

    def _get_data(self, model, inputs, targets):
        result = model(inputs)
        
        scaler = MinMaxScaler()
        output_sorted, _ = torch.sort(result, descending=True) #commenting out sorting for RNN 
        output_sorted =  F.softmax(output_sorted, dim=1)
        # output  = result;
        output = F.softmax(result, dim=1) #Now data is PVs becuse of the spftmax and its unsorted
        
        _, predicts = result.max(1)
    
        output_ranks = torch.argsort(output, axis=0)/9
        
        prediction = predicts.eq(targets).float()
        
        # prediction = []
        # for predict in predicts:
        #     prediction.append([1,] if predict else [0,])

        # prediction = torch.Tensor(prediction)

        # final_inputs = torch.cat((results, prediction), 1)
        # print(final_inputs.shape)

        return output_sorted, prediction.unsqueeze(-1)

    def prepare_dataset(self):
        with open(self.ATTACK_SETS + "train.p", "wb") as f:
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if inputs.size()[0] == 64:
            
                    output, prediction = self._get_data(self.shadow_model, inputs, targets)
                    # output = output.cpu().detach().numpy()
                    # output size: torch.Size([64, 10]), prediction size: torch.Size([64, 1]), members: torch.Size([64]), batch of 64
                    # prediction: a specific sample in the batch is predicted correct (1) or predicted wrong (0)
                    # output: Not PVs but raw 10 logits (based on the number of classes)
                    # print(f"output size: {output.shape}, prediction size: {prediction.shape}, members: {members.shape}")
                    # print(output)
                    # print(prediction)
                    # print(members)
                    # break;
                    pickle.dump((output, prediction, members), f)
                else:
                    print("skipping: ",inputs.size()[0])


        print("Finished Saving Train Dataset")
        

        with open(self.ATTACK_SETS + "test.p", "wb") as f:
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if inputs.size()[0] == 64:
                    
                    output, prediction = self._get_data(self.target_model, inputs, targets)
                    # output = output.cpu().detach().numpy()
                
                    pickle.dump((output, prediction, members), f)
                else:
                     print("test data skipping: ",inputs.size()[0])

        print("Finished Saving Test Dataset")

    def train(self, epoch, result_path):
        self.rnn_model.train()
        batch_idx = 1
        train_loss = 0
        correct = 0
        total = 0

        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []

        final_result = []

        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while(True):
                try:
                    output, prediction, members = pickle.load(f)
                    output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)
                
                   
        
                    # print(f"output in training of LSTM: {output.size()}"); # output [64, 10]
                    
                    # results = self.rnn_model(output, prediction)  
                    # print(f"output first 2: {output[:2]}")
                    results = self.rnn_model(output) #only using output
                    # print(f"results size in training of LSTM: {results.size()}"); # results [64, 10]
                    # print(f"rnn out: {results[:2]}")
                    results = F.softmax(results, dim=1)
                    # print(f"rnn out softmax: {results[:2]}")
                    self.optimizer.zero_grad()
                    losses = self.criterion(results, members)
                    # print(f"losses: {losses.item()}")
                    
                    losses.backward()
                    self.optimizer.step()

                    train_loss += losses.item()
                    _, predicted = results.max(1)
                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    if epoch:
                        final_train_gndtrth.append(members)
                        final_train_predict.append(predicted)
                        final_train_probabe.append(results[:, 1])

                    batch_idx += 1
                except EOFError:
                    break

        if epoch:
            final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
            final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
            final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

            train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
            train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

            final_result.append(train_f1_score)
            final_result.append(train_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
            
            print("Saved rnn_model Train Ground Truth and Predict Sets")
            print("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

        final_result.append(1.*correct/total)
        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return final_result

    def test(self, epoch, result_path):
        self.rnn_model.eval()
        batch_idx = 1
        correct = 0
        total = 0

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []

        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while(True):
                    try:
                        output, prediction, members = pickle.load(f)
                        output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

                        results = self.rnn_model(output)
                        results = F.softmax(results, dim=1)
                        _, predicted = results.max(1)
                        total += members.size(0)
                        correct += predicted.eq(members).sum().item()
                        

                        if epoch:
                            final_test_gndtrth.append(members)
                            final_test_predict.append(predicted)
                            final_test_probabe.append(results[:, 1])

                        batch_idx += 1
                    except EOFError:
                        break

        if epoch:
            final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
            final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
            final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

            test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
            test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

            final_result.append(test_f1_score)
            final_result.append(test_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

            print("Saved rnn_mode Test Ground Truth and Predict Sets")
            print("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))

        final_result.append(1.*correct/total)
        print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

        return final_result

    def delete_pickle(self):
        train_file = glob.glob(self.ATTACK_SETS +"train.p")
        for trf in train_file:
            os.remove(trf)

        test_file = glob.glob(self.ATTACK_SETS +"test.p")
        for tef in test_file:
            os.remove(tef)

    def saveModel(self, path):
        torch.save(self.rnn_model.state_dict(), path)
        
class attack_for_blackbox():
    def __init__(self, SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, target_model, shadow_model, attack_model, device):
        self.device = device

        self.TARGET_PATH = TARGET_PATH
        self.SHADOW_PATH = SHADOW_PATH
        self.ATTACK_SETS = ATTACK_SETS

        self.target_model = target_model.to(self.device)
        self.shadow_model = shadow_model.to(self.device)

        
        print( 'self.TARGET_PATH: %s' % self.TARGET_PATH)
        # exit()
        
        self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
        self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH))

        self.target_model.eval()
        self.shadow_model.eval()

        self.attack_train_loader = attack_train_loader
        self.attack_test_loader = attack_test_loader

        self.attack_model = attack_model.to(self.device)
        torch.manual_seed(0)
        self.attack_model.apply(weights_init)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-5)

    def _get_data(self, model, inputs, targets):
        result = model(inputs)
        
        output, _ = torch.sort(result, descending=True)
        # results = F.softmax(results[:,:5], dim=1)
        _, predicts = result.max(1)

        prediction = predicts.eq(targets).float()
        
        # prediction = []
        # for predict in predicts:
        #     prediction.append([1,] if predict else [0,])

        # prediction = torch.Tensor(prediction)

        # final_inputs = torch.cat((results, prediction), 1)
        # print(final_inputs.shape)

        return output, prediction.unsqueeze(-1)

    def prepare_dataset(self):
        with open(self.ATTACK_SETS + "train.p", "wb") as f:
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(self.shadow_model, inputs, targets)
                # output = output.cpu().detach().numpy()
            
                pickle.dump((output, prediction, members), f)

        print("Finished Saving Train Dataset")

        with open(self.ATTACK_SETS + "test.p", "wb") as f:
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(self.target_model, inputs, targets)
                # output = output.cpu().detach().numpy()
            
                pickle.dump((output, prediction, members), f)

        print("Finished Saving Test Dataset")

    def train(self, epoch, result_path):
        self.attack_model.train()
        batch_idx = 1
        train_loss = 0
        correct = 0
        total = 0

        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []

        final_result = []

        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while(True):
                try:
                    output, prediction, members = pickle.load(f)
                    output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

                    results = self.attack_model(output, prediction)
                    results = F.softmax(results, dim=1)

                    losses = self.criterion(results, members)
                    losses.backward()
                    self.optimizer.step()

                    train_loss += losses.item()
                    _, predicted = results.max(1)
                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    if epoch:
                        final_train_gndtrth.append(members)
                        final_train_predict.append(predicted)
                        final_train_probabe.append(results[:, 1])

                    batch_idx += 1
                except EOFError:
                    break

        if epoch:
            final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
            final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
            final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

            train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
            train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

            final_result.append(train_f1_score)
            final_result.append(train_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
            
            print("Saved Attack Train Ground Truth and Predict Sets")
            print("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

        final_result.append(1.*correct/total)
        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return final_result

    def test(self, epoch, result_path):
        self.attack_model.eval()
        batch_idx = 1
        correct = 0
        total = 0

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []

        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while(True):
                    try:
                        output, prediction, members = pickle.load(f)
                        output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

                        results = self.attack_model(output, prediction)
                        _, predicted = results.max(1)
                        total += members.size(0)
                        correct += predicted.eq(members).sum().item()
                        results = F.softmax(results, dim=1)

                        if epoch:
                            final_test_gndtrth.append(members)
                            final_test_predict.append(predicted)
                            final_test_probabe.append(results[:, 1])

                        batch_idx += 1
                    except EOFError:
                        break

        if epoch:
            final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
            final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
            final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

            test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
            test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

            final_result.append(test_f1_score)
            final_result.append(test_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

            print("Saved Attack Test Ground Truth and Predict Sets")
            print("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))

        final_result.append(1.*correct/total)
        print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

        return final_result

    def delete_pickle(self):
        train_file = glob.glob(self.ATTACK_SETS +"train.p")
        for trf in train_file:
            os.remove(trf)

        test_file = glob.glob(self.ATTACK_SETS +"test.p")
        for tef in test_file:
            os.remove(tef)

    def saveModel(self, path):
        torch.save(self.attack_model.state_dict(), path)


class attack_for_blackbox_com():
    def __init__(self, SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, target_model, shadow_model, attack_model, device):
        self.device = device

        self.TARGET_PATH = TARGET_PATH
        self.SHADOW_PATH = SHADOW_PATH
        self.ATTACK_SETS = ATTACK_SETS

        self.target_model = target_model.to(self.device)
        self.shadow_model = shadow_model.to(self.device)

        
        print( 'self.TARGET_PATH: %s' % self.TARGET_PATH)
        # exit()
        
        self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
        self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH))

        self.target_model.eval()
        self.shadow_model.eval()

        self.attack_train_loader = attack_train_loader
        self.attack_test_loader = attack_test_loader

        self.attack_model = attack_model.to(self.device)
        torch.manual_seed(0)
        self.attack_model.apply(weights_init)

        self.criterion = nn.CrossEntropyLoss()
        # fff
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-5)

    def _get_data(self, model, inputs, targets):
        result = model(inputs)
        
        output, _ = torch.sort(result, descending=True)
        # results = F.softmax(results[:,:5], dim=1)
        _, predicts = result.max(1)

        prediction = predicts.eq(targets).float()
        
        # prediction = []
        # for predict in predicts:
        #     prediction.append([1,] if predict else [0,])

        # prediction = torch.Tensor(prediction)

        # final_inputs = torch.cat((results, prediction), 1)
        # print(final_inputs.shape)

        return output, prediction.unsqueeze(-1)

    def prepare_dataset(self):
        with open(self.ATTACK_SETS + "train.p", "wb") as f:
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if inputs.size()[0] == 64:
            
                    output, prediction = self._get_data(self.shadow_model, inputs, targets)
                    # output = output.cpu().detach().numpy()
                    # output size: torch.Size([64, 10]), prediction size: torch.Size([64, 1]), members: torch.Size([64]), batch of 64
                    # prediction: a specific sample in the batch is predicted correct (1) or predicted wrong (0)
                    # output: Not PVs but raw 10 logits (based on the number of classes)
                    # print(f"output size: {output.shape}, prediction size: {prediction.shape}, members: {members.shape}")
                    # print(output)
                    # print(prediction)
                    # print(members)
                    # break;
                    pickle.dump((output, prediction, members, targets), f)
                else:
                    print("skipping: ",inputs.size()[0])


        print("Finished Saving Train Dataset")
    

        with open(self.ATTACK_SETS + "test.p", "wb") as f:
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if inputs.size()[0] == 64:
                    
                    output, prediction = self._get_data(self.target_model, inputs, targets)
                    # output = output.cpu().detach().numpy()
                
                    pickle.dump((output, prediction, members, targets), f)
                else:
                     print("test data skipping: ",inputs.size()[0])
        
    def prepare_dataset_mul(self, num_classes):
        batch_size = 8
        # read the whole attack_train_loader batche by batch 
        # put the samples into corresponding buckets
        
        # transform each k-bucket into batches using dataloader
        # loade from dataloader, get the corresponing predictions
        # save into train_i.p file
        
    
        #! Traing Data class buckets 
        with torch.no_grad():
            counter=0
            print(f"classes: {num_classes}")
            for class_name in range(num_classes):
                # class_name = 29
                output_coll =  torch.empty((0, num_classes))
                predictions_coll = torch.empty((0, 1))
                members_coll = torch.empty((0, 1))
                targets_Coll = torch.empty((0, 1))
            
                file_path = self.ATTACK_SETS + f"_train_{class_name}.p"
                # print(f"new path: {file_path}")
                
                counter = 0
                with open(file_path, "wb") as f:
                    # This loop will iterate over all batches and put samples in their corresponding class file
                    for inputs, targets, members in self.attack_train_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        if 32 == 32:
                            counter+=1
                            # Assuming _get_data returns output and prediction based on the class
                            output, prediction = self._get_data(self.shadow_model, inputs, targets)
                            # print(f"output : {output.size()}")
                            # exit()
                            # print(f"inputs: {inputs[0]} \n\nand targets: {targets[0]}")
                            # print(f"CUDA sumary: {torch.cuda.memory_summary()}")
                            # print(f"allocated memory: {torch.cuda.memory_allocated()}")
                            output, prediction, targets  = output.cpu(), prediction.cpu(), targets.cpu()
                            # Find the indices where the value is class_name
                            # print(f"type of output: {type(output)}, device: {output.device}")
                            # print(f"type of prediction: {type(prediction)}, device: {prediction.device}")
                            # print(f"type of targets: {type(targets)}, device: {targets.device}")
                            # exit()
                            # print(f"targets: {targets}\n class_name: {class_name}")
                            indices = torch.where(targets == class_name)[0]
                            # print(f"indices in the batch: {indices}")
                            # exit()
                            # print(f"output_coll device: {output_coll.device}, output device: {output.device}")
                            output_coll = torch.vstack((output_coll, output[indices]))
                            # print(f"output_coll: {output_coll.size()}")
                            predictions_coll = torch.vstack((predictions_coll, prediction[indices]))
                            # print(f"targets_Coll size: {targets_Coll.size()}, targets size: {targets[indices].unsqueeze(1).size()}")
                            targets_Coll = torch.vstack((targets_Coll, targets[indices].unsqueeze(1)))
                            members_coll = torch.vstack((members_coll, members[indices].unsqueeze(1)))
                            
                            
                            
                        
                            
                        else:
                            print("skipping: ", inputs.size()[0])
                        
                        del inputs
                        del targets
                        torch.cuda.empty_cache()
                    
                    # print(f"Class {class_name} information")
                    # print(f"output_coll class {class_name}: {output_coll.size()} and size: {output_coll.size()[0]}")
                    # print(f"predictions: {predictions_coll.size()}")
                    # print(f"members: {members_coll.size()}")
                    # print(f"targets: {targets_Coll.size()}")
                    # exit()
                    # print(f"counter: {counter}")
            
                    # save in class_i file 
                    # attack_train = (output_coll, predictions_coll.squeeze(), members_coll.squeeze(), targets_Coll.squeeze())
                    attack_train = []
                    for i in range(output_coll.size()[0]):
                        attack_train.append((output_coll[i], predictions_coll[i].item(), members_coll[i].item(), targets_Coll[i].item()))
                        
                    # print(f"output_coll: {attack_train[0].shape}, predictions_coll : {attack_train[1].shape}, members_coll: {attack_train[2].shape}, targets_coll : {attack_train[3].shape}")
                    # print(f"attack_train size: {attack_train[0]}, len: {len(attack_train)}")
                    
                    # get track of the dimension of the dataset and append for later use
                    attack_trainloader = torch.utils.data.DataLoader(attack_train, batch_size=batch_size, shuffle=True, num_workers=2)
                    for output_coll, predictions_coll, members_coll, targets_Coll  in attack_trainloader:
                        # print(f"len of batch output_coll: {len(output_coll)}, and size: {output_coll.shape}")
                        if output_coll.size()[0] == batch_size:
                            pickle.dump((output_coll, predictions_coll, members_coll, targets_Coll), f)
                        # output, prediction, members = pickle.load(f)
                        else:
                            print(f"skipping the last {output_coll.size()[0]} samples")
                     
            #     exit()
            # exit()
        print(f"Finished Saving {num_classes} Train Dataset")   
            
            
        
        #! Test Data class buckets 
        with torch.no_grad():
            counter=0
            print(f"classes: {num_classes}")
            for class_name in range(num_classes):
                output_coll =  torch.empty((0, num_classes))
                predictions_coll = torch.empty((0, 1))
                members_coll = torch.empty((0, 1))
                targets_Coll = torch.empty((0, 1))
            
                file_path = self.ATTACK_SETS + f"_test_{class_name}.p"
                # print(f"new path: {file_path}")
                
                counter = 0
            
                with open(file_path, "wb") as f:
                    for inputs, targets, members in self.attack_test_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        
                        if 32 == 32:
                            counter+=1
                            output, prediction = self._get_data(self.target_model, inputs, targets)
                            # output = output.cpu().detach().numpy()
                            
                            
                            # print(f"output : {output.size()}")
                            # exit()
                            # print(f"inputs: {inputs[0]} \n\nand targets: {targets[0]}")
                            # print(f"CUDA sumary: {torch.cuda.memory_summary()}")
                            # print(f"allocated memory: {torch.cuda.memory_allocated()}")
                            output, prediction, targets  = output.cpu(), prediction.cpu(), targets.cpu()
                            # Find the indices where the value is class_name
                            # print(f"type of output: {type(output)}, device: {output.device}")
                            # print(f"type of prediction: {type(prediction)}, device: {prediction.device}")
                            # print(f"type of targets: {type(targets)}, device: {targets.device}")
                            # exit()
                            indices = torch.where(targets == class_name)[0]
                            # print(f"indices: {indices}")
                            # exit()
                            # print(f"output_coll device: {output_coll.device}, output device: {output.device}")
                            output_coll = torch.vstack((output_coll, output[indices]))
                            # print(f"output_coll: {output_coll.size()}")
                            predictions_coll = torch.vstack((predictions_coll, prediction[indices]))
                            # print(f"targets_Coll size: {targets_Coll.size()}, targets size: {targets[indices].unsqueeze(1).size()}")
                            targets_Coll = torch.vstack((targets_Coll, targets[indices].unsqueeze(1)))
                            members_coll = torch.vstack((members_coll, members[indices].unsqueeze(1)))
                            
                            # pickle.dump((output, prediction, members), f)
                        else:
                            print("test data skipping: ",inputs.size()[0])
                        
                        del inputs
                        del targets
                        torch.cuda.empty_cache()
                    
                    # print(f"Class {class_name} information Training")
                    # print(f"output_coll class {class_name}: {output_coll.size()} and size: {output_coll.size()[0]}")
                    # print(f"predictions: {predictions_coll.size()}")
                    # print(f"members: {members_coll.size()}")
                    # print(f"targets: {targets_Coll.size()}")
                    
            
                    # save in class_i file 
                    # attack_train = (output_coll, predictions_coll.squeeze(), members_coll.squeeze(), targets_Coll.squeeze())
                    attack_train = []
                    for i in range(output_coll.size()[0]):
                        attack_train.append((output_coll[i], predictions_coll[i].item(), members_coll[i].item(), targets_Coll[i].item()))
                        
                    # print(f"output_coll: {attack_train[0].shape}, predictions_coll : {attack_train[1].shape}, members_coll: {attack_train[2].shape}, targets_coll : {attack_train[3].shape}")
                    # print(f"attack_test size: {attack_train[0]}, len: {len(attack_train)}")
                    
                    # get track of the dimension of the dataset and append for later use
                    attack_testloader = torch.utils.data.DataLoader(attack_train, batch_size=batch_size, shuffle=True, num_workers=2)
                    for output_coll, predictions_coll, members_coll, targets_Coll  in attack_testloader:
                        # print(f"len of batch output_coll: {len(output_coll)}, and size: {output_coll.shape}")
                        if output_coll.size()[0] == batch_size:
                            pickle.dump((output_coll, predictions_coll, members_coll, targets_Coll), f)
                        # output, prediction, members = pickle.load(f)
                        else:
                            print(f"skipping the last {output_coll.size()[0]} samples")
                        
                
         
        print(f"Finished Saving {num_classes} Test Dataset")   
        # exit()

    def train(self, epoch, result_path, result_path_csv):
        self.attack_model.train()
        batch_idx = 1
        train_loss = 0
        correct = 0
        prec = 0
        recall = 0
        total = 0
        bcm = BinaryConfusionMatrix().to(self.device)
        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []

        final_result = []

        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while(True):
                try:
                    output, prediction, members, targets = pickle.load(f)
                    output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)
                    # print(f"1-output: {output.size(), prediction.size(), members.size()}")
                    # print(f"1-output output: {type(output),output.dtype, type(prediction), prediction.dtype, members.dtype}")
                    # exit()
                    
                    results = self.attack_model(output, prediction, targets)
                    
                    results = F.softmax(results, dim=1)
                    
                    
                    self.optimizer.zero_grad()
                    
                    losses = self.criterion(results, members)
                    losses.backward()
                    self.optimizer.step()

                    train_loss += losses.item()
                    _, predicted = results.max(1)
                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()
                    
                    # print(f"correctly predicted member and non-members: {predicted.eq(members).sum().item()} out of :{members.size(0)}")
                   
                    # print(f"members type: {type(members)}, device: {members.get_device()}, predicted: {type(predicted)}, device: {predicted.get_device()}")
                    conf_mat = bcm(predicted, members)
                    
                    prec += conf_mat[1,1]/torch.sum(conf_mat[:,-1])    
                    recall+=conf_mat[1,1]/torch.sum(conf_mat[-1,:])
                    # print(conf_mat)
                    # print(f"correct: {torch.sum(torch.diagonal(conf_mat, 0))}")
                    # print(f"last col sum: {torch.sum(conf_mat[:,-1])}")
                    
                    if epoch:
                        final_train_gndtrth.append(members)
                        final_train_predict.append(predicted)
                        final_train_probabe.append(results[:, 1])

                    batch_idx += 1
                except EOFError:
                    break

        if epoch:
            final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
            final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
            final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

            train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
            train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

            # final_result.append(train_f1_score)
            # final_result.append(train_roc_auc_score)
            
            train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
            train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)
            
            final_result.append(1.*correct/total)
            final_result.append((prec/batch_idx).item())
            
            final_result.append((recall/batch_idx).item())
            
            final_result.append(train_f1_score)
            final_result.append(train_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
                
            with open(result_path_csv, "w") as f:
                # Encode the pickled data using Base64
                pickled_data = pickle.dumps((final_train_gndtrth, final_train_predict, final_train_probabe))
                encoded_data = base64.b64encode(pickled_data)

                # Write the encoded data to the CSV file
                f.write(encoded_data.decode('utf-8'))
                # pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
            
            print("Saved Attack Train Ground Truth and Predict Sets")
            print("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

        # final_result.append(1.*correct/total)
        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f precision: %.3f recall: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx,100*prec/batch_idx,100*recall/batch_idx))
        
        # exit()

        return final_result

    def test(self, epoch, result_path):
        self.attack_model.eval()
        batch_idx = 1
        correct = 0
        total = 0
        prec = 0
        recall = 0

        bcm = BinaryConfusionMatrix().to(self.device)
        
        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []

        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while(True):
                    try:
                        output, prediction, members, targets = pickle.load(f)
                        output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

                        results = self.attack_model(output, prediction, targets)
                        results = F.softmax(results, dim=1)
                        _, predicted = results.max(1)
                        total += members.size(0)
                        correct += predicted.eq(members).sum().item()
                        
                        # print(f"members type: {type(members)}, device: {members.get_device()}, predicted: {type(predicted)}, device: {predicted.get_device()}")
                        conf_mat = bcm(predicted, members)
                        
                        prec += conf_mat[1,1]/torch.sum(conf_mat[:,-1])    
                        recall+=conf_mat[1,1]/torch.sum(conf_mat[-1,:])
                        
                        if epoch:
                            final_test_gndtrth.append(members)
                            final_test_predict.append(predicted)
                            final_test_probabe.append(results[:, 1])

                        batch_idx += 1
                    except EOFError:
                        break

        if epoch:
            final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
            final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
            final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

            # test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
            # test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

            # final_result.append(test_f1_score)
            # final_result.append(test_roc_auc_score)
            
            test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
            test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)
            
            final_result.append(1.*correct/total)
            final_result.append((prec/batch_idx).item())
            
            final_result.append((recall/batch_idx).item())
            
            final_result.append(test_f1_score)
            final_result.append(test_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

            print("Saved Attack Test Ground Truth and Predict Sets")
            print("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))

        # final_result.append(1.*correct/total)
        print( 'Test Acc: %.3f%% (%d/%d), precision: %.3f, recall: %.3f' % (100.*correct/(1.0*total), correct, total, 100.*prec/(1.0*batch_idx),100*recall/batch_idx))

        return final_result
    
    def train_i(self, epoch, result_path, result_path_csv, class_name):
        
        self.attack_model.train()
        batch_idx = 1
        train_loss = 0
        correct = 0
        prec = 0
        recall = 0
        total = 0
        bcm = BinaryConfusionMatrix().to(self.device)
        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []

        final_result = []
        file_path = self.ATTACK_SETS + f"_train_{class_name}.p"
        # print(f"file size is : {os.path.getsize(file_path)}")
        # exit()
        
        if os.path.getsize(file_path) != 0:
            with open(file_path, "rb") as f:
                while(True):
                    try:
                        output, prediction, members, targets = pickle.load(f)
                        # print(targets)
                        output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)
                        prediction = prediction.unsqueeze(1).to(dtype=torch.float32)
                        members = members.to(dtype=torch.int64)
                        # print(f"i-output: {output.size(), prediction.size(), members.size()}")
                        # print(f"i-output output: {type(output),output.dtype, type(prediction), prediction.dtype, members.dtype}")

                        # # exit()
                        # print(f"members: {members, targets}")
                        # exit()
                        results = self.attack_model(output, prediction, targets)
                        
                        # print(f"i-results: {results}")
                        
                        
                        results = F.softmax(results, dim=1)
                        # exit()
                        
                        self.optimizer.zero_grad()
                        
                        losses = self.criterion(results, members)
                        # exit()
                        
                        losses.backward()
                        self.optimizer.step()

                        train_loss += losses.item()
                        _, predicted = results.max(1)
                        total += members.size(0)
                        correct += predicted.eq(members).sum().item()
                        # exit()
                        # print(f"correctly predicted member and non-members: {predicted.eq(members).sum().item()} out of :{members.size(0)}")
                    
                        # print(f"members type: {type(members)}, device: {members.get_device()}, predicted: {type(predicted)}, device: {predicted.get_device()}")
                        conf_mat = bcm(predicted, members)
                        
                        prec += conf_mat[1,1]/torch.sum(conf_mat[:,-1])    
                        recall+=conf_mat[1,1]/torch.sum(conf_mat[-1,:])
                        # print(conf_mat)
                        # print(f"correct: {torch.sum(torch.diagonal(conf_mat, 0))}")
                        # print(f"last col sum: {torch.sum(conf_mat[:,-1])}")
                        # exit()
                        if epoch:
                            final_train_gndtrth.append(members)
                            final_train_predict.append(predicted)
                            final_train_probabe.append(results[:, 1])

                        batch_idx += 1
                    except EOFError:
                        break

            if epoch:
                final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
                final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
                final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

                train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
                train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

                # final_result.append(train_f1_score)
                # final_result.append(train_roc_auc_score)
                
                train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
                train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)
                
                final_result.append(1.*correct/total)
                final_result.append((prec/batch_idx).item())
                
                final_result.append((recall/batch_idx).item())
                
                final_result.append(train_f1_score)
                final_result.append(train_roc_auc_score)
                
                # print(f"prec/batch_idx: {prec}")
                # print(f"prec/batch_idx: {recall}")
                
                
                with open(result_path, "wb") as f:
                    pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
                    
                with open(result_path_csv, "w") as f:
                    # Encode the pickled data using Base64
                    pickled_data = pickle.dumps((final_train_gndtrth, final_train_predict, final_train_probabe))
                    encoded_data = base64.b64encode(pickled_data)

                    # Write the encoded data to the CSV file
                    f.write(encoded_data.decode('utf-8'))
                    # pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
                
                print("Saved Attack Train Ground Truth and Predict Sets")
                print("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

        # final_result.append(1.*correct/total)
            print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f precision: %.3f recall: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx,100*prec/batch_idx,100*recall/batch_idx))
        
        # exit()
        else:
            print(f"skipping class {class_name} in training, size is : {os.path.getsize(file_path)}")
            
        

        return final_result

    def test_i(self, epoch, result_path, class_name):
        self.attack_model.eval()
        batch_idx = 1
        correct = 0
        total = 0
        prec = 0
        recall = 0

        bcm = BinaryConfusionMatrix().to(self.device)
        
        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []

        final_result = []
        file_path = self.ATTACK_SETS + f"_test_{class_name}.p"
        if os.path.getsize(file_path) != 0:
            with torch.no_grad():
                with open(file_path, "rb") as f:
                    while(True):
                        try:
                            output, prediction, members, targets = pickle.load(f)
                            output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)
                            
                            prediction = prediction.unsqueeze(1).to(dtype=torch.float32)
                            members = members.to(dtype=torch.int64)
                        
                            results = self.attack_model(output, prediction, targets)
                            results = F.softmax(results, dim=1)
                            _, predicted = results.max(1)
                            total += members.size(0)
                            correct += predicted.eq(members).sum().item()
                            
                            # print(f"members type: {type(members)}, device: {members.get_device()}, predicted: {type(predicted)}, device: {predicted.get_device()}")
                            conf_mat = bcm(predicted, members)
                            
                            prec += conf_mat[1,1]/torch.sum(conf_mat[:,-1])    
                            recall+=conf_mat[1,1]/torch.sum(conf_mat[-1,:])
                            
                            if epoch:
                                final_test_gndtrth.append(members)
                                final_test_predict.append(predicted)
                                final_test_probabe.append(results[:, 1])

                            batch_idx += 1
                        except EOFError:
                            break

                if epoch:
                    final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
                    final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
                    final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

                    # test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
                    # test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

                    # final_result.append(test_f1_score)
                    # final_result.append(test_roc_auc_score)
                    
                    test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
                    test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)
                    
                    final_result.append(1.*correct/total)
                    final_result.append((prec/batch_idx).item())
                    
                    final_result.append((recall/batch_idx).item())
                    
                    final_result.append(test_f1_score)
                    final_result.append(test_roc_auc_score)

                    with open(result_path, "wb") as f:
                        pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

                    print("Saved Attack Test Ground Truth and Predict Sets")
                    print("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))

        # final_result.append(1.*correct/total)
            print( 'Test Acc: %.3f%% (%d/%d), precision: %.3f, recall: %.3f' % (100.*correct/(1.0*total), correct, total, 100.*prec/(1.0*batch_idx),100*recall/batch_idx))

        else:
            print(f"skipping class {class_name}, size is in testing: {os.path.getsize(file_path)}")
            
        
        return final_result
    
    def delete_pickle_mul(self, num_classes):
        # train_file = glob.glob(self.ATTACK_SETS +"train.p")
        for class_name in range(num_classes):
            file_path = self.ATTACK_SETS + f"_train_{class_name}.p"
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                continue
            # exit()
        # for trf in train_file:
        #     os.remove(trf)

        
        for class_name in range(num_classes):
            file_path = self.ATTACK_SETS + f"_test_{class_name}.p"
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                continue

            
    def delete_pickle(self):
        train_file = glob.glob(self.ATTACK_SETS +"train.p")
        for trf in train_file:
            os.remove(trf)

        test_file = glob.glob(self.ATTACK_SETS +"test.p")
        for tef in test_file:
            os.remove(tef)

    def saveModel(self, path):
        torch.save(self.attack_model.state_dict(), path)
        
class attack_for_whitebox():
    def __init__(self, TARGET_PATH, SHADOW_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, target_model, shadow_model, attack_model, device, class_num):
        self.device = device
        self.class_num = class_num

        self.ATTACK_SETS = ATTACK_SETS

        self.TARGET_PATH = TARGET_PATH
        self.target_model = target_model.to(self.device)
        self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
        self.target_model.eval()


        self.SHADOW_PATH = SHADOW_PATH
        self.shadow_model = shadow_model.to(self.device)
        self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH))
        self.shadow_model.eval()

        self.attack_train_loader = attack_train_loader
        self.attack_test_loader = attack_test_loader

        self.attack_model = attack_model.to(self.device)
        torch.manual_seed(0)
        self.attack_model.apply(weights_init)

        self.target_criterion = nn.CrossEntropyLoss(reduction='none')
        self.attack_criterion = nn.CrossEntropyLoss()
        #self.optimizer = optim.SGD(self.attack_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-5)

        self.attack_train_data = None
        self.attack_test_data = None
        

    def _get_data(self, model, inputs, targets):
        results = model(inputs)
        # outputs = F.softmax(outputs, dim=1)
        losses = self.target_criterion(results, targets)

        gradients = []
        
        for loss in losses:
            loss.backward(retain_graph=True)

            gradient_list = reversed(list(model.named_parameters()))

            for name, parameter in gradient_list:
                if 'weight' in name:
                    gradient = parameter.grad.clone() # [column[:, None], row].resize_(100,100)
                    gradient = gradient.unsqueeze_(0)
                    gradients.append(gradient.unsqueeze_(0))
                    break

        labels = []
        for num in targets:
            label = [0 for i in range(self.class_num)]
            label[num.item()] = 1
            labels.append(label)

        gradients = torch.cat(gradients, dim=0)
        losses = losses.unsqueeze_(1).detach()
        outputs, _ = torch.sort(results, descending=True)
        labels = torch.Tensor(labels)

        return outputs, losses, gradients, labels

    def prepare_dataset(self):
        with open(self.ATTACK_SETS + "train.p", "wb") as f:
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, loss, gradient, label = self._get_data(self.shadow_model, inputs, targets)

                pickle.dump((output, loss, gradient, label, members), f)

        print("Finished Saving Train Dataset")

        with open(self.ATTACK_SETS + "test.p", "wb") as f:
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, loss, gradient, label = self._get_data(self.target_model, inputs, targets)
            
                pickle.dump((output, loss, gradient, label, members), f)

            # pickle.dump((output, loss, gradient, label, members), open(self.ATTACK_PATH + "test.p", "wb"))

        print("Finished Saving Test Dataset")

    
    def train(self, epoch, result_path):
        self.attack_model.train()
        batch_idx = 1
        train_loss = 0
        correct = 0
        total = 0

        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []

        final_result = []

        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while(True):
                try:
                    output, loss, gradient, label, members = pickle.load(f)
                    output, loss, gradient, label, members = output.to(self.device), loss.to(self.device), gradient.to(self.device), label.to(self.device), members.to(self.device)

                    results = self.attack_model(output, loss, gradient, label)
                    # results = F.softmax(results, dim=1)
                    losses = self.attack_criterion(results, members)
                    
                    losses.backward()
                    self.optimizer.step()

                    train_loss += losses.item()
                    _, predicted = results.max(1)
                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    if epoch:
                        final_train_gndtrth.append(members)
                        final_train_predict.append(predicted)
                        final_train_probabe.append(results[:, 1])

                    batch_idx += 1
                except EOFError:
                    break	

        if epoch:
            final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
            final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
            final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

            train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
            train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

            final_result.append(train_f1_score)
            final_result.append(train_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
            
            print("Saved Attack Train Ground Truth and Predict Sets")
            print("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

        final_result.append(1.*correct/total)
        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return final_result


    def test(self, epoch, result_path):
        self.attack_model.eval()
        batch_idx = 1
        correct = 0
        total = 0

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []

        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while(True):
                    try:
                        output, loss, gradient, label, members = pickle.load(f)
                        output, loss, gradient, label, members = output.to(self.device), loss.to(self.device), gradient.to(self.device), label.to(self.device), members.to(self.device)

                        results = self.attack_model(output, loss, gradient, label)

                        _, predicted = results.max(1)
                        total += members.size(0)
                        correct += predicted.eq(members).sum().item()

                        results = F.softmax(results, dim=1)

                        if epoch:
                            final_test_gndtrth.append(members)
                            final_test_predict.append(predicted)
                            final_test_probabe.append(results[:, 1])

                        batch_idx += 1
                    except EOFError:
                        break

        if epoch:
            final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
            final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
            final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

            test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
            test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

            final_result.append(test_f1_score)
            final_result.append(test_roc_auc_score)


            with open(result_path, "wb") as f:
                pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

            print("Saved Attack Test Ground Truth and Predict Sets")
            print("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))

        final_result.append(1.*correct/total)
        print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

        return final_result

    def delete_pickle(self):
        train_file = glob.glob(self.ATTACK_SETS +"train.p")
        for trf in train_file:
            os.remove(trf)

        test_file = glob.glob(self.ATTACK_SETS +"test.p")
        for tef in test_file:
            os.remove(tef)

    def saveModel(self, path):
        torch.save(self.attack_model.state_dict(), path)


def train_shadow_model(PATH, device, shadow_model, train_loader, test_loader, use_DP, noise, norm, loss, optimizer, delta):
    model = shadow(train_loader, test_loader, shadow_model, device, use_DP, noise, norm, loss, optimizer, delta)
    acc_train = 0
    acc_test = 0

    for i in range(60):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("shadow training")

        acc_train = model.train()
        print("shadow testing")
        acc_test = model.test()


        overfitting = round(acc_train - acc_test, 6)

        print('The overfitting rate is %s' % overfitting)

    FILE_PATH = PATH + "_shadow.pth"
    model.saveModel(FILE_PATH)
    print("saved shadow model!!!")
    print("Finished training!!!")

    return acc_train, acc_test, overfitting

def train_shadow_distillation(MODEL_PATH, DL_PATH, device, target_model, student_model, train_loader, test_loader):
    distillation = distillation_training(MODEL_PATH, train_loader, test_loader, student_model, target_model, device)

    for i in range(100):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("shadow distillation training")

        acc_distillation_train = distillation.train()
        print("shadow distillation testing")
        acc_distillation_test = distillation.test()


        overfitting = round(acc_distillation_train - acc_distillation_test, 6)

        print('The overfitting rate is %s' % overfitting)

        
    result_path = DL_PATH + "_shadow.pth"

    distillation.saveModel(result_path)
    print("Saved shadow model!!!")
    print("Finished training!!!")

    return acc_distillation_train, acc_distillation_test, overfitting

def get_attack_dataset_without_shadow(train_set, test_set, batch_size):
    mem_length = len(train_set)//3
    nonmem_length = len(test_set)//3
    mem_train, mem_test, _ = torch.utils.data.random_split(train_set, [mem_length, mem_length, len(train_set)-(mem_length*2)])
    nonmem_train, nonmem_test, _ = torch.utils.data.random_split(test_set, [nonmem_length, nonmem_length, len(test_set)-(nonmem_length*2)])
    mem_train, mem_test, nonmem_train, nonmem_test = list(mem_train), list(mem_test), list(nonmem_train), list(nonmem_test)

    for i in range(len(mem_train)):
        mem_train[i] = mem_train[i] + (1,)
    for i in range(len(nonmem_train)):
        nonmem_train[i] = nonmem_train[i] + (0,)
    for i in range(len(nonmem_test)):
        nonmem_test[i] = nonmem_test[i] + (0,)
    for i in range(len(mem_test)):
        mem_test[i] = mem_test[i] + (1,)
        
    attack_train = mem_train + nonmem_train
    attack_test = mem_test + nonmem_test

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=batch_size, shuffle=True, num_workers=2)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=batch_size, shuffle=True, num_workers=2)

    return attack_trainloader, attack_testloader

# def get_attack_dataset_with_shadow(target_train, target_test, shadow_train, shadow_test, attack_test, batch_size):
#     mem_train, nonmem_train, mem_test, nonmem_test,  nonmem_test_atk= list(shadow_train), list(shadow_test), list(target_train), list(target_test), list(attack_test)

#     # shadow_test = target_test
    
#     # print(f"shadow_train size: {shadow_train[10][0].size()}")
    
    
#     for i in range(len(mem_train)):
#         mem_train[i] = mem_train[i] + (1,)
#         # print(f"shadow_train size: {mem_train[i]}")
        
#         # exit()
   
            
#     for i in range(len(nonmem_train)):
#         nonmem_train[i] = nonmem_train[i] + (0,)
#     for i in range(len(nonmem_test)):
#         nonmem_test[i] = nonmem_test[i] + (0,)
    
#     # --------------
#     for i in range(len(nonmem_test_atk)):
#         nonmem_test_atk[i] = nonmem_test_atk[i] + (0,)
#     for i in range(len(mem_test)):
#         mem_test[i] = mem_test[i] + (1,)
#     # --------------

#     train_length = min(len(mem_train), len(nonmem_train))
#     # test_length = min(len(mem_test), len(nonmem_test))
    
#     test_length = min(len(mem_test), len(nonmem_test_atk))
    
    

#     mem_train, _ = torch.utils.data.random_split(mem_train, [train_length, len(mem_train) - train_length])
#     non_mem_train, _ = torch.utils.data.random_split(nonmem_train, [train_length, len(nonmem_train) - train_length])
#     mem_test, _ = torch.utils.data.random_split(mem_test, [test_length, len(mem_test) - test_length])
#     # non_mem_test, _ = torch.utils.data.random_split(nonmem_test, [test_length, len(nonmem_test) - test_length])
#     nonmem_test_atk, _ = torch.utils.data.random_split(nonmem_test_atk, [test_length, len(nonmem_test) - test_length])
    
    
#     attack_train = mem_train + non_mem_train
#     # attack_test = mem_test + non_mem_test
    
#     attack_test = mem_test + nonmem_test_atk
    

#     # print(f"attack_train size: {attack_train[0]}")
#     # exit()
#     attack_trainloader = torch.utils.data.DataLoader(
#         attack_train, batch_size=batch_size, shuffle=True, num_workers=2)
#     # for output, targets, members in attack_trainloader:
        
#     #     break
#     # print(f"len of batch output_coll: {len(output)}, and size: {output.shape}")
#     # exit()
#     attack_testloader = torch.utils.data.DataLoader(
#         attack_test, batch_size=batch_size, shuffle=True, num_workers=2)

#     return attack_trainloader, attack_testloader

def get_attack_dataset_with_shadow(target_train, target_test, shadow_train, shadow_test, batch_size):
    mem_train, nonmem_train, mem_test, nonmem_test = list(shadow_train), list(shadow_test), list(target_train), list(target_test)

    for i in range(len(mem_train)):
        mem_train[i] = mem_train[i] + (1,)
    for i in range(len(nonmem_train)):
        nonmem_train[i] = nonmem_train[i] + (0,)
    for i in range(len(nonmem_test)):
        nonmem_test[i] = nonmem_test[i] + (0,)
    for i in range(len(mem_test)):
        mem_test[i] = mem_test[i] + (1,)


    train_length = min(len(mem_train), len(nonmem_train))
    test_length = min(len(mem_test), len(nonmem_test))

    mem_train, _ = torch.utils.data.random_split(mem_train, [train_length, len(mem_train) - train_length])
    non_mem_train, _ = torch.utils.data.random_split(nonmem_train, [train_length, len(nonmem_train) - train_length])
    mem_test, _ = torch.utils.data.random_split(mem_test, [test_length, len(mem_test) - test_length])
    non_mem_test, _ = torch.utils.data.random_split(nonmem_test, [test_length, len(nonmem_test) - test_length])
    
    attack_train = mem_train + non_mem_train
    attack_test = mem_test + non_mem_test

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=batch_size, shuffle=True, num_workers=2)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=batch_size, shuffle=True, num_workers=2)

    return attack_trainloader, attack_testloader



# black shadow
def attack_mode0(TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, get_attack_set, num_classes):
    MODELS_PATH = ATTACK_PATH + "_meminf_attack0.pth"
    RESULT_PATH = ATTACK_PATH + "_meminf_attack0.p"
    ATTACK_SETS = ATTACK_PATH + "_meminf_attack_mode0_"

    # MODELS_PATH, RESULT_PATH, ATTACK_PATH
 
    print(f"MODELS_PATH: {MODELS_PATH}, RESULT_PATH: {RESULT_PATH}, ATTACK_PATH: {ATTACK_SETS}")
    # exit()
    attack = attack_for_blackbox(SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, device)

    if get_attack_set:
        attack.delete_pickle()
        attack.prepare_dataset()

    for i in range(50):
        flag = 1 if i == 49 else 0
        print("Epoch %d :" % (i+1))
        res_train = attack.train(flag, RESULT_PATH)
        # exit()
        res_test = attack.test(flag, RESULT_PATH)

    attack.saveModel(MODELS_PATH)
    print("Saved Attack Model")

    return res_train, res_test

# Combined attack
def attack_mode0_com(TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, get_attack_set, num_classes, mode):
    MODELS_PATH = ATTACK_PATH + "_meminf_attack0_com.pth"
    RESULT_PATH = ATTACK_PATH + "_meminf_attack0_com.p"
    RESULT_PATH_csv = ATTACK_PATH + "_meminf_attack0_com.csv"
    
    ATTACK_SETS = ATTACK_PATH + "_meminf_attack_mode0__com"

    # MODELS_PATH, RESULT_PATH, ATTACK_PATH
    #! for weak shadow_model change model architecture to just taking the PV int simple NN and 
    
    print(f"MODELS_PATH: {MODELS_PATH}, \nRESULT_PATH: {RESULT_PATH}, \nATTACK_PATH: {ATTACK_SETS}")
    
    epoch_data = []
    train_accuracy_list = []
    test_accuracy_list = []
    res_list = []

    if mode == -1:
        print(f"mode is: {mode}, treating LABELS as unavailabile")
        attack = attack_for_blackbox_com(SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, device)    
        
        if get_attack_set:
            attack.delete_pickle()
            attack.prepare_dataset()
        #    aaaaaaaaaaaaaaaaa
        epochs = 80
        tr_sum = 0.0;
        ts_sum = 0.0;
        for i in range(epochs):
            flag = 1 if i == (epochs-1) else 0
            print("Epoch %d :" % (i+1))
            tic()
            res_train = attack.train(flag, RESULT_PATH, RESULT_PATH_csv)
            tr_sum+=toc();
            # exit()
            # print(f"res_train: {res_train}")
            tic()
            res_test = attack.test(flag, RESULT_PATH)
            ts_sum+= toc();
            # if flag == 0:
            #     train_accuracy_list.append(res_train)
            #     test_accuracy_list.append(res_test)
            
        print(f"\ntrain times: {tr_sum}\n")
        # print(f"test times: {ts_sum}\n")
        res_list.append({'acc': res_test[0], 'prec': res_test[1], 'rec': res_test[2],'f1': res_test[3], 'auc': res_test[4] })
            # res_list.append({'class': class_name,'acc': '--', 'prec': '--', 'rec': '--','f1': '--', 'auc': '--' })

            
        df = pd.DataFrame(res_list)
        file_path = ATTACK_SETS + f"_Results-Mean.csv" 
        df.to_csv(file_path, index=False)
        
        
        attack.saveModel(MODELS_PATH)
        print("Saved Attack Model")
        
    elif mode == -2:
        print(f"mode is: {mode}, treating LABELS in hard manner")
        
        # fff
        attack = attack_for_blackbox_com(SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, device)
        
        get_attack_set = 0 #! for makeing the k-classes datasets
        
        if get_attack_set==1:
            attack.delete_pickle_mul(num_classes)
            attack.prepare_dataset_mul(num_classes) # this should generate K train and test batches files for k attack models
        else:
            print("skipping prepare_dataset_mul")
            

    # exit()
        
        all_c_sum = 0.0;
        for class_name in range(num_classes):
            # class_name = 29
            tr_sum = 0.0;
            ts_sum = 0.0;
            epochs = 80
            for i in range(epochs):
                flag = 1 if i == (epochs-1) else 0
                print("Epoch %d :" % (i+1))
                tic()
                res_train = attack.train_i(flag, RESULT_PATH, RESULT_PATH_csv, class_name)
                tr_sum+=toc();
                # exit()
                res_test = attack.test_i(flag, RESULT_PATH, class_name)
                
            # exit()
            all_c_sum+=tr_sum  
            # print(f"class mean tr_time (mode -2): {tr_sum}")
            # print(f"mean tr_time: {ts_sum}")
          
            print(f"class {class_name} train results: {res_train}")
            print(f"class {class_name} train results: {res_test}")
            
            if not res_train:
                print("The list is empty.")
                # res_list.append({'class': class_name,'acc': 'nill', 'prec': 'nill', 'rec': r'nill','f1': 'nill', 'auc': 'nill'})
            else:
                # print("The list is not empty.")
                res_list.append({'class': class_name,'acc': res_test[0], 'prec': res_test[1], 'rec': res_test[2],'f1': res_test[3], 'auc': res_test[4]})
                
            # exit()
            # res_list.append({'class': class_name,'acc': '--', 'prec': '--', 'rec': '--','f1': '--', 'auc': '--' })
            
            # res_list.append({'class': class_name,'acc': res_train[0], 'prec': res_train[1], 'rec': res_train[2],'f1': res_train[3], 'auc': res_train[4] })
            
            # res_list.append({'class': class_name,'acc': res_test[0], 'prec': res_test[1], 'rec': res_test[2],'f1': res_test[3], 'auc': res_test[4]})
            # res_list.append({'class': class_name,'acc': '--', 'prec': '--', 'rec': '--','f1': '--', 'auc': '--' })

        print(f"call class mean tr_time (mode -2): {all_c_sum}")
        df = pd.DataFrame(res_list)
        file_path = ATTACK_SETS + f"_Results-K-classes.csv" 
        df.to_csv(file_path, index=False)
            
       
    else:
        print("mode is not -1 or -2")
   
    return res_train, res_test

def attack_mode0_rnn(TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, target_model, shadow_model, rnn_model, get_attack_set, num_classes):
    MODELS_PATH = ATTACK_PATH + "_meminf_attack0_rnn.pth"
    RESULT_PATH = ATTACK_PATH + "_meminf_attack0_rnn.p"
    ATTACK_SETS = ATTACK_PATH + "_meminf_attack_mode0__rnn"

    # MODELS_PATH, RESULT_PATH, ATTACK_PATH
 
    print(f"MODELS_PATH: {MODELS_PATH}, RESULT_PATH: {RESULT_PATH}, ATTACK_PATH: {ATTACK_SETS}")
    # exit()
    attack = RNN_attack_for_blackbox(SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, target_model, shadow_model, rnn_model, device)

    if get_attack_set:
        attack.delete_pickle()
        attack.prepare_dataset() # how data is prepated, i mean batches, batch size
        print(f"attack mode data preparation successful! PVs")
        
    epochs = 120
    for i in range(epochs):
        flag = 1 if i == (epochs-1) else 0
        print("Epoch %d :" % (i+1))
        res_train = attack.train(flag, RESULT_PATH)
        # exit()
        res_test = attack.test(flag, RESULT_PATH)

    attack.saveModel(MODELS_PATH)
    print("Saved Attack Model")

    return res_train, res_test

# black partial
def attack_mode1(TARGET_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, target_model, attack_model, get_attack_set, num_classes):
    MODELS_PATH = ATTACK_PATH + "_meminf_attack1.pth"
    RESULT_PATH = ATTACK_PATH + "_meminf_attack1.p"
    ATTACK_SETS = ATTACK_PATH + "_meminf_attack_mode1_"

    attack = attack_for_blackbox(TARGET_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, target_model, target_model, attack_model, device)

    if get_attack_set:
        attack.delete_pickle()
        attack.prepare_dataset()

    for i in range(50):
        flag = 1 if i == 49 else 0
        print("Epoch %d :" % (i+1))
        res_train = attack.train(flag, RESULT_PATH)
        res_test = attack.test(flag, RESULT_PATH)

    attack.saveModel(MODELS_PATH)
    print("Saved Attack Model")

    return res_train, res_test

# white partial
def attack_mode2(TARGET_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, target_model, attack_model, get_attack_set, num_classes):
    MODELS_PATH = ATTACK_PATH + "_meminf_attack2.pth"
    RESULT_PATH = ATTACK_PATH + "_meminf_attack2.p"
    ATTACK_SETS = ATTACK_PATH + "_meminf_attack_mode2_"

    attack = attack_for_whitebox(TARGET_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, target_model, target_model, attack_model, device, num_classes)
    
    if get_attack_set:
        attack.delete_pickle()
        attack.prepare_dataset()

    for i in range(50):
        flag = 1 if i == 49 else 0
        print("Epoch %d :" % (i+1))
        res_train = attack.train(flag, RESULT_PATH)
        res_test = attack.test(flag, RESULT_PATH)

    attack.saveModel(MODELS_PATH)
    print("Saved Attack Model")

    return res_train, res_test

# white shadow
def attack_mode3(TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, get_attack_set, num_classes):
    MODELS_PATH = ATTACK_PATH + "_meminf_attack3.pth"
    RESULT_PATH = ATTACK_PATH + "_meminf_attack3.p"
    ATTACK_SETS = ATTACK_PATH + "_meminf_attack_mode3_"

    attack = attack_for_whitebox(TARGET_PATH, SHADOW_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, device, num_classes)
    
    if get_attack_set:
        attack.delete_pickle()
        attack.prepare_dataset()

    for i in range(50):
        flag = 1 if i == 49 else 0
        print("Epoch %d :" % (i+1))
        res_train = attack.train(flag, RESULT_PATH)
        res_test = attack.test(flag, RESULT_PATH)

    attack.saveModel(MODELS_PATH)
    print("Saved Attack Model")

    return res_train, res_test

def get_gradient_size(model):
    gradient_size = []
    gradient_list = reversed(list(model.named_parameters()))
    for name, parameter in gradient_list:
        if 'weight' in name:
            gradient_size.append(parameter.shape)

    return gradient_size
    