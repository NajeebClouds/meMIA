import os
import sys
import torch
import argparse
import torch.nn as nn
import torchvision.models as models
from meMIA.meminf import *
from demoloader.dataloader import *
import torch
import torch.nn as nn
import torch.nn.functional as F



class LSTMClassifier(nn.Module):
    def __init__(self, class_num,  device,  hidden_dim=128, layer_dim=1, output_dim=1, batch_size=64):
        super().__init__()
       
        self.h_size_1 = 256
        self.h_size_2 = 128
        # print(f"y: {y.size()}")
        # print(y)
    
        
        self.h_size_3 = 32
        
        self.lstm1 = nn.LSTM(class_num, self.h_size_1, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(self.h_size_1, self.h_size_2, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        self.lstm3 = nn.LSTM(self.h_size_2, self.h_size_3, batch_first=True)
        self.hidden2label = nn.Linear(self.h_size_3, 2)
        

      
        
        self.input_dim = class_num
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.device = device

        self.hidden1 = self.init_hidden1()
        self.hidden2 = self.init_hidden2()
        self.hidden3 = self.init_hidden3()
        
    def init_hidden1(self):
        return (Variable(torch.zeros(1, self.batch_size, self.h_size_1).to(self.device)),
                Variable(torch.zeros(1, self.batch_size, self.h_size_1).to(self.device)))
    def init_hidden2(self):
        return (Variable(torch.zeros(1, self.batch_size, self.h_size_2).to(self.device)),
                Variable(torch.zeros(1, self.batch_size, self.h_size_2).to(self.device)))

    def init_hidden3(self):
        return (Variable(torch.zeros(1, self.batch_size, self.h_size_3).to(self.device)),
                Variable(torch.zeros(1, self.batch_size, self.h_size_3).to(self.device)))

    def forward(self, x):
      
        # print(f"input dim= {input_dim}")
        x = x.view(self.batch_size, 1, self.input_dim)
        
        # print(f"input dim= {input_dim}")
        # print(f"x in forward pass: {x.size()}")
        
        # lstm_out, self.hidden = self.lstm(x, self.hidden)
        # y  = self.hidden2label(lstm_out[:, -1, :])
        x1, self.hidden1 = self.lstm1(x, self.hidden1)
        x2 = self.dropout1(x1[:, -1, :])

        # print(f"x1 size= {x1.size()}, x2 size= {x2.size()}")

        
        x2 = x2.view(self.batch_size, 1, x2.size()[1])
        
        # print(f"reshaped x2 size= {x2.size()}")
  
        x3, self.hidden2 = self.lstm2(x2, self.hidden2)
        x4 = self.dropout2(x3[:, -1, :])
        
        # print(f"after x3 size= {x3.size()}, x4 size= {x4.size()}")
        
        x4 = x4.view(self.batch_size, 1, x4.size()[1])
        
        # print(f"reshaped x4 size= {x4.size()}")
  
        x5, self.hidden2 = self.lstm3(x4, self.hidden3)
        
        # print(f"after x5 size= {x5.size()}")
        
        y  = self.hidden2label(x5[:, -1, :])
      
        
        self.hidden1 = self.init_hidden1()
        self.hidden2 = self.init_hidden2()
        self.hidden3 = self.init_hidden3()
        return y
    

class CombinedShadowAttackModel(nn.Module):
    def __init__(self, class_num,  device,  hidden_dim=128, layer_dim=1, output_dim=1, batch_size=64):
        
        super(CombinedShadowAttackModel, self).__init__()
        
        # batch_size = 2
        self.h_size_1 = 256
        self.h_size_2 = 128
        self.h_size_3 = 50
        
        self.lstm1 = nn.LSTM(class_num, self.h_size_1, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.lstm2 = nn.LSTM(self.h_size_1, self.h_size_2, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.lstm3 = nn.LSTM(self.h_size_2, self.h_size_3, batch_first=True)
        
        self.hidden2label = nn.Linear(self.h_size_3, 2)
        
        self.input_dim = class_num
        self.batch_size = batch_size
        
        self.batch_size = 64
        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.device = device
        
        
        
        self.hidden1 = self.init_hidden1()
        self.hidden2 = self.init_hidden2()
        self.hidden3 = self.init_hidden3()
        
        
        self.Output_NSH = nn.Sequential(
			nn.Linear(class_num, 10),
			nn.ReLU(),
			nn.Linear(10, 30),
            nn.ReLU(),
			nn.Linear(30, 10),
		)
        
        self.label_NSH = nn.Sequential(
			nn.Linear(class_num, 100),
			nn.ReLU(),
			nn.Linear(100, 5),
		)
        
        self.final_NSH = nn.Sequential(
			nn.Linear(5+10, 100),
			nn.ReLU(),
			nn.Linear(100, 50),
            nn.ReLU(),
			nn.Linear(50, 2),
		)
        
        
        
        self.Output_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(class_num, 512),
			nn.ReLU(),
			nn.Linear(512, 64),
            # nn.ReLU(),
			# nn.Linear(256, 64),
		)
        
        self.Output_Component_meMIA = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(class_num, 512),
			nn.ReLU(),
			nn.Linear(512, 256),
            nn.ReLU(),
			nn.Linear(256, 128),
            nn.ReLU(),
			nn.Linear(128, 64),
		)
        
        self.Prediction_Component = nn.Sequential(
			# nn.Dropout(p=0.5),
			nn.Linear(1, 512),
			nn.ReLU(),
			nn.Linear(512, 64),
          
		)

        self.meMIA_Encoder_Component_joint = nn.Sequential(
			nn.Linear(self.h_size_3+64+64, 512), #mine
			nn.ReLU(),
			# nn.Dropout(p=0.5),
			nn.Linear(512, 256),
			nn.ReLU(),
			# nn.Dropout(p=0.5),
			nn.Linear(256, 128),
			nn.ReLU(),
			# nn.Dropout(p=0.5),
            nn.Linear(128, 2),
		)
        
        self.mia_Encoder_Component = nn.Sequential(
			nn.Linear(class_num, 512), #mia
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
            nn.Linear(128, 2),
           
		)
        
        self.meMIA_Encoder_Component = nn.Sequential(
			nn.Linear(class_num+64, 512), #meMIA
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
            nn.Linear(128, 2),
           
		)
        self.Encoder_Component = nn.Sequential(
			nn.Linear(class_num+64, 512), #mia_actual
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
            nn.Linear(128, 2),
           
		)
        
    def init_hidden1(self):
        return (Variable(torch.zeros(1, self.batch_size, self.h_size_1).to(self.device)),
                Variable(torch.zeros(1, self.batch_size, self.h_size_1).to(self.device)))
    def init_hidden2(self):
        return (Variable(torch.zeros(1, self.batch_size, self.h_size_2).to(self.device)),
                Variable(torch.zeros(1, self.batch_size, self.h_size_2).to(self.device)))
    def init_hidden3(self):
        return (Variable(torch.zeros(1, self.batch_size, self.h_size_3).to(self.device)),
                Variable(torch.zeros(1, self.batch_size, self.h_size_3).to(self.device)))
        
    def forward(self, output, prediction, label):
        
        self.hidden1 = self.init_hidden1()
        self.hidden2 = self.init_hidden2()
        self.hidden3 = self.init_hidden3()
       
       
    #    #! NSH attack
        
        # label_one_hot_encoded = torch.nn.functional.one_hot(label.to(torch.int64), self.input_dim).float().to(self.device)
        # # print(f"size of label: {label_one_hot_encoded.size()}")
        # # print(f"size of label: {label_one_hot_encoded.dtype}")
        
        # # exit()
        
        # out_nsh = self.Output_NSH(output)#ouput --> class_num
        # lable_nsh =  self.label_NSH(label_one_hot_encoded)
        
        # combined_nsh = torch.cat((out_nsh, lable_nsh), 1)
        # final_result = self.final_NSH(combined_nsh)


        # # #! mia
        # Prediction_Component_result = self.Prediction_Component(prediction) #ouput --> class_num 64
        # # output = self.Output_Component(output) #64
        # final_result = self.Encoder_Component(torch.cat((Prediction_Component_result, output), 1))
        # # final_result = self.mia_Encoder_Component(output)
        
        #! Mine combined architecture
        label_one_hot_encoded = torch.nn.functional.one_hot(label.to(torch.int64), self.input_dim).float()
        
        x = output.view(self.batch_size, 1, self.input_dim)
        x1, self.hidden1 = self.lstm1(x, self.hidden1)
        # x2 = self.dropout1(x1[:, -1, :])
        x2 = x1[:, -1, :]
        x2 = x2.view(self.batch_size, 1, x2.size()[1])
        x3, self.hidden2 = self.lstm2(x2, self.hidden2)
        # x4 = self.dropout2(x3[:, -1, :])
        x4 = x3[:, -1, :]
        x4 = x4.view(self.batch_size, 1, x4.size()[1])
        x5, self.hidden2 = self.lstm3(x4, self.hidden3)
        
        output = self.Output_Component_meMIA(output) #64
        # exit()
        Prediction_Component_result = self.Prediction_Component(prediction) #ouput --> class_num|64
        final_inputs = torch.cat((x5[:, -1, :],Prediction_Component_result, output), 1)
        final_result = self.meMIA_Encoder_Component_joint(final_inputs)
        
        # # Droping the LSTMs part (meMIA only on NN)
        # Prediction_Component_result = self.Prediction_Component(prediction) #ouput --> class_num 64
        # final_result = self.meMIA_Encoder_Component(torch.cat((Prediction_Component_result, output), 1))
        
        # Dropping NN (meMIA only on LSTMs)
        # x = output.view(self.batch_size, 1, self.input_dim)
        # x1, self.hidden1 = self.lstm1(x, self.hidden1)
        # x2 = self.dropout1(x1[:, -1, :])
        # x2 = x2.view(self.batch_size, 1, x2.size()[1])
        # x3, self.hidden2 = self.lstm2(x2, self.hidden2)
        # x4 = self.dropout2(x3[:, -1, :])
        # x4 = x4.view(self.batch_size, 1, x4.size()[1])
        # x5, self.hidden2 = self.lstm3(x4, self.hidden3)
        # final_result = self.hidden2label(x5[:, -1, :])
        
        # ! sqeMIA
        # print(f"input Dim: {self.input_dim}")
        # print(f"output Dim: {output.size()}")
        
        
        # exit()
        
        # x = output.view(self.batch_size, 1, self.input_dim)
        
        # x1, self.hidden1 = self.lstm1(x, self.hidden1)
        # x2 = self.dropout1(x1[:, -1, :])
        # x2 = x2.view(self.batch_size, 1, x2.size()[1])
        # x3, self.hidden2 = self.lstm2(x2, self.hidden2)
        # x4 = self.dropout2(x3[:, -1, :])
        # x4 = x4.view(self.batch_size, 1, x4.size()[1])
        # x5, self.hidden2 = self.lstm3(x4, self.hidden3)
        
        # final_result  = self.hidden2label(x5[:, -1, :])
        
        # print(f"input Dim: {self.input_dim}")
        # print(f"output Dim: {output.size()}")
        # # exit()
        # x = output.view(self.batch_size, 1, self.input_dim)
        # x1, self.hidden1 = self.lstm1(x, self.hidden1)
        # x2 = self.dropout1(x1[:, -1, :])
        # x2 = x2.view(self.batch_size, 1, x2.size()[1])
        # x3, self.hidden2 = self.lstm2(x2, self.hidden2)
        # x4 = self.dropout2(x3[:, -1, :])
        # x4 = x4.view(self.batch_size, 1, x4.size()[1])
        # x5, self.hidden2 = self.lstm3(x4, self.hidden3)
        # final_result = self.hidden2label(x5[:, -1, :])
        
        return final_result



class ShadowAttackModel(nn.Module):
	def __init__(self, class_num):
		super(ShadowAttackModel, self).__init__()
		self.Output_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Prediction_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(1, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Encoder_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(class_num, 256),
			nn.ReLU(),
			# nn.Dropout(p=0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			# nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)


	def forward(self, output, prediction):
		Output_Component_result = self.Output_Component(output)
		Prediction_Component_result = self.Prediction_Component(prediction)
		
		final_inputs = torch.cat((Output_Component_result, Prediction_Component_result), 1)
		final_result = self.Encoder_Component(output)

		return final_result


# class PartialAttackModel(nn.Module):
# 	def __init__(self, class_num):
# 		super(PartialAttackModel, self).__init__()
# 		self.Output_Component = nn.Sequential(
# 			# nn.Dropout(p=0.2),
# 			nn.Linear(class_num, 128),
# 			nn.ReLU(),
# 			nn.Linear(128, 64),
# 		)

# 		self.Prediction_Component = nn.Sequential(
# 			# nn.Dropout(p=0.2),
# 			nn.Linear(1, 128),
# 			nn.ReLU(),
# 			nn.Linear(128, 64),
# 		)

# 		self.Encoder_Component = nn.Sequential(
# 			# nn.Dropout(p=0.2),
# 			nn.Linear(128, 256),
# 			nn.ReLU(),
# 			# nn.Dropout(p=0.2),
# 			nn.Linear(256, 128),
# 			nn.ReLU(),
# 			# nn.Dropout(p=0.2),
# 			nn.Linear(128, 64),
# 			nn.ReLU(),
# 			nn.Linear(64, 2),
# 		)


# 	def forward(self, output, prediction):
# 		Output_Component_result = self.Output_Component(output)
# 		Prediction_Component_result = self.Prediction_Component(prediction)
		
# 		final_inputs = torch.cat((Output_Component_result, Prediction_Component_result), 1)
# 		final_result = self.Encoder_Component(final_inputs)

# 		return final_result



def train_model(PATH, device, train_set, test_set, model, use_DP, noise, norm, delta, dataset_name):
    print("Training model: train set shape", len(train_set), " test set shape: ", len(test_set), ", device: ", device)
    print(f"dataset Name: {dataset_name}")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=2)
    
    model = model_training(train_loader, test_loader, model, device, use_DP, noise, norm, delta)
    acc_train = 0
    acc_test = 0
	
    for i in range(60):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("target training")

        acc_train = model.train()
        print("target testing")
        acc_test = model.test()

        overfitting = round(acc_train - acc_test, 6)
        print('The overfitting rate is %s' % overfitting)

    FILE_PATH = PATH + "_target.pth"
    model.saveModel(FILE_PATH)
    print("Saved target model!!!")
    print("Finished training!!!")

    return acc_train, acc_test, overfitting

  
def test_meminf(PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, train_rnn, train_shadow, use_DP, noise, norm, delta, mode):
    
    batch_size = 64
    if train_shadow:
        shadow_trainloader = torch.utils.data.DataLoader(
            shadow_train, batch_size=64, shuffle=True, num_workers=2)
        shadow_testloader = torch.utils.data.DataLoader(
            shadow_test, batch_size=64, shuffle=True, num_workers=2)

        loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(shadow_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        # optimizer = optim.SGD(shadow_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-7)
        
        print(f"training shadow model")
        
        train_shadow_model(PATH, device, shadow_model, shadow_trainloader, shadow_testloader, use_DP, noise, norm, loss, optimizer, delta)
        # exit()
    
    # collect training data for attack model as we have already trained shadow and target model
    if mode == 0 or mode == 3 or mode == -1 or mode == -2:
        attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
            target_train, target_test, shadow_train, shadow_test, batch_size)
    else:
        attack_trainloader, attack_testloader = get_attack_dataset_without_shadow(target_train, target_test, batch_size)

   
    if mode == -1:
        attack_model = CombinedShadowAttackModel(num_classes,  device,  hidden_dim=128, layer_dim=1, output_dim=1, batch_size=batch_size)
        
        print(attack_model)
        # attack_mode0(PATH + "_target.pth", PATH + "_shadow.pth", PATH, device, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, 1, num_classes)
        attack_mode0_com(PATH + "_target.pth", PATH + "_shadow.pth", PATH, device, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, 1, num_classes, mode)
    # hard black box settings
    elif mode == -2: 
        attack_model = CombinedShadowAttackModel(num_classes,  device,  hidden_dim=128, layer_dim=1, output_dim=1, batch_size=16)
        # attack_model = ShadowAttackModel(num_classes)
        print(attack_model)
        # attack_mode0(PATH + "_target.pth", PATH + "_shadow.pth", PATH, device, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, 1, num_classes)
        attack_mode0_com(PATH + "_target.pth", PATH + "_shadow.pth", PATH, device, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, 1, num_classes, mode)
      
         
    elif mode == 0:
        attack_model = ShadowAttackModel(num_classes)
        attack_mode0(PATH + "_target.pth", PATH + "_shadow.pth", PATH, device, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, 1, num_classes)
    else:
        raise Exception("Wrong mode")
    


def str_to_bool(string):
    if isinstance(string, bool):
       return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, default="0")
    parser.add_argument('-a', '--attributes', type=str, default="race", help="For attrinf, two attributes should be in format x_y e.g. race_gender")
    parser.add_argument('-dn', '--dataset_name', type=str, default="STL10")
    parser.add_argument('-at', '--attack_type', type=int, default=0)
    parser.add_argument('-tm', '--train_model', action='store_true')
    parser.add_argument('-ts', '--train_shadow', action='store_true')
    parser.add_argument('-trnn', '--train_rnn', action='store_true') # if the not mentioned in cmd, it will be false else true
    parser.add_argument('-ud', '--use_DP', action='store_true',)
    parser.add_argument('-ne', '--noise', type=float, default=1.3)
    parser.add_argument('-nm', '--norm', type=float, default=1.5)
    parser.add_argument('-d', '--delta', type=float, default=1e-5)
    parser.add_argument('-m', '--mode', type=int, default=0)
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    dataset_name = args.dataset_name
    attr = args.attributes
    if "_" in attr:
        attr = attr.split("_")
    root = "./data"
    use_DP = args.use_DP
    noise = args.noise
    norm = args.norm
    delta = args.delta
    mode = args.mode
 
    train_shadow = args.train_shadow
    train_rnn = args.train_rnn

    TARGET_ROOT = "./demoloader/trained_model/"
    if not os.path.exists(TARGET_ROOT):
        print(f"Create directory named {TARGET_ROOT}")
        os.makedirs(TARGET_ROOT)
    TARGET_PATH = TARGET_ROOT + dataset_name
    print("Target_patth: ", TARGET_PATH)
    
    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(dataset_name, attr, root, device)
    
    if args.train_model:
        print("Training Target model")
        
        train_model(TARGET_PATH, device, target_train, target_test, target_model, use_DP, noise, norm, delta, dataset_name)
   
    # membership inference
    if args.attack_type == 0:
        # test_meminf(TARGET_PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, attack_test, target_model, shadow_model, train_rnn, train_shadow, use_DP, noise, norm, delta, mode)
        test_meminf(TARGET_PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, train_rnn, train_shadow, use_DP, noise, norm, delta, mode)
    else:
        sys.exit("we have not supported this mode yet! 0c0")

if __name__ == "__main__":
    main()
    
