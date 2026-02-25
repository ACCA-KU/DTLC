from rdkit import Chem

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch.utils.data import Subset
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import SimpleConv
from torch_geometric.data import Batch
from tqdm.auto import tqdm
import os


def r2_score(preds, targets):
    nan_mask = ~torch.isnan(targets)
    preds = preds[nan_mask]
    targets = targets[nan_mask]
    mean = targets.mean()
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - mean) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-12))




class Run_deep_Rf():

    def __init__(self, device, relay, model, loss_fn, optimizer, scheduler, callback, last_ckpt_path, early_stopping=False, total_epoch = 1000):

        # relay : True or False --> True 일 경우 이어서 학습
        self.device = device
        self.relay = relay
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callback = callback
        self.start_epoch = 0
        self.total_epoch = total_epoch
        self.error_log_train= []
        self.r2_log_train=[]
        self.error_log_valid = []
        self.r2_log_valid=[]
        self.early_stopping = early_stopping
        
        if self.relay:
            self.last_ckpt = torch.load(last_ckpt_path)
            self.model.load_state_dict(self.last_ckpt['model_state_dict'])
            self.optimizer.load_state_dict(self.last_ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(self.last_ckpt['scheduler_state_dict'])
            self.start_epoch = self.last_ckpt['epoch']+1
            self.error_log_train = self.last_ckpt.get('train_loss', [])
            self.r2_log_train = self.last_ckpt.get('train_r2', [])
            self.error_log_valid = self.last_ckpt.get('valid_loss', [])
            self.r2_log_valid = self.last_ckpt.get('valid_r2', [])

            if self.start_epoch >=  self.total_epoch:
                raise ValueError(f'총 epoch: {self.total_epoch}가 이전에 저장된 epoch: {self.start_epoch} 보다 작음')
               
    def __call__(self, train_loader, valid_loader):
        
        for epoch in tqdm(range(self.start_epoch, self.total_epoch)):
            self.model.train()
            train_out_all=[]
            train_label_all=[]
            valid_out_all=[]
            valid_label_all=[]
            for com, elu1, elu2 in train_loader:
                com_train_batch = com.to(self.device)
                elu1_train_batch = elu1.to(self.device)
                elu2_train_batch = elu2.to(self.device)
                out = self.model(com_train_batch, elu1_train_batch, elu2_train_batch)   
                self.optimizer.zero_grad()             
                loss = self.loss_fn(out, com_train_batch.y)
                loss.backward()
                self.optimizer.step()
                
                train_out_all.append(out.detach().view(-1))
                train_label_all.append(com_train_batch.y.detach().view(-1))
                
            train_out_all = torch.cat(train_out_all, dim=0)
            train_label_all = torch.cat(train_label_all, dim=0)
            train_epoch_MAE = self.loss_fn(train_out_all, train_label_all, purpose = 'lookup')
            self.error_log_train.append(train_epoch_MAE.item())
            train_epoch_r2 = r2_score(train_out_all, train_label_all)
            self.r2_log_train.append(train_epoch_r2.item())
        
            self.model.eval()
            with torch.no_grad():      
                for com_val, elu1_val, elu2_val in valid_loader:
                    com_valid_batch = com_val.to(self.device)
                    elu1_valid_batch = elu1_val.to(self.device)
                    elu2_valid_batch = elu2_val.to(self.device)
                    pred = self.model(com_valid_batch, elu1_valid_batch, elu2_valid_batch)                    

                    valid_out_all.append(pred.detach().view(-1))
                    valid_label_all.append(com_valid_batch.y.detach().view(-1))
        
            valid_out_all = torch.cat(valid_out_all, dim=0)
            valid_label_all = torch.cat(valid_label_all, dim=0)
            valid_epoch_MAE = self.loss_fn(valid_out_all, valid_label_all, purpose = 'lookup')
            self.error_log_valid.append(valid_epoch_MAE.item())
            valid_epoch_r2 = r2_score(valid_out_all, valid_label_all)
            self.r2_log_valid.append(valid_epoch_r2.item())
        
            # schedular
            self.scheduler.step()
        
            # Early stop & callback
            early_stop = self.callback(epoch, self.model, self.optimizer, self.scheduler, 
                           train_loss = self.error_log_train,
                           train_r2 = self.r2_log_train,
                           valid_loss = self.error_log_valid,
                           valid_r2 = self.r2_log_valid)
                           
        
            print(f'Epoch : {epoch+1} | Train_loss : {self.error_log_train[-1]:.4f}, Train_r2 : {self.r2_log_train[-1]:.4f} | Valid_loss : {self.error_log_valid[-1]:.4f}, Valid_r2 : {self.r2_log_valid[-1]:.4f}')
        
            logging_path = 'train&valid_logging.txt'
            with open(logging_path, 'a') as f:
                f.write(f'Epoch : {epoch+1} | Train_loss : {self.error_log_train[-1]:.4f}, Train_r2 : {self.r2_log_train[-1]:.4f} | Valid_loss : {self.error_log_valid[-1]:.4f}, Valid_r2 : {self.r2_log_valid[-1]:.4f}')

            if early_stop:
                if self.early_stopping == True:
                    print(f'Early stopping at epoch {epoch+1}')
                    break



##############################################################################################################################################

class Run_deep_Rf_unbalance():

    def __init__(self, device, relay, model, loss_fn1, label_counts, loss_fn2,  optimizer, scheduler, callback, last_ckpt_path, early_stopping=False, total_epoch = 1000):

        # relay : True or False --> True 일 경우 이어서 학습
        self.device = device
        self.relay = relay
        self.model = model
        self.loss_fn1 = loss_fn1
        self.label_counts = label_counts
        self.loss_fn2 = loss_fn2
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callback = callback
        self.start_epoch = 0
        self.total_epoch = total_epoch
        self.error_log_train= []
        self.r2_log_train=[]
        self.error_log_valid = []
        self.r2_log_valid=[]
        self.early_stopping = early_stopping
        
        if self.relay:
            self.last_ckpt = torch.load(last_ckpt_path)
            self.model.load_state_dict(self.last_ckpt['model_state_dict'])
            self.optimizer.load_state_dict(self.last_ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(self.last_ckpt['scheduler_state_dict'])
            self.start_epoch = self.last_ckpt['epoch']+1
            self.error_log_train = self.last_ckpt.get('train_loss', [])
            self.r2_log_train = self.last_ckpt.get('train_r2', [])
            self.error_log_valid = self.last_ckpt.get('valid_loss', [])
            self.r2_log_valid = self.last_ckpt.get('valid_r2', [])

            if self.start_epoch >=  self.total_epoch:
                raise ValueError(f'총 epoch: {self.total_Epoch}가 이전에 저장된 epoch: {self.start_epoch} 보다 작음')
               
    def __call__(self, train_loader, valid_loader):
        
        for epoch in tqdm(range(self.start_epoch, self.total_epoch)):
            self.model.train()
            epoch_loss_train = 0
            epoch_r2_train=0
            num_train_batch = 0
            for com, elu1, elu2 in train_loader:
                com_train_batch = com.to(self.device)
                elu1_train_batch = elu1.to(self.device)
                elu2_train_batch = elu2.to(self.device)
                out = self.model(com_train_batch, elu1_train_batch, elu2_train_batch)   
                self.optimizer.zero_grad()             
                loss, real_loss, r2 = self.loss_fn1(out, com_train_batch.y, self.label_counts)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss_train += real_loss.item()
                epoch_r2_train += r2.item()
                num_train_batch += 1
                
            self.error_log_train.append(epoch_loss_train / num_train_batch)
            self.r2_log_train.append(epoch_r2_train / num_train_batch)
        
            self.model.eval()
            epoch_loss_valid = 0
            epoch_r2_valid = 0
            num_valid_batch = 0
            with torch.no_grad():      
                for com_val, elu1_val, elu2_val in valid_loader:
                    com_valid_batch = com_val.to(self.device)
                    elu1_valid_batch = elu1_val.to(self.device)
                    elu2_valid_batch = elu2_val.to(self.device)
                    pred = self.model(com_valid_batch, elu1_valid_batch, elu2_valid_batch)                    
                    loss, real_loss, r2 = self.loss_fn2(pred, com_valid_batch.y)
                    epoch_loss_valid += real_loss.item()
                    epoch_r2_valid += r2.item()
                    num_valid_batch += 1
        
            self.error_log_valid.append(epoch_loss_valid / num_valid_batch)
            self.r2_log_valid.append(epoch_r2_valid / num_valid_batch)
        
            # schedular
            self.scheduler.step()
        
            # Early stop & callback
            early_stop = self.callback(epoch, self.model, self.optimizer, self.scheduler, 
                           train_loss = self.error_log_train,
                           train_r2 = self.r2_log_train,
                           valid_loss = self.error_log_valid,
                           valid_r2 = self.r2_log_valid)
                           
        
            print(f'Epoch : {epoch+1} | Train_loss : {self.error_log_train[-1]:.4f}, Train_r2 : {self.r2_log_train[-1]:.4f} | Valid_loss : {self.error_log_valid[-1]:.4f}, Valid_r2 : {self.r2_log_valid[-1]:.4f}')
        
            logging_path = 'train&valid_logging.txt'
            with open(logging_path, 'a') as f:
                f.write(f'Epoch : {epoch+1} | Train_loss : {self.error_log_train[-1]:.4f}, Train_r2 : {self.r2_log_train[-1]:.4f} | Valid_loss : {self.error_log_valid[-1]:.4f}, Valid_r2 : {self.r2_log_valid[-1]:.4f}')

            if early_stop:
                if self.early_stopping == True:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

                    ##############################################


class Run_deep_Rf_sigmoid():

    def __init__(self, device, relay, model, loss_fn1, loss_fn2,  optimizer, scheduler, callback, last_ckpt_path, early_stopping=False, total_epoch = 1000):

        # relay : True or False --> True 일 경우 이어서 학습
        self.device = device
        self.relay = relay
        self.model = model
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callback = callback
        self.start_epoch = 0
        self.total_epoch = total_epoch
        self.error_log_train= []
        self.r2_log_train=[]
        self.error_log_valid = []
        self.r2_log_valid=[]
        self.early_stopping = early_stopping
        
        if self.relay:
            self.last_ckpt = torch.load(last_ckpt_path)
            self.model.load_state_dict(self.last_ckpt['model_state_dict'])
            self.optimizer.load_state_dict(self.last_ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(self.last_ckpt['scheduler_state_dict'])
            self.start_epoch = self.last_ckpt['epoch']+1
            self.error_log_train = self.last_ckpt.get('train_loss', [])
            self.r2_log_train = self.last_ckpt.get('train_r2', [])
            self.error_log_valid = self.last_ckpt.get('valid_loss', [])
            self.r2_log_valid = self.last_ckpt.get('valid_r2', [])

            if self.start_epoch >=  self.total_epoch:
                raise ValueError(f'총 epoch: {self.total_Epoch}가 이전에 저장된 epoch: {self.start_epoch} 보다 작음')
               
    def __call__(self, com_train_loader, elu1_train_loader, elu2_train_loader,
                 com_valid_loader, elu1_valid_loader, elu2_valid_loader):
        
        for epoch in tqdm(range(self.start_epoch, self.total_epoch)):
            self.model.train()
            epoch_loss_train = 0
            epoch_r2_train=0
            num_train_batch = 0
            for data_batch_train in zip(com_train_loader, elu1_train_loader, elu2_train_loader):
                com_train_batch = data_batch_train[0].to(self.device)
                elu1_train_batch = data_batch_train[1].to(self.device)
                elu2_train_batch = data_batch_train[2].to(self.device)
                out = self.model(com_train_batch, elu1_train_batch, elu2_train_batch)   
                self.optimizer.zero_grad()             
                loss, real_loss, r2 = self.loss_fn1(out, com_train_batch.y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss_train += real_loss.item()
                epoch_r2_train += r2.item()
                num_train_batch += 1
                
            self.error_log_train.append(epoch_loss_train / num_train_batch)
            self.r2_log_train.append(epoch_r2_train / num_train_batch)
        
            self.model.eval()
            epoch_loss_valid = 0
            epoch_r2_valid = 0
            num_valid_batch = 0
            with torch.no_grad():      
                for data_batch_valid in zip(com_valid_loader, elu1_valid_loader, elu2_valid_loader):
                    com_valid_batch = data_batch_valid[0].to(self.device)
                    elu1_valid_batch = data_batch_valid[1].to(self.device)
                    elu2_valid_batch = data_batch_valid[2].to(self.device)
                    pred = self.model(com_valid_batch, elu1_valid_batch, elu2_valid_batch)                    
                    loss, real_loss, r2 = self.loss_fn2(pred, com_valid_batch.y)
                    epoch_loss_valid += real_loss.item()
                    epoch_r2_valid += r2.item()
                    num_valid_batch += 1
        
            self.error_log_valid.append(epoch_loss_valid / num_valid_batch)
            self.r2_log_valid.append(epoch_r2_valid / num_valid_batch)
        
            # schedular
            self.scheduler.step()
        
            # Early stop & callback
            early_stop = self.callback(epoch, self.model, self.optimizer, self.scheduler, 
                           train_loss = self.error_log_train,
                           train_r2 = self.r2_log_train,
                           valid_loss = self.error_log_valid,
                           valid_r2 = self.r2_log_valid)
                           
        
            print(f'Epoch : {epoch+1} | Train_loss : {self.error_log_train[-1]:.4f}, Train_r2 : {self.r2_log_train[-1]:.4f} | Valid_loss : {self.error_log_valid[-1]:.4f}, Valid_r2 : {self.r2_log_valid[-1]:.4f}')
        
            logging_path = 'train&valid_logging.txt'
            with open(logging_path, 'a') as f:
                f.write(f'Epoch : {epoch+1} | Train_loss : {self.error_log_train[-1]:.4f}, Train_r2 : {self.r2_log_train[-1]:.4f} | Valid_loss : {self.error_log_valid[-1]:.4f}, Valid_r2 : {self.r2_log_valid[-1]:.4f}')

            if early_stop:
                if self.early_stopping == True:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

############################################## using monotone penalty #############################################################################
'''
def monotone_penalty(model_out, snyder, lambda_mono = 1):
    # snyder: (B,4) with requires_grad=True
    grad = torch.autograd.grad(
        outputs=model_out,
        inputs=snyder,
        grad_outputs = torch.ones_like(model_out),
        create_graph=True
    )[0]  
    #torch.ones_like(model_out) --> jocobian에 (1,B)모양의 1 매트릭스 를 행렬곱함 (= 각 배치내 샘플마다 가중치1 로 가중합)
    # 이렇게 하면 jocobina에서 어차피 대각 행렬 제외하면 편미분 한 결과가 0이니까 각 입력-출력 쌍에 대한 미분값만 남음
    # (1,4) 형식 출력
    grad = grad / model_out.numel()  #--> 합이니까 평균 구하기 위해 나누기
    print(f'real derivative value : {grad[0]}')
    return lambda_mono * torch.relu(-grad).mean()

def compute_monotone_penalty_safe_bn(model, com_batch, elu1_batch, elu2_batch, monotone_penalty_fn):

    #BatchNorm의 영향을 제거하면서 gradient 흐름을 유지한 채 monotone_penalty 계산.
    # mono penalty 구할때, 다변수 미분을 하는데 jacobian에서 각 샘플이 독립적이라 가정하고 그래야 각 샘플별 입력에 대한 미분이 제대로 구해짐
    # 그런데, batch normalization layer에서 여러 샘플의 평균을 구하고 그걸로 정규화 하면서 독립적이지 않게됨
    # 따라서 model.eval로 전환 해서 BN 과정에서 평균,표준편차를 새롭게 구하는게 아닌 기존 누적 통계를 사용해 구하게 하면 각 샘플이 독립성을 유지하게 됨
    # eval 모드는 계산 그래프는 유지 되는데, BN과정에서 현재 입력받은 샘플 값으로 통계 구하지 않고 train에서 구한 통계값 이용. drop out도 적용 안 함.
    model.eval()
    out = model(com_batch, elu1_batch, elu2_batch)
    penalty = monotone_penalty_fn(out, com_batch.snyder_param)
    return penalty




class Run_deep_Rf_mono_penalty():

    def __init__(self, device, relay, model, loss_fn, optimizer, scheduler, callback, last_ckpt_path, early_stopping=False, total_epoch = 1000):

        # relay : True or False --> True 일 경우 이어서 학습
        self.device = device
        self.relay = relay
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callback = callback
        self.start_epoch = 0
        self.total_epoch = total_epoch
        self.error_log_train= []
        self.r2_log_train=[]
        self.error_log_valid = []
        self.r2_log_valid=[]
        self.mono_penalty=[]
        self.early_stopping = early_stopping
        
        if self.relay:
            self.last_ckpt = torch.load(last_ckpt_path)
            self.model.load_state_dict(self.last_ckpt['model_state_dict'])
            self.optimizer.load_state_dict(self.last_ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(self.last_ckpt['scheduler_state_dict'])
            self.start_epoch = self.last_ckpt['epoch']+1
            self.error_log_train = self.last_ckpt.get('train_loss', [])
            self.r2_log_train = self.last_ckpt.get('train_r2', [])
            self.error_log_valid = self.last_ckpt.get('valid_loss', [])
            self.r2_log_valid = self.last_ckpt.get('valid_r2', [])

            if self.start_epoch >=  self.total_epoch:
                raise ValueError(f'총 epoch: {self.total_Epoch}가 이전에 저장된 epoch: {self.start_epoch} 보다 작음')
               
    def __call__(self, com_train_loader, elu1_train_loader, elu2_train_loader,
                 com_valid_loader, elu1_valid_loader, elu2_valid_loader):
        
        for epoch in tqdm(range(self.start_epoch, self.total_epoch)):
            self.model.train()
            epoch_loss_train = 0
            epoch_r2_train=0
            epoch_mono_penalty_train=0   ################################################################################################################
            num_train_batch = 0
            for data_batch_train in zip(com_train_loader, elu1_train_loader, elu2_train_loader):
                com_train_batch = data_batch_train[0].to(self.device)
                elu1_train_batch = data_batch_train[1].to(self.device)
                elu2_train_batch = data_batch_train[2].to(self.device)
                #com_train_batch.snyder_param = com_train_batch.snyder_param.detach().clone().requires_grad_(True)   #####################################
                # snyder param 과 연결된 가중치들이 loss1과 loss2 모두에 피드백 받으려면 위 코드를 꺼야하고, loss2(단조성)'만' 피드백 할꺼면 켜라 --> 전자 택
                self.optimizer.zero_grad()
                out = self.model(com_train_batch, elu1_train_batch, elu2_train_batch)                
                loss_1, real_loss, r2 = self.loss_fn(out, com_train_batch.y)

                loss_2 = compute_monotone_penalty_safe_bn(self.model, com_train_batch, elu1_train_batch, elu2_train_batch,monotone_penalty_fn=monotone_penalty)
                self.model.train()

                loss = loss_1+loss_2   ##################################################################################################################
                loss.backward()
                self.optimizer.step()
                
                epoch_loss_train += real_loss.item()
                epoch_r2_train += r2.item()
                epoch_mono_penalty_train += loss_2.item()   #############################################################################################
                num_train_batch += 1
                
            self.error_log_train.append(epoch_loss_train / num_train_batch)
            self.r2_log_train.append(epoch_r2_train / num_train_batch)
            self.mono_penalty.append(epoch_mono_penalty_train / num_train_batch)   ######################################################################
        
            self.model.eval()
            epoch_loss_valid = 0
            epoch_r2_valid = 0
            num_valid_batch = 0
            with torch.no_grad():      
                for data_batch_valid in zip(com_valid_loader, elu1_valid_loader, elu2_valid_loader):
                    com_valid_batch = data_batch_valid[0].to(self.device)
                    elu1_valid_batch = data_batch_valid[1].to(self.device)
                    elu2_valid_batch = data_batch_valid[2].to(self.device)
                    pred = self.model(com_valid_batch, elu1_valid_batch, elu2_valid_batch)                    
                    loss, real_loss, r2 = self.loss_fn(pred, com_valid_batch.y)
                    epoch_loss_valid += real_loss.item()
                    epoch_r2_valid += r2.item()
                    num_valid_batch += 1
        
            self.error_log_valid.append(epoch_loss_valid / num_valid_batch)
            self.r2_log_valid.append(epoch_r2_valid / num_valid_batch)
        
            # schedular
            self.scheduler.step()
        
            # Early stop & callback
            early_stop = self.callback(epoch, self.model, self.optimizer, self.scheduler, 
                           train_loss = self.error_log_train,
                           train_r2 = self.r2_log_train,
                           valid_loss = self.error_log_valid,
                           valid_r2 = self.r2_log_valid,
                           mono_penalty = self.mono_penalty ),
                           
        
            print(f'Epoch : {epoch+1} | Train_loss : {self.error_log_train[-1]:.4f}, Train_r2 : {self.r2_log_train[-1]:.4f} | Valid_loss : {self.error_log_valid[-1]:.4f}, Valid_r2 : {self.r2_log_valid[-1]:.4f}, penalty : {self.mono_penalty[-1]:.4f}')####################
        
            logging_path = 'train&valid_logging.txt'
            with open(logging_path, 'a') as f:
                f.write(f'Epoch : {epoch+1} | Train_loss : {self.error_log_train[-1]:.4f}, Train_r2 : {self.r2_log_train[-1]:.4f} | Valid_loss : {self.error_log_valid[-1]:.4f}, Valid_r2 : {self.r2_log_valid[-1]:.4f}, penalty : {self.mono_penalty[-1]:.4f}')#######################

            if early_stop:
                if self.early_stopping == True:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
'''
###################################################################################################################################################

############################################   monotone_penalty / variance penalty    ##############################################################
'''
def monotone_penalty(model_out, snyder, lambda_mono):
    # snyder: (B,4) with requires_grad=True
    grad = torch.autograd.grad(
        outputs=model_out,
        inputs=snyder,
        grad_outputs = torch.ones_like(model_out),
        create_graph=True,
    )[0]  
    #torch.ones_like(model_out) --> jocobian에 (1,B)모양의 1 매트릭스 를 행렬곱함 (= 각 배치내 샘플마다 가중치1 로 가중합)
    # 이렇게 하면 jocobina에서 어차피 대각 행렬(입-출력 쌍) 제외하면 편미분 한 결과가 0이니까 각 입력-출력 쌍에 대한 미분값만 남음
    # (500,4) 형식 출력

    target = 0.6

    #print('delta_grad',delta_grad)
    print('grad.mean()', grad.mean())
    
    return lambda_mono * F.softplus(target - grad).sum(), F.softplus(target - grad).sum().item()

def compute_monotone_penalty_safe_bn(model, com_batch, elu1_batch, elu2_batch, w_mono, monotone_penalty_fn):

    #BatchNorm의 영향을 제거하면서 gradient 흐름을 유지한 채 monotone_penalty 계산.
    # mono penalty 구할때, 다변수 미분을 하는데 jacobian에서 각 샘플이 독립적이라 가정하고 그래야 각 샘플별 입력에 대한 미분이 제대로 구해짐
    # 그런데, batch normalization layer에서 여러 샘플의 평균을 구하고 그걸로 정규화 하면서 독립적이지 않게됨
    # 따라서 model.eval로 전환 해서 BN 과정에서 평균,표준편차를 새롭게 구하는게 아닌 기존 누적 통계를 사용해 구하게 하면 각 샘플이 독립성을 유지하게 됨
    # eval 모드는 계산 그래프는 유지 되는데, BN과정에서 현재 입력받은 샘플 값으로 통계 구하지 않고 train에서 구한 통계값 이용. drop out도 적용 안 함.
    
    model.eval()
    out = model(com_batch, elu1_batch, elu2_batch)
    penalty, grad = monotone_penalty_fn(out, com_batch.snyder_param, lambda_mono = w_mono)
    return penalty, grad  

def variance_penalty_bn(aug_out, lambda_var, group_size = 11, threshold = 0.218):
 
    # 11개 혹은 같은 그룹 끼리의 분산 구하는 걸로 변경해야 함
    # shuffle = False로 입력 받았다 가정하고 그냥 순차적으로 11개씩 묶기

    #aug_var_mean= torch.stack([torch.var(aug_out[group_size*i: group_size*(i+1)]) for i in range(500)]).mean

    # 분산대신 MAD(절대표준편차) 로 변경. 제곱 후 루트는 이상치 발생시 매우 커져서 이상치 하나+비슷한 값들 조합으로도 수치 극대화 가능한 맹점이 있음
    # 분산 보다 표준편차가 직관적임
    aug_MAD_mean = torch.stack([torch.abs(aug_out[group_size*i: group_size*(i+1)]-aug_out[group_size*i: group_size*(i+1)].mean()) for i in range(500)]).mean()
    #print(aug_MAD_mean)
    aug_penalty = torch.relu(lambda_var*(threshold - aug_MAD_mean))
    return aug_penalty, aug_MAD_mean.item()

def aug_chunk_loader(dataset, aug_index, batch_size=500):
    subset = Subset(dataset, aug_index) #len(aug_index) = (original batch size)*11. maybe 5500
    return DataLoader(subset, batch_size=batch_size, shuffle=False)



class Run_deep_Rf_mono_var_penalty():

    def __init__(self, device, relay, model, loss_fn, optimizer, scheduler, callback, last_ckpt_path, early_stopping=False, total_epoch = 1000):

        # relay : True or False --> True 일 경우 이어서 학습
        self.device = device
        self.relay = relay
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callback = callback
        self.start_epoch = 0
        self.total_epoch = total_epoch
        self.error_log_train= []
        self.r2_log_train=[]
        self.error_log_valid = []
        self.r2_log_valid=[]
        self.mono_penalty=[0]
        self.var_penalty=[0]
        self.early_stopping = early_stopping

        ########## penalty ############
        self.alpha_mono = 1       # 초기에는 제약 가중치 0 (제약 없음)
        self.alpha_var = 1
        self.alpha_max   = 100.0      # 너무 커지면 학습 폭주할 수 있으니 상한을 둬
        self.eta = 0.5 
        
        if self.relay:
            self.last_ckpt = torch.load(last_ckpt_path)
            self.model.load_state_dict(self.last_ckpt['model_state_dict'])
            self.optimizer.load_state_dict(self.last_ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(self.last_ckpt['scheduler_state_dict'])
            self.start_epoch = self.last_ckpt['epoch']+1
            self.error_log_train = self.last_ckpt.get('train_loss', [])
            self.r2_log_train = self.last_ckpt.get('train_r2', [])
            self.error_log_valid = self.last_ckpt.get('valid_loss', [])
            self.r2_log_valid = self.last_ckpt.get('valid_r2', [])
            self.mono_penalty =  self.last_ckpt.get('mono_penalty', [])
            self.var_penalty =  self.last_ckpt.get('variance_penalty', [])
            self.alpha_mono =  self.last_ckpt.get('alpha_mono', [])
            self.alpha_var =  self.last_ckpt.get('alpha_var', [])

            if self.start_epoch >=  self.total_epoch:
                raise ValueError(f'총 epoch: {self.total_epoch}가 이전에 저장된 epoch: {self.start_epoch} 보다 작음')
               
    def __call__(self, com_train_loader, elu1_train_loader, elu2_train_loader,
                 com_valid_loader, elu1_valid_loader, elu2_valid_loader, compound_aug_set, elu1_aug_set, elu2_aug_set, start_p):
        
        for epoch in tqdm(range(self.start_epoch, self.total_epoch)):

            if epoch < start_p:
                self.alpha_mono = 0.0
                self.alpha_var = 0.0
            elif epoch == start_p :
                self.alpha_mono = 1
                self.alpha_var = 1
            elif epoch > start_p :
                self.alpha_mono  = float(np.clip(self.alpha_mono  + self.eta *(self.mono_penalty[-1]-self.mono_penalty[-2]), 0.0, self.alpha_max))
                self.alpha_var = float(np.clip(self.alpha_var + self.eta * (self.var_penalty[-1]-self.var_penalty[-2]), 0.0, self.alpha_max))


            self.model.train()
            epoch_loss_train = 0
            epoch_r2_train=0
            bad_grad=0
            epoch_mono_penalty_train=0   ################################################################################################################
            epoch_var_penalty_train=0    #####################
            num_train_batch = 0
            for batch_idx,data_batch_train in enumerate(zip(com_train_loader, elu1_train_loader, elu2_train_loader)):
                com_train_batch = data_batch_train[0].to(self.device)
                elu1_train_batch = data_batch_train[1].to(self.device)
                elu2_train_batch = data_batch_train[2].to(self.device)
                #com_train_batch.snyder_param = com_train_batch.snyder_param.detach().clone().requires_grad_(True)   #####################################
                # snyder param 과 연결된 가중치들이 loss1과 loss2 모두에 피드백 받으려면 위 코드를 꺼야하고, loss2(단조성)'만' 피드백 할꺼면 켜라 --> 전자 택
                com_train_batch.snyder_param = com_train_batch.snyder_param.requires_grad_(True)
                out = self.model(com_train_batch, elu1_train_batch, elu2_train_batch)                
                loss_1, real_loss, r2 = self.loss_fn(out, com_train_batch.y)
                
                loss_2, bad_grad_batch = compute_monotone_penalty_safe_bn(self.model, com_train_batch, elu1_train_batch, elu2_train_batch, self.alpha_mono, monotone_penalty_fn=monotone_penalty)
                bad_grad += bad_grad_batch

                #aug_index 정의 (5500개 인덱스 선정하는데 그냥 순차적으로 넣자)
                max_index = len(compound_aug_set)
                aug_index = torch.arange((batch_idx*5500), min((batch_idx+1)*5500 , max_index))
                aug_com_loader = aug_chunk_loader(compound_aug_set, aug_index, batch_size = 500)
                aug_elu1_loader = aug_chunk_loader(elu1_aug_set, aug_index, batch_size = 500)
                aug_elu2_loader = aug_chunk_loader(elu2_aug_set, aug_index, batch_size = 500)
                aug_out_batch=[]
                for aug_batch_mini in zip (aug_com_loader, aug_elu1_loader, aug_elu2_loader):
                    aug_com_batch_mini = aug_batch_mini[0].to(self.device)
                    aug_elu1_batch_mini = aug_batch_mini[1].to(self.device)
                    aug_elu2_batch_mini = aug_batch_mini[2].to(self.device)
                    # 일단  aug batch out(500) 보관 했다가 배치들 다 모이면 전체(5500)에서 패널티 구하자 (실제 배치 하나에 aug배치 11개)
                    aug_out_mini = self.model(aug_com_batch_mini, aug_elu1_batch_mini, aug_elu2_batch_mini)
                    aug_out_batch.append(aug_out_mini)
                aug_out_cat = torch.cat(aug_out_batch)

                loss_3, aug_var = variance_penalty_bn(aug_out_cat, lambda_var=self.alpha_var)
                

                self.model.train()    ####################################################################################################################
                self.optimizer.zero_grad()
                loss = loss_1 + loss_2 + loss_3   ###############################################################################################
                loss.backward()
                self.optimizer.step()
                
                epoch_loss_train += real_loss.item()
                epoch_r2_train += r2.item()
                epoch_mono_penalty_train += loss_2.item()   #############################################################################################
                epoch_var_penalty_train += loss_3.item()    ###########
                num_train_batch += 1
                
            bad_grad_epoch = bad_grad/num_train_batch    
            self.error_log_train.append(epoch_loss_train / num_train_batch)
            self.r2_log_train.append(epoch_r2_train / num_train_batch)
            self.mono_penalty.append(epoch_mono_penalty_train / num_train_batch)   ######################################################################
            self.var_penalty.append(epoch_var_penalty_train / num_train_batch)

            print('alpha_mono : ',self.alpha_mono)
            print('bad_grad_P_epoch : ',bad_grad_epoch)

            self.model.eval()
            epoch_loss_valid = 0
            epoch_r2_valid = 0
            num_valid_batch = 0
            with torch.no_grad():      
                for data_batch_valid in zip(com_valid_loader, elu1_valid_loader, elu2_valid_loader):
                    com_valid_batch = data_batch_valid[0].to(self.device)
                    elu1_valid_batch = data_batch_valid[1].to(self.device)
                    elu2_valid_batch = data_batch_valid[2].to(self.device)
                    pred = self.model(com_valid_batch, elu1_valid_batch, elu2_valid_batch)                    
                    loss, real_loss, r2 = self.loss_fn(pred, com_valid_batch.y)
                    epoch_loss_valid += real_loss.item()
                    epoch_r2_valid += r2.item()
                    num_valid_batch += 1
        
            self.error_log_valid.append(epoch_loss_valid / num_valid_batch)
            self.r2_log_valid.append(epoch_r2_valid / num_valid_batch)
        
            # schedular
            self.scheduler.step()
        
            # Early stop & callback
            early_stop = self.callback(epoch, self.model, self.optimizer, self.scheduler, 
                           train_loss = self.error_log_train,
                           train_r2 = self.r2_log_train,
                           valid_loss = self.error_log_valid,
                           valid_r2 = self.r2_log_valid,
                           mono_penalty = self.mono_penalty,
                           var_penalty = self.var_penalty,
                           a_mono = self.alpha_mono,
                           a_var = self.alpha_var)
                           
        
            log = f'Epoch : {epoch+1} | Train_loss : {self.error_log_train[-1]:.4f}, Train_r2 : {self.r2_log_train[-1]:.4f} | Valid_loss : {self.error_log_valid[-1]:.4f}, Valid_r2 : {self.r2_log_valid[-1]:.4f} | mono_p : {self.mono_penalty[-1]:.4f}, variance_p : {self.var_penalty[-1]:.4f}'

            print(log)

            logging_path = 'train&valid_logging.txt'
            with open(logging_path, 'a') as f:
                f.write(log)

            if early_stop:
                if self.early_stopping == True:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

'''
##################################################################################################################################################################################
#################################################################### mvar and delta penalty ##################################################################################
'''
def monotone_order_penalty( aug_out, snyder_param4, lambda_mono):

    snyder_4_p=[]
    for i in range(4):
        snyder_param = snyder_param4[:,i]
        # (B,1)로 변환해서 broadcasting으로 모든 쌍 비교
        s_i = snyder_param.view(-1,1)
        s_j = snyder_param.view(1,-1)
        y_i = aug_out.view(-1,1)
        y_j = aug_out.view(1,-1)

        # (s_i - s_j) * (y_j - y_i) 가 양수면 위배
        violation = (s_i - s_j) * (y_j - y_i)

        # 음수(순서 지킴)는 0으로 클리핑 → 위배한 경우만 페널티
        penalty = torch.relu(violation).mean()
        snyder_4_p.append(penalty)


    return lambda_mono * torch.stack(snyder_4_p).mean(), torch.stack(snyder_4_p).mean().detach()


def variance_penalty_bn(aug_out, lambda_var, group_size = 11, threshold = 0.27):
    # 11개 혹은 같은 그룹 끼리의 분산 구하는 걸로 변경해야 함
    # shuffle = False로 입력 받았다 가정하고 그냥 순차적으로 11개씩 묶기

    #aug_var_mean= torch.stack([torch.var(aug_out[group_size*i: group_size*(i+1)]) for i in range(500)]).mean

    # 분산대신 MAD(절대표준편차) 로 변경. 제곱 후 루트는 이상치 발생시 매우 커져서 이상치 하나+비슷한 값들 조합으로도 수치 극대화 가능한 맹점이 있음
    # 분산 보다 표준편차가 직관적임
    num_group = len(aug_out)//group_size
    aug_MAD_mean = torch.stack([torch.abs(aug_out[group_size*i: group_size*(i+1)]-aug_out[group_size*i: group_size*(i+1)].mean()) for i in range(num_group)]).mean()
    aug_penalty = torch.relu(lambda_var*(threshold - aug_MAD_mean))
    print(aug_MAD_mean)
   
    return aug_penalty, aug_MAD_mean.detach()

def aug_chunk_loader(dataset, aug_index, batch_size=500):
    subset = Subset(dataset, aug_index) #len(aug_index) = (original batch size)*11. maybe 5500
    return DataLoader(subset, batch_size=batch_size, shuffle=False)




class Run_deep_Rf_delta_var_penalty():

    def __init__(self, device, relay, model, loss_fn, optimizer, scheduler, callback, last_ckpt_path, opt_ckpt1_path, opt_ckpt2_path, start_p, early_stopping=False, total_epoch = 1000):

        # relay : True or False --> True 일 경우 이어서 학습
        self.device = device
        self.relay = relay
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callback = callback
        self.start_p = start_p
        self.start_epoch = 0
        self.total_epoch = total_epoch
        self.error_log_train= []
        self.r2_log_train=[]
        self.error_log_valid = []
        self.r2_log_valid=[]
        self.mono_penalty=[0]
        self.var_penalty=[0]
        self.early_stopping = early_stopping

        ########## penalty ############
        self.alpha_mono = 1       # 초기에는 제약 가중치 0 (제약 없음)
        self.alpha_var = 1
        self.alpha_max   = 100.0      # 너무 커지면 학습 폭주할 수 있으니 상한을 둬
        self.eta = 0.5 
        
        if self.relay:
            self.last_ckpt = torch.load(last_ckpt_path)
            self.opt_ckpt1_path = torch.load(opt_ckpt1_path)
            self.opt_ckpt2_path = torch.load(opt_ckpt2_path)
            self.model.load_state_dict(self.last_ckpt['model_state_dict'])
            self.optimizer.load_state_dict(self.last_ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(self.last_ckpt['scheduler_state_dict'])
            self.start_epoch = self.last_ckpt['epoch']+1

            self.error_log_train = self.last_ckpt.get('train_loss', [])
            self.r2_log_train = self.last_ckpt.get('train_r2', [])
            self.error_log_valid = self.last_ckpt.get('valid_loss', [])
            self.r2_log_valid = self.last_ckpt.get('valid_r2', [])
            self.mono_penalty =  self.last_ckpt.get('mono_penalty', [])
            self.var_penalty =  self.last_ckpt.get('variance_penalty', [])
            self.alpha_mono =  self.last_ckpt.get('alpha_mono', [])
            self.alpha_var =  self.last_ckpt.get('alpha_var', [])

            if self.start_epoch >=  self.total_epoch:
                raise ValueError(f'총 epoch: {self.total_epoch}가 이전에 저장된 epoch: {self.start_epoch} 보다 작음')
               
    def __call__(self, com_train_loader, elu1_train_loader, elu2_train_loader,
                 com_valid_loader, elu1_valid_loader, elu2_valid_loader, compound_aug_set, elu1_aug_set, elu2_aug_set):
        
        for epoch in tqdm(range(self.start_epoch, self.total_epoch)):
            
            #train set (+ augmentation set)
            self.model.train()
            epoch_loss_train = 0
            epoch_r2_train=0
            num_train_batch = 0
            epoch_mono_penalty_train=0   ################################################################################################################
            epoch_var_penalty_train=0    #####################
            aug_mono_epoch=0
            aug_var_epoch=0
            for batch_idx,data_batch_train in enumerate(zip(com_train_loader, elu1_train_loader, elu2_train_loader)):
                com_train_batch = data_batch_train[0].to(self.device)
                elu1_train_batch = data_batch_train[1].to(self.device)
                elu2_train_batch = data_batch_train[2].to(self.device)
                out = self.model(com_train_batch, elu1_train_batch, elu2_train_batch)                
                loss_1, real_loss, r2 = self.loss_fn(out, com_train_batch.y)


                if epoch < self.start_p:
                    aug_mono=0
                    aug_var=0
                    self.optimizer.zero_grad()
                    loss = loss_1
                    loss.backward()
                    self.optimizer.step()

                    epoch_mono_penalty_train += 0
                    epoch_var_penalty_train += 0

                elif epoch >= self.start_p :

                    #self.model.eval()
                    #aug_index 정의 (5500개 인덱스 선정하는데 그냥 순차적으로 넣자)
                    max_index = len(compound_aug_set)
                    aug_index = torch.arange((batch_idx*5500), min((batch_idx+1)*5500 , max_index))
                    aug_com_loader = aug_chunk_loader(compound_aug_set, aug_index, batch_size = 500)
                    aug_elu1_loader = aug_chunk_loader(elu1_aug_set, aug_index, batch_size = 500)
                    aug_elu2_loader = aug_chunk_loader(elu2_aug_set, aug_index, batch_size = 500)
                    aug_out_batch=[]
                    aug_snyder_batch=[]
                    for aug_batch_mini in zip (aug_com_loader, aug_elu1_loader, aug_elu2_loader):
                        aug_com_batch_mini = aug_batch_mini[0].to(self.device)
                        aug_elu1_batch_mini = aug_batch_mini[1].to(self.device)
                        aug_elu2_batch_mini = aug_batch_mini[2].to(self.device)
                        # 일단  aug batch out(500) 보관 했다가 배치들 다 모이면 전체(5500)에서 패널티 구하자 (실제 배치 하나에 aug배치 11개)
                        aug_out_mini = self.model(aug_com_batch_mini, aug_elu1_batch_mini, aug_elu2_batch_mini)
                        aug_out_batch.append(aug_out_mini)
                        aug_snyder_mini = aug_com_batch_mini.snyder_param.view(-1,4)
                        aug_snyder_batch.append(aug_snyder_mini)
                    aug_out_cat = torch.cat(aug_out_batch)
                    aug_snyder_cat = torch.cat(aug_snyder_batch, dim=0)

                    loss_2, aug_mono = monotone_order_penalty( aug_out_cat, aug_snyder_cat, lambda_mono = self.alpha_mono)
                    loss_3, aug_var = variance_penalty_bn( aug_out_cat, lambda_var=self.alpha_var)
                    print(aug_var)

                    #self.model.train()   
                    self.optimizer.zero_grad()
                    loss = loss_1 + loss_2 + loss_3   
                    loss.backward()
                    self.optimizer.step()

                    epoch_mono_penalty_train += loss_2.item()  
                    epoch_var_penalty_train += loss_3.item()   
                
                epoch_loss_train += real_loss.item()
                epoch_r2_train += r2.item()
                num_train_batch += 1
                aug_mono_epoch += aug_mono
                aug_var_epoch += aug_var            

            self.error_log_train.append(epoch_loss_train / num_train_batch)
            self.r2_log_train.append(epoch_r2_train / num_train_batch)
            self.mono_penalty.append(epoch_mono_penalty_train / num_train_batch)   ######################################################################
            self.var_penalty.append(epoch_var_penalty_train / num_train_batch)

            print('alpha_mono : ',self.alpha_mono)
            print('mono_p_pure : ',aug_mono_epoch/num_train_batch)
            print('alpha_var : ',self.alpha_var)
            print('var_p_pure : ',aug_var_epoch/num_train_batch)


            # valiation set
            self.model.eval()
            epoch_loss_valid = 0
            epoch_r2_valid = 0
            num_valid_batch = 0
            with torch.no_grad():      
                for data_batch_valid in zip(com_valid_loader, elu1_valid_loader, elu2_valid_loader):
                    com_valid_batch = data_batch_valid[0].to(self.device)
                    elu1_valid_batch = data_batch_valid[1].to(self.device)
                    elu2_valid_batch = data_batch_valid[2].to(self.device)
                    pred = self.model(com_valid_batch, elu1_valid_batch, elu2_valid_batch)                    
                    loss, real_loss, r2 = self.loss_fn(pred, com_valid_batch.y)
                    epoch_loss_valid += real_loss.item()
                    epoch_r2_valid += r2.item()
                    num_valid_batch += 1
        
            self.error_log_valid.append(epoch_loss_valid / num_valid_batch)
            self.r2_log_valid.append(epoch_r2_valid / num_valid_batch)
        
            # schedular
            self.scheduler.step()
        
            # Early stop & callback
            early_stop = self.callback(epoch, self.model, self.optimizer, self.scheduler,
                           train_loss = self.error_log_train,
                           train_r2 = self.r2_log_train,
                           valid_loss = self.error_log_valid,
                           valid_r2 = self.r2_log_valid,
                           mono_penalty = self.mono_penalty,
                           var_penalty = self.var_penalty,
                           a_mono = self.alpha_mono,
                           a_var = self.alpha_var)

            self.alpha_mono  = float(np.clip(self.alpha_mono  + self.eta *(self.mono_penalty[-1]-self.mono_penalty[-2]), 0.0, self.alpha_max))
            self.alpha_var = float(np.clip(self.alpha_var + self.eta * (self.var_penalty[-1]-self.var_penalty[-2]), 0.0, self.alpha_max))                          
                           
        
            log = f'Epoch : {epoch+1} | Train_loss : {self.error_log_train[-1]:.4f}, Train_r2 : {self.r2_log_train[-1]:.4f} | Valid_loss : {self.error_log_valid[-1]:.4f}, Valid_r2 : {self.r2_log_valid[-1]:.4f} | mono_p : {self.mono_penalty[-1]:.4f}, variance_p : {self.var_penalty[-1]:.4f}'

            print(log)

            logging_path = 'train&valid_logging.txt'
            with open(logging_path, 'a') as f:
                f.write(log)

            if early_stop:
                if self.early_stopping == True:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
'''

##################################################################################################################################################################################
#################################################################### var and delta penalty ##################################################################################
'''     
def monotone_order_penalty( aug_out, eluent_param8, lambda_mono_list):

    penalty_list=[]
    for i, param in enumerate(['HBD', 'HBA', 'CalcTPSA', 'TPSA_ratio', 'HBA+HBD', 'MolLogP', 'CalcLabuteASA', 'MolMR']):
        one_param = eluent_param8[:,i]
        # (B,1)로 변환해서 broadcasting으로 모든 쌍 비교
        s_i = one_param.view(-1,1)
        s_j = one_param.view(1,-1)
        y_i = aug_out.view(-1,1)
        y_j = aug_out.view(1,-1)
        # (s_i - s_j) * (y_j - y_i) 가 양수면 위배
        if i<=4:
            violation = (s_i - s_j) * (y_j - y_i)
        # (s_i - s_j) * (y_i - y_j) 가 양수면 위배
        else:
            violation = (s_i - s_j) * (y_i - y_j)
        # 음수(순서 지킴)는 0으로 클리핑 → 위배한 경우만 페널티
        penalty = torch.relu(violation).mean()
        penalty_list.append(penalty)

    penalty_tensor = torch.stack(penalty_list).squeeze(0)
    final_penalty_tensor = lambda_mono_list.squeeze(0) * penalty_tensor # (element-wise product)

    return final_penalty_tensor, penalty_tensor.detach()


def variance_penalty_bn(aug_out, lambda_var, group_size = 11, threshold = 0.27):
    # 11개 혹은 같은 그룹 끼리의 분산 구하는 걸로 변경해야 함
    # shuffle = False로 입력 받았다 가정하고 그냥 순차적으로 11개씩 묶기

    #aug_var_mean= torch.stack([torch.var(aug_out[group_size*i: group_size*(i+1)]) for i in range(500)]).mean

    # 분산대신 MAD(절대표준편차) 로 변경. 제곱 후 루트는 이상치 발생시 매우 커져서 이상치 하나+비슷한 값들 조합으로도 수치 극대화 가능한 맹점이 있음
    # 분산 보다 표준편차가 직관적임
    num_group = len(aug_out)//group_size
    aug_MAD_mean = torch.stack([torch.abs(aug_out[group_size*i: group_size*(i+1)]-aug_out[group_size*i: group_size*(i+1)].mean()) for i in range(num_group)]).mean()
    aug_penalty = torch.relu(lambda_var*(threshold - aug_MAD_mean))
   
    return aug_penalty, aug_MAD_mean.detach()

def aug_chunk_loader(dataset, aug_index, batch_size=500):
    subset = Subset(dataset, aug_index) #len(aug_index) = (original batch size)*11. maybe 5500
    return DataLoader(subset, batch_size=batch_size, shuffle=False)




class Run_deep_Rf_delta_var_penalty_97():

    def __init__(self, device, relay, model, loss_fn, optimizer, scheduler, callback, last_ckpt_path, opt_ckpt1_path, opt_ckpt2_path, start_p, early_stopping=False, total_epoch = 1000):

        # relay : True or False --> True 일 경우 이어서 학습
        self.device = device
        self.relay = relay
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callback = callback
        self.start_p = start_p
        self.start_epoch = 0
        self.total_epoch = total_epoch
        self.error_log_train= []
        self.r2_log_train=[]
        self.error_log_valid = []
        self.r2_log_valid=[]
        self.mono_penalty=[0]
        self.var_penalty=[0]
        self.early_stopping = early_stopping

        ########## penalty ############
        self.alpha_mono = torch.ones(8, device = self.device)       # 초기에는 제약 가중치 0 (제약 없음)
        self.alpha_var = torch.tensor(1.0, device = self.device)
        self.alpha_max   = 100.0      # 너무 커지면 학습 폭주할 수 있으니 상한을 둠
        self.eta = 0.5 
        
        if self.relay:
            self.last_ckpt = torch.load(last_ckpt_path)
            self.opt_ckpt1_path = torch.load(opt_ckpt1_path)
            self.opt_ckpt2_path = torch.load(opt_ckpt2_path)
            self.model.load_state_dict(self.last_ckpt['model_state_dict'])
            self.optimizer.load_state_dict(self.last_ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(self.last_ckpt['scheduler_state_dict'])
            self.start_epoch = self.last_ckpt['epoch']+1

            self.error_log_train = self.last_ckpt.get('train_loss', [])
            self.r2_log_train = self.last_ckpt.get('train_r2', [])
            self.error_log_valid = self.last_ckpt.get('valid_loss', [])
            self.r2_log_valid = self.last_ckpt.get('valid_r2', [])
            self.mono_penalty =  self.last_ckpt.get('mono_penalty', [])
            self.var_penalty =  self.last_ckpt.get('variance_penalty', [])
            self.alpha_mono =  self.last_ckpt.get('alpha_mono', torch.ones(8)).to(self.device)
            self.alpha_var =  self.last_ckpt.get('alpha_var', torch.tensor(1)).to(self.device)

            if self.start_epoch >=  self.total_epoch:
                raise ValueError(f'총 epoch: {self.total_epoch}가 이전에 저장된 epoch: {self.start_epoch} 보다 작음')
               
    def __call__(self, com_train_loader, elu1_train_loader, elu2_train_loader,
                 com_valid_loader, elu1_valid_loader, elu2_valid_loader, compound_aug_set, elu1_aug_set, elu2_aug_set):
        
        for epoch in tqdm(range(self.start_epoch, self.total_epoch)):
            
            #train set (+ augmentation set)
            self.model.train()
            epoch_loss_train = 0
            epoch_r2_train=0
            num_train_batch = 0
            epoch_mono_penalty_train=torch.zeros(8, device =self.device )
            epoch_var_penalty_train=0   
            aug_mono_epoch=0
            aug_var_epoch=0
            for batch_idx,data_batch_train in enumerate(zip(com_train_loader, elu1_train_loader, elu2_train_loader)):
                com_train_batch = data_batch_train[0].to(self.device)
                elu1_train_batch = data_batch_train[1].to(self.device)
                elu2_train_batch = data_batch_train[2].to(self.device)
                out = self.model(com_train_batch, elu1_train_batch, elu2_train_batch)                
                loss_1, real_loss, r2 = self.loss_fn(out, com_train_batch.y)


                if epoch < self.start_p:
                    aug_mono=0
                    aug_var=0
                    self.optimizer.zero_grad()
                    loss = loss_1
                    loss.backward()
                    self.optimizer.step()

                    #epoch_mono_penalty_train += 0
                    #epoch_var_penalty_train += 0

                elif epoch >= self.start_p :

                    #self.model.eval()
                    #aug_index 정의 (5500개 인덱스 선정하는데 그냥 순차적으로 넣자)
                    max_index = len(compound_aug_set)
                    aug_index = torch.arange((batch_idx*5500), min((batch_idx+1)*5500 , max_index))
                    aug_com_loader = aug_chunk_loader(compound_aug_set, aug_index, batch_size = 500)
                    aug_elu1_loader = aug_chunk_loader(elu1_aug_set, aug_index, batch_size = 500)
                    aug_elu2_loader = aug_chunk_loader(elu2_aug_set, aug_index, batch_size = 500)
                    aug_out_batch=[]
                    aug_elu_param_batch=[]
                    for aug_batch_mini in zip (aug_com_loader, aug_elu1_loader, aug_elu2_loader):
                        aug_com_batch_mini = aug_batch_mini[0].to(self.device)
                        aug_elu1_batch_mini = aug_batch_mini[1].to(self.device)
                        aug_elu2_batch_mini = aug_batch_mini[2].to(self.device)
                        # 일단  aug batch out(500) 보관 했다가 배치들 다 모이면 전체(5500)에서 패널티 구하자 (실제 배치 하나에 aug배치 11개)
                        aug_out_mini = self.model(aug_com_batch_mini, aug_elu1_batch_mini, aug_elu2_batch_mini)
                        aug_out_batch.append(aug_out_mini)
                        aug_elu_param_mini = aug_com_batch_mini.eluent_param.view(-1,8)
                        aug_elu_param_batch.append(aug_elu_param_mini)
                    aug_out_cat = torch.cat(aug_out_batch)
                    aug_elu_param_cat = torch.cat(aug_elu_param_batch, dim=0)

                    loss_2_list, aug_mono = monotone_order_penalty( aug_out_cat, aug_elu_param_cat, lambda_mono_list = self.alpha_mono)
                    loss_3, aug_var = variance_penalty_bn( aug_out_cat, lambda_var=self.alpha_var)
                    print(aug_var)

                    #self.model.train()   
                    self.optimizer.zero_grad()
                    loss = loss_1 + loss_2_list.sum() + loss_3   
                    loss.backward()
                    self.optimizer.step()

                    epoch_mono_penalty_train += loss_2_list.detach()  
                    epoch_var_penalty_train += loss_3.item()   
                
                epoch_loss_train += real_loss.item()
                epoch_r2_train += r2.item()
                num_train_batch += 1
                aug_mono_epoch += aug_mono
                aug_var_epoch += aug_var            

            self.error_log_train.append(epoch_loss_train / num_train_batch)
            self.r2_log_train.append(epoch_r2_train / num_train_batch)
            self.mono_penalty.append((epoch_mono_penalty_train / num_train_batch).squeeze(0).cpu()) 
            self.var_penalty.append(epoch_var_penalty_train / num_train_batch)

            print('alpha_mono : ',self.alpha_mono)
            #print('mono_p_pure : ',aug_mono_epoch/num_train_batch)
            print('alpha_var : ',self.alpha_var)
            #print('var_p_pure : ',aug_var_epoch/num_train_batch)


            # valiation set
            self.model.eval()
            epoch_loss_valid = 0
            epoch_r2_valid = 0
            num_valid_batch = 0
            with torch.no_grad():      
                for data_batch_valid in zip(com_valid_loader, elu1_valid_loader, elu2_valid_loader):
                    com_valid_batch = data_batch_valid[0].to(self.device)
                    elu1_valid_batch = data_batch_valid[1].to(self.device)
                    elu2_valid_batch = data_batch_valid[2].to(self.device)
                    pred = self.model(com_valid_batch, elu1_valid_batch, elu2_valid_batch)                    
                    loss, real_loss, r2 = self.loss_fn(pred, com_valid_batch.y)
                    epoch_loss_valid += real_loss.item()
                    epoch_r2_valid += r2.item()
                    num_valid_batch += 1
        
            self.error_log_valid.append(epoch_loss_valid / num_valid_batch)
            self.r2_log_valid.append(epoch_r2_valid / num_valid_batch)
        
            # schedular
            self.scheduler.step()
        
            # Early stop & callback
            early_stop = self.callback(epoch, self.model, self.optimizer, self.scheduler,
                           train_loss = self.error_log_train,
                           train_r2 = self.r2_log_train,
                           valid_loss = self.error_log_valid,
                           valid_r2 = self.r2_log_valid,
                           mono_penalty = self.mono_penalty,
                           var_penalty = self.var_penalty,
                           a_mono = self.alpha_mono,
                           a_var = self.alpha_var)

            mono_delta = (self.mono_penalty[-1]-self.mono_penalty[-2]).to(self.device)
            self.alpha_mono  = torch.clamp(self.alpha_mono  + self.eta *mono_delta, 0.0, self.alpha_max)
            var_delta = self.var_penalty[-1]-self.var_penalty[-2]
            self.alpha_var = torch.clamp(self.alpha_var + self.eta * var_delta, 0.0, self.alpha_max)                          
                           
            mono_str = ", ".join([f"{x:.4f}" for x in self.mono_penalty[-1].tolist()])
            log = f'Epoch : {epoch+1} | Train_loss : {self.error_log_train[-1]:.4f}, Train_r2 : {self.r2_log_train[-1]:.4f} | Valid_loss : {self.error_log_valid[-1]:.4f}, Valid_r2 : {self.r2_log_valid[-1]:.4f} | 8_mono_p : [{mono_str}], variance_p : {self.var_penalty[-1]:.4f}'

            print(log)

            logging_path = 'train&valid_logging.txt'
            with open(logging_path, 'a') as f:
                f.write(log)

            if early_stop:
                if self.early_stopping == True:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
                '''