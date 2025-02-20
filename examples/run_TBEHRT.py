#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, 'C:/Project/HealthDataInst/T-BEHRT/Targeted-BEHRT')

import os
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_pretrained_bert as Bert

from  pytorch_pretrained_bert import optimizer
# import sklearn.metrics as skm
from torch.utils.data.dataset import Dataset
from src.utils import *
from src.model import *
from src.data import *

from torch import optim as toptimizer




# 根据不同的beta类型计算beta值
def get_beta(batch_idx, m, beta_type):
    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard": 
        beta = 1 / m
    else:
        beta = 0
    return beta


# 无监督训练函数
def trainunsup(e, sched, patienceMetric, MEM=True):
    """
    无监督训练函数,实现论文中提到的两部分MEM(Masked EHR Modeling)训练:
    1. 时序变量建模(temporal variable modeling):类似MLM,预测被掩盖的医疗记录
    2. 静态变量建模(static variable modeling):使用VAE对静态变量(如性别、地区等)进行建模
    
    Args:
        e: 当前训练轮数
        sched: 学习率调度器
        patienceMetric: 早停指标
        MEM: 是否使用MEM训练,默认为True
    Returns:
        sched: 更新后的学习率调度器
        patienceMetric: 更新后的早停指标
    """
    # 重置数据索引,确保每轮训练使用相同的数据顺序
    sampled = datatrain.reset_index(drop=True)
    
    # 创建数据集对象,处理输入数据
    # TBEHRT_data_formation类用于处理和格式化医疗数据:
    # - BertVocab['token2idx']: 将医疗代码映射为数字ID的词典
    # - sampled: 输入的医疗数据样本
    # - code='code': 指定包含医疗代码的列名
    # - age='age': 指定包含年龄信息的列名 
    # - year='year': 指定包含年份信息的列名
    # - static='static': 指定包含静态特征(如性别等)的列名
    # - max_len: 时序数据的最大长度,超过会被截断
    # - expColumn: 指定解释标签的列名
    # - outcomeColumn: 指定结果标签的列名
    # - yvocab: 年份到ID的映射词典
    # - list2avoid: 需要避免的医疗代码列表,默认为None
    # - MEM: 是否使用MEM(Masked EHR Modeling)训练模式
    Dset = TBEHRT_data_formation(BertVocab['token2idx'], sampled, code= 'code', 
                                 age = 'age', year = 'year' , static= 'static' , 
                                 max_len=global_params['max_len_seq'],expColumn='explabel', outcomeColumn='label',  
                                 yvocab=YearVocab['token2idx'], list2avoid=None, MEM=MEM)
    
    # 创建PyTorch数据加载器DataLoader:
    # - dataset=Dset: 使用上面创建的数据集
    # - batch_size: 每个批次的样本数,由global_params指定
    # - shuffle=True: 随机打乱数据顺序
    # - num_workers=3: 使用3个子进程并行加载数据
    # - sampler=None: 不使用自定义采样器,使用默认的随机采样
    trainload = DataLoader(dataset=Dset, batch_size=global_params['batch_size'], shuffle=True, num_workers=3,
                           sampler=None)

    # 设置模型为训练模式
    model.train()
    
    # 初始化训练过程中的各种计数器和损失值
    tr_loss = 0  # 总训练损失
    temp_loss = 0  # 临时损失(用于定期打印)
    nb_tr_examples, nb_tr_steps = 0, 0  # 样本数和步数计数器
    oldloss = 10 ** 10  # 用于比较损失变化的旧损失值
    
    # 遍历每个batch进行训练
    for step, batch in enumerate(trainload):
        # 将batch中的每个张量t移动到指定的设备(GPU或CPU)上
        # global_params['device']指定了使用的设备
        # 使用tuple()而不是list的原因:
        # 1. 元组是不可变的,可以防止训练过程中意外修改batch数据
        # 2. 元组比列表更节省内存
        # 3. PyTorch的DataLoader默认返回元组,保持数据类型一致性
        batch = tuple(t.to(global_params['device']) for t in batch)

        # 解包batch数据,包括时序数据和静态数据
        age_ids, input_ids, input_idsMLM, posi_ids, segment_ids, year_ids, attMask, masked_label, outcome_label, treatment_label, vaelabel = batch

        # 前向传播,获取各种损失和预测结果
        # masked_lm_loss: 时序变量MEM损失
        # vaelosspure: 静态变量MEM(VAE)损失
        masked_lm_loss, lossT, pred, label, treatOut, treatLabel, out, outLabel, treatindex, targreg, vaelosspure = model(
            input_idsMLM,
            age_ids,
            segment_ids,
            posi_ids,
            year_ids,
            attention_mask=attMask,
            masked_lm_labels=masked_label,
            outcomeT=outcome_label,
            treatmentCLabel=treatment_label,
            fullEval=False,
            vaelabel=vaelabel)
            
        # 获取VAE损失(静态变量MEM损失)
        vaeloss = vaelosspure['loss']

        # 计算总损失(这里只使用时序变量MEM损失)
        totalL = masked_lm_loss
        
        # 如果使用梯度累积,则对损失进行相应缩放
        if global_params['gradient_accumulation_steps'] > 1:
            totalL = totalL / global_params['gradient_accumulation_steps']
            
        # 反向传播计算梯度
        totalL.backward()
        
        # 获取处理后的预测结果
        treatFull = treatOut
        treatLabelFull = treatLabel
        treatLabelFull = treatLabelFull.cpu().detach()

        outFull = out
        outLabelFull = outLabel
        treatindex = treatindex.cpu().detach().numpy()
        # 获取treatment=0的样本索引及对应预测
        zeroind = np.where(treatindex == 0)
        outzero = outFull[0][zeroind]
        outzeroLabel = outLabelFull[zeroind]

        # 累计损失值
        temp_loss += totalL.item()
        tr_loss += totalL.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        # 每600步打印一次VAE相关损失(静态变量MEM损失)
        if step % 600 == 0:
            print([(keyvae, valvae) for (keyvae, valvae) in vaelosspure.items() if
                   keyvae in ['loss', 'Reconstruction_Loss', 'KLD']])
            # 检查损失是否增加,用于学习率调整
            if oldloss < vaelosspure['loss']:
                patienceMetric = patienceMetric + 1
                if patienceMetric >= 10:
                    sched.step()
                    print("LR: ", sched.get_lr())
                    patienceMetric = 0
            oldloss = vaelosspure['loss']

        # 每200步打印训练状态,包括MLM准确率和VAE损失
        if step % 200 == 0:
            precOut0 = -1
            if len(zeroind[0]) > 0:
                precOut0, _, _ = OutcomePrecision(outzero, outzeroLabel, False)

            print(
                "epoch: {0}| Loss: {1:6.5f}\t| MLM: {2:6.5f}\t| TOutP: {3:6.5f}\t|vaeloss: {4:6.5f}\t|ExpP: {5:6.5f}".format(
                    e, temp_loss / 200, cal_acc(label, pred), precOut0, vaeloss,
                    cal_acc(treatLabelFull, treatFull, False)))
            temp_loss = 0

        # 每累积指定步数后进行参数更新
        if (step + 1) % global_params['gradient_accumulation_steps'] == 0:
            optim.step()  # 更新参数
            optim.zero_grad()  # 清空梯度

    # 清理内存
    del sampled, Dset, trainload
    return sched, patienceMetric


# 多任务训练函数 - 对应论文中的完整目标函数训练(等式6)
def train_multi(e, MEM=True):
    sampled = datatrain.reset_index(drop=True)

    # 创建数据集 - 包含时序和静态变量
    Dset =  TBEHRT_data_formation(BertVocab['token2idx'], sampled, code= 'code', 
                                 age = 'age', year = 'year' , static= 'static' , 
                                 max_len=global_params['max_len_seq'],expColumn='explabel', outcomeColumn='label',  
                                 yvocab=YearVocab['token2idx'], list2avoid=None, MEM=MEM)
    
    # 创建数据加载器    
    trainload = DataLoader(dataset=Dset, batch_size=global_params['batch_size'], shuffle=True, num_workers=3,
                           sampler=None)
    
    # 设置模型为训练模式
    model.train()
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    
    # 遍历每个batch进行训练
    for step, batch in enumerate(trainload):
        batch = tuple(t.to(global_params['device']) for t in batch)

        # 获取输入数据:包含时序数据(input_idsMLM)和静态数据(vaelabel)
        age_ids, input_ids, input_idsMLM, posi_ids, segment_ids, year_ids, attMask, masked_label, outcome_label, treatment_label, vaelabel = batch
        
        # 前向传播 - 计算三部分损失:
        # 1. masked_lm_loss: 时序MEM损失(等式4)
        # 2. vaelosspure: 静态MEM损失(等式5) 
        # 3. lossT: 监督学习损失(等式3)
        masked_lm_loss, lossT, pred, label, treatOut, treatLabel, out, outLabel, treatindex, targreg, vaelosspure = model(
            input_idsMLM,
            age_ids,
            segment_ids,
            posi_ids,
            year_ids,
            attention_mask=attMask,
            masked_lm_labels=masked_label,
            outcomeT=outcome_label,
            treatmentCLabel=treatment_label,
            fullEval=False,
            vaelabel=vaelabel)

        vaeloss = vaelosspure['loss']
        
        # 计算总损失 - 对应论文等式(6)的加权组合
        # global_params['fac']对应论文中的超参数δ
        totalL = 1 * (lossT) + 0 + (global_params['fac'] * masked_lm_loss)
        if global_params['gradient_accumulation_steps'] > 1:
            totalL = totalL / global_params['gradient_accumulation_steps']
            
        # 反向传播
        totalL.backward()
        
        treatFull = treatOut
        treatLabelFull = treatLabel
        treatLabelFull = treatLabelFull.cpu().detach()

        outFull = out
        outLabelFull = outLabel
        treatindex = treatindex.cpu().detach().numpy()
        zeroind = np.where(treatindex == 0)
        outzero = outFull[0][zeroind]
        outzeroLabel = outLabelFull[zeroind]

        # 累计损失
        temp_loss += totalL.item()
        tr_loss += totalL.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        # 每200步打印训练状态,包括:
        # - MLM准确率(时序MEM)
        # - VAE损失(静态MEM) 
        # - 倾向性预测和事实结果预测的准确率
        if step % 200 == 0:
            precOut0 = -1

            if len(zeroind[0]) > 0:
                precOut0, _, _ = OutcomePrecision(outzero, outzeroLabel, False)

            print(
                "epoch: {0}| Loss: {1:6.5f}\t| MLM: {2:6.5f}\t| TOutP: {3:6.5f}\t|vaeloss: {4:6.5f}\t|ExpP: {5:6.5f}".format(
                    e, temp_loss / 200, cal_acc(label, pred), precOut0, vaeloss,
                    cal_acc(treatLabelFull, treatFull, False)))

            print([(keyvae, valvae) for (keyvae, valvae) in vaelosspure.items() if
                   keyvae in ['loss', 'Reconstruction_Loss', 'KLD']])
            temp_loss = 0

        # 梯度更新
        if (step + 1) % global_params['gradient_accumulation_steps'] == 0:
            optim.step()
            optim.zero_grad()

    # 清理内存
    del sampled, Dset, trainload


# 多次评估函数
def evaluation_multi_repeats():
    # 设置模型为评估模式
    model.eval()
    y = []
    y_label = []
    t_label = []
    t_output = []
    count = 0
    totalL = 0
    
    # 遍历测试数据进行评估
    for step, batch in enumerate(testload):
        model.eval()
        count = count + 1
        batch = tuple(t.to(global_params['device']) for t in batch)

        age_ids, input_ids, input_idsMLM, posi_ids, segment_ids, year_ids, attMask, masked_label, outcome_label, treatment_label, vaelabel = batch
        
        # 不计算梯度
        with torch.no_grad():
            masked_lm_loss, lossT, pred, label, treatOut, treatLabel, out, outLabel, treatindex, targreg, vaelosspure = model(
                input_idsMLM,
                age_ids,
                segment_ids,
                posi_ids,
                year_ids,
                attention_mask=attMask,
                masked_lm_labels=masked_label,
                outcomeT=outcome_label,
                treatmentCLabel=treatment_label, vaelabel=vaelabel)

        # 累计损失
        totalL = totalL + lossT.item() + 0 + (global_params['fac'] * masked_lm_loss)
        
        treatFull = treatOut
        treatLabelFull = treatLabel
        treatLabelFull = treatLabelFull.detach()
        outFull = out
        outLabelFull = outLabel
        treatindex = treatindex.cpu().detach().numpy()
        
        # 收集每个处理的预测结果
        outPred = []
        outexpLab = []
        for el in range(global_params['treatments']):
            zeroind = np.where(treatindex == el)
            outPred.append(outFull[el][zeroind])
            outexpLab.append(outLabelFull[zeroind])

        # 收集标签和预测
        y_label.append(torch.cat(outexpLab))
        y.append(torch.cat(outPred))

        treatOut = treatFull.cpu()
        treatLabel = treatLabelFull.cpu()
        
        # 每200步打印评估状态
        if step % 200 == 0:
            print(step, "tempLoss:", totalL / count)

        t_label.append(treatLabel)
        t_output.append(treatOut)

    # 合并所有预测结果
    y_label = torch.cat(y_label, dim=0)
    y = torch.cat(y, dim=0)
    t_label = torch.cat(t_label, dim=0)
    treatO = torch.cat(t_output, dim=0)

    # 计算精确度和ROC AUC
    tempprc, output, label = precision_test(y, y_label, False)
    treatPRC = cal_acc(t_label, treatO, False)
    tempprc2, output2, label2 = roc_auc(y, y_label, False)

    print("LossEval: ", float(totalL) / float(count))

    return tempprc, tempprc2, treatPRC, float(totalL) / float(count)


# 完整评估函数
def fullEval_4analysis_multi(tr, te, filetest):
    if tr:
        sampled = datatrain.reset_index(drop=True)

    if te:
        data = filetest

        if tr:
            sampled = pd.concat([sampled, data]).reset_index(drop=True)
        else:
            sampled = data
            
    # 创建数据集
    Fulltset = TBEHRT_data_formation(BertVocab['token2idx'], sampled, code= 'code', 
                                 age = 'age', year = 'year' , static= 'static' , 
                                 max_len=global_params['max_len_seq'],expColumn='explabel', outcomeColumn='label',  
                                 yvocab=YearVocab['token2idx'], list2avoid=None, MEM=False)
    
    # 创建数据加载器    
    fullDataLoad = DataLoader(dataset=Fulltset, batch_size=int(global_params['batch_size']), shuffle=False,
                              num_workers=0)

    # 设置模型为评估模式
    model.eval()
    y = []
    y_label = []
    t_label = []
    t_output = []
    count = 0
    totalL = 0
    eps_array = []

    # 初始化预测列表
    for yyy in range(model_config['num_treatment']):
        y.append([yyy])
        y_label.append([yyy])

    print(y)
    
    # 遍历数据进行评估
    for step, batch in enumerate(fullDataLoad):
        model.eval()

        count = count + 1
        batch = tuple(t.to(global_params['device']) for t in batch)

        age_ids, input_ids, input_idsMLM, posi_ids, segment_ids, year_ids, attMask, masked_label, outcome_label, treatment_label, vaelabel = batch

        # 不计算梯度
        with torch.no_grad():
            masked_lm_loss, lossT, pred, label, treatOut, treatLabel, out, outLabel, treatindex, targreg, vaeloss = model(
                input_idsMLM,
                age_ids,
                segment_ids,
                posi_ids,
                year_ids,
                attention_mask=attMask,
                masked_lm_labels=masked_label,
                outcomeT=outcome_label,
                treatmentCLabel=treatment_label, fullEval=True, vaelabel=vaelabel)

        outFull = out
        outLabelFull = outLabel

        # 收集每个处理的预测结果
        for el in range(global_params['treatments']):
            y[el].append(outFull[el].cpu())
            y_label[el].append(outLabelFull.cpu())

        totalL = totalL + (1 * (lossT)).item()

        # 每200步打印评估状态
        if step % 200 == 0:
            print(step, "tempLoss:", totalL / count)

        t_label.append(treatLabel)
        t_output.append(treatOut)

    # 合并预测结果
    for idd, elem in enumerate(y):
        elem = torch.cat(elem[1:], dim=0)
        y[idd] = elem
    for idd, elem in enumerate(y_label):
        elem = torch.cat(elem[1:], dim=0)
        y_label[idd] = elem

    t_label = torch.cat(t_label, dim=0)
    treatO = torch.cat(t_output, dim=0)
    treatPRC = cal_acc(t_label, treatO)

    print("LossEval: ", float(totalL) / float(count), "prec treat:", treatPRC)
    return y, y_label, t_label, treatO, treatPRC, eps_array


# 转换预测结果的格式
def fullCONV(y, y_label, t_label, treatO):
    # 转换多热编码
    def convert_multihot(label, pred):
        label = label.cpu().numpy()
        truepred = pred.detach().cpu().numpy()
        truelabel = label
        newpred = []
        for i, x in enumerate(truelabel):
            temppred = []
            temppred.append(truepred[i][0])
            temppred.append(truepred[i][x[0]])
            newpred.append(temppred)
        return truelabel, np.array(truepred)

    # 转换二进制标签
    def convert_bin(logits, label, treatmentlabel2):
        output = logits
        label, output = label.cpu().numpy(), output.detach().cpu().numpy()
        label = label[treatmentlabel2[0]]
        return label, output

    # 转换处理标签
    treatmentlabel2, treatment2 = convert_multihot(t_label, treatO)
    
    # 重塑预测结果
    y = torch.cat(y, dim=0).view(global_params['treatments'], -1)
    y = y.transpose(1, 0)
    y_label = torch.cat(y_label, dim=0).view(global_params['treatments'], -1)
    y_label = y_label.transpose(1, 0)
    
    # 转换预测结果格式
    y2 = []
    y2label = []
    for i, elem in enumerate(y):
        j, k = convert_bin(elem, y_label[i], treatmentlabel2[i])
        y2.append(k)
        y2label.append(j)
    y2 = np.array(y2)
    y2label = np.array(y2label)
    y2label = np.expand_dims(y2label, -1)

    return y2, y2label, treatmentlabel2, treatment2



file_config = {
       'data':  'test.parquet',
}
optim_config = {
    'lr': 1e-4,
    'warmup_proportion': 0.1
}


BertVocab = {}
token2idx = {'MASK': 4,
  'CLS': 3,
  'SEP': 2,
  'UNK': 1,
  'PAD': 0,
            'disease1':5,
             'disease2':6,
             'disease3':7,
             'disease4':8,
             'disease5':9,
             'disease6':10,
             'medication1':11,
             'medication2':12,
             'medication3':13,
             'medication4':14,
             'medication5':15,
             'medication6':16,
            }
idx2token = {}
for x in token2idx:
    idx2token[token2idx[x]]=x
BertVocab['token2idx']= token2idx
BertVocab['idx2token']= idx2token





YearVocab = {'token2idx': {'PAD': 0,
  '1987': 1,
  '1988': 2,
  '1989': 3,
  '1990': 4,
  '1991': 5,
  '1992': 6,
  '1993': 7,
  '1994': 8,
  '1995': 9,
  '1996': 10,
  '1997': 11,
  '1998': 12,
  '1999': 13,
  '2000': 14,
  '2001': 15,
  '2002': 16,
  '2003': 17,
  '2004': 18,
  '2005': 19,
  '2006': 20,
  '2007': 21,
  '2008': 22,
  '2009': 23,
  '2010': 24,
  '2011': 25,
  '2012': 26,
  '2013': 27,
  '2014': 28,
  '2015': 29,
  'UNK': 30},
 'idx2token': {0: 'PAD',
  1: '1987',
  2: '1988',
  3: '1989',
  4: '1990',
  5: '1991',
  6: '1992',
  7: '1993',
  8: '1994',
  9: '1995',
  10: '1996',
  11: '1997',
  12: '1998',
  13: '1999',
  14: '2000',
  15: '2001',
  16: '2002',
  17: '2003',
  18: '2004',
  19: '2005',
  20: '2006',
  21: '2007',
  22: '2008',
  23: '2009',
  24: '2010',
  25: '2011',
  26: '2012',
  27: '2013',
  28: '2014',
  29: '2015',
  30: 'UNK'}}



global_params = {
    'batch_size': 128,
    'gradient_accumulation_steps': 1,
    'num_train_epochs': 3,
    'device': 'cuda:0',
    'output_dir': "save_models",
    'save_model': True,
    'max_len_seq': 250,
    'max_age': 110,
    'age_year': False,
    'age_symbol': None,
    'fac': 0.1,
    'diseaseI': 1,
    'treatments': 2
}

ageVocab, _ = age_vocab(max_age=global_params['max_age'], year=global_params['age_year'],
                        symbol=global_params['age_symbol'])

model_config = {
    'vocab_size': len(BertVocab['token2idx'].keys()),  # number of disease + symbols for word embedding
    'hidden_size': 150,  # word embedding and seg embedding hidden size
    'seg_vocab_size': 2,  # number of vocab for seg embedding
    'age_vocab_size': len(ageVocab.keys()),  # number of vocab for age embedding
    'max_position_embedding': global_params['max_len_seq'],  # maximum number of tokens
    'hidden_dropout_prob': 0.3,  # dropout rate
    'num_hidden_layers': 4,  # number of multi-head attention layers required
    'num_attention_heads': 6,  # number of attention heads
    'attention_probs_dropout_prob': 0.4,  # multi-head attention dropout rate
    'intermediate_size': 108,  # the size of the "intermediate" layer in the transformer encoder
    'hidden_act': 'gelu',
    'initializer_range': 0.02,  # parameter weight initializer range
    'num_treatment': global_params['treatments'],
    'device': global_params['device'],
    'year_vocab_size': len(YearVocab['token2idx'].keys()),

    'batch_size': global_params['batch_size'],
    'MEM': True,
    'poolingSize': 50,
    'unsupVAE': True,
    'unsupSize': ([[3,2]] *22) ,
    'vaelatentdim': 40,
    'vaehidden': 50,
    'vaeinchannels':39,



}


# 读取数据
data = pd.read_parquet (file_config['data'])

# 创建5折交叉验证对象
kf = KFold(n_splits = 5, shuffle = True, random_state = 2)

print('Begin experiments....')

# 进行5次交叉验证实验
for cutiter in (range(5)):
    print("_________________\nfold___" + str(cutiter) + "\n_________________")
    data = pd.read_parquet (file_config['data'])

    # 获取当前折的训练集和测试集索引
    result = next(kf.split(data), None)

    # 根据索引分割数据集
    # result[0]包含了当前折的训练集索引
    # data.iloc[result[0]]使用这些索引从原始数据中选择训练样本
    # reset_index(drop=True)重置索引,确保训练集的索引从0开始连续
    # 最终得到的datatrain就是当前折的训练数据集
    datatrain = data.iloc[result[0]].reset_index(drop=True)
    testdata =  data.iloc[result[1]].reset_index(drop=True)

    # 构建测试数据集
    tset = TBEHRT_data_formation(BertVocab['token2idx'], testdata, code= 'code', 
                                 age = 'age', year = 'year' , static= 'static' , 
                                 max_len=global_params['max_len_seq'],expColumn='explabel', outcomeColumn='label',  
                                 yvocab=YearVocab['token2idx'], list2avoid=None, MEM=False)
    
    # 创建测试数据加载器    
    testload = DataLoader(dataset=tset, batch_size=int(global_params['batch_size']), shuffle=False, num_workers=0)

    # 配置模型参数
    model_config['klpar']= float(1.0/(len(datatrain)/global_params['batch_size']))
    conf = BertConfig(model_config)
    model = TBEHRT(conf, 1)

    # 配置优化器
    optim = optimizer.adam(params=list(model.named_parameters()), config=optim_config)

    # 设置模型保存路径
    model_to_save_name =  'TBEHRT_Test' + "__CUT" + str(cutiter) + ".bin"

    import warnings
    warnings.filterwarnings(action='ignore')

    # 设置学习率调度器
    scheduler = toptimizer.lr_scheduler.ExponentialLR(optim, 0.95, last_epoch=-1)
    patience = 0
    best_pre = -100000000000000000000
    LossC = 0.1

    # 无监督预训练
    for e in range(2):
        scheduler , patience= trainunsup(e, scheduler, patience)

    # 有监督训练
    for e in range(2):
        train_multi(e)
        # 评估模型性能
        auc, auroc, auc2, loss = evaluation_multi_repeats()
        aucreal = -1 * loss
        
        # 保存最佳模型
        if aucreal > best_pre:
            patience = 0
            print("** ** * Saving best fine - tuned model ** ** * ")
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(global_params['output_dir'], model_to_save_name)
            create_folder(global_params['output_dir'])
            if global_params['save_model']:
                torch.save(model_to_save.state_dict(), output_model_file)

            best_pre = aucreal
            print("auc-mean: ", aucreal)
        else:
            # 学习率衰减
            if patience % 2 == 0 and patience != 0:
                scheduler.step()
                print("LR: ", scheduler.get_lr())

            patience = patience + 1
        print('auprc : {}, auroc : {}, Treat-auc : {}, time: {}'.format(auc, auroc, auc2, "long....."))

    # 最终评估
    LossC = 0.1
    conf = BertConfig(model_config)
    model = TBEHRT(conf, 1)
    optim = optimizer.VAEadam(params=list(model.named_parameters()), config=optim_config)
    output_model_file = os.path.join(global_params['output_dir'], model_to_save_name)
    model = toLoad(model, output_model_file)

    # 进行全面评估并获取结果
    y, y_label, t_label, treatO, tprc, eps = fullEval_4analysis_multi(False, True, testdata)
    y2, y2label, treatmentlabel2, treatment2 = fullCONV(y, y_label, t_label, treatO)

    # 保存评估结果
    NPSaveNAME =  'TBEHRT_Test' + "__CUT" + str(cutiter) + ".npz"
    np.savez(  NPSaveNAME,
             outcome=y2,
             outcome_label=y2label, treatment=treatment2, treatment_label=treatmentlabel2,
             epsilon=np.array([0]))
             
    # 清理内存
    del y, y_label, t_label, treatO, tprc, eps, y2, y2label, treatmentlabel2, treatment2, datatrain, conf, model, optim, output_model_file,  best_pre, LossC,
    print("\n\n\n\n\n")

