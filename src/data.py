from torch.utils.data.dataset import Dataset
import torch
from src.utils import *
# data for var autoencoder deep unsup learning with tbehrt


# 用于TBEHRT模型的数据预处理类,实现了论文中描述的数据格式化逻辑
class TBEHRT_data_formation(Dataset):
    def __init__(self, token2idx, dataframe, code='code', age='age', year='year', static='static', max_len=1000, 
                 expColumn='explabel', outcomeColumn='label', max_age=110, yvocab=None, list2avoid=None, MEM=True):
        """
        TBEHRT模型的PyTorch数据集类
        
        参数:
            token2idx: 将医疗记录代码映射为数字索引的字典
            dataframe: 包含医疗记录的pandas数据框,需要包含以下列:
                - code: 医疗代码列(诊断、药物等)
                - age: 年龄列
                - year: 年份列 
                - static: 静态特征列(性别、地区等)
            code: 医疗代码列名
            age: 年龄列名
            year: 年份列名
            static: 静态特征列名
            max_len: 序列最大长度
            expColumn: 暴露(处理)标签列名
            outcomeColumn: 结果标签列名
            max_age: 最大年龄限制
            yvocab: 年份词典,用于年份编码
            list2avoid: MEM训练中需要避免mask的代码列表
            MEM: 是否使用MEM(Masked EHR Modeling)训练模式
        """

        # 根据list2avoid过滤词典,避免mask某些特定代码
        if list2avoid is None:
            self.acceptableVoc = token2idx
        else:
            self.acceptableVoc = {x: y for x, y in token2idx.items() if x not in list2avoid}
            print("old Vocab size: ", len(token2idx), ", and new Vocab size: ", len(self.acceptableVoc))
            
        # 保存基本属性    
        self.vocab = token2idx
        self.max_len = max_len
        self.code = dataframe[code]
        self.age = dataframe[age] 
        self.year = dataframe[year]
        
        # 获取标签
        if outcomeColumn is None:
            self.label = dataframe.deathLabel
        else:
            self.label = dataframe[outcomeColumn]
            
        # 创建年龄词典    
        self.age2idx, _ = age_vocab(110, year, symbol=None)
        
        # 获取处理标签
        if expColumn is None:
            self.treatmentLabel = dataframe.diseaseLabel
        else:
            self.treatmentLabel = dataframe[expColumn]
            
        # 保存年份词典和静态特征    
        self.year2idx = yvocab
        self.codeS = dataframe[static]
        self.MEM = MEM

    def __getitem__(self, index):
        """
        获取单个样本的处理后数据
        
        返回:
            age: 年龄序列
            code2: 包含静态特征的医疗代码序列
            codeMLM: 经过mask的医疗代码序列(用于MEM训练)
            position: 位置编码
            segment: 分段标记
            year: 年份序列
            mask: 注意力掩码
            labelMLM: MLM标签
            [labelOutcome]: 结果标签
            treatmentOutcome: 处理标签
            labelcovar: 静态特征的MLM标签
        """
        # 获取原始数据
        age = self.age[index]
        code = self.code[index]
        year = self.year[index]

        # 截取最大长度
        age = age[(-self.max_len + 1):]
        code = code[(-self.max_len + 1):]
        year = year[(-self.max_len + 1):]

        # 获取处理标签
        treatmentOutcome = torch.LongTensor([self.treatmentLabel[index]])
        
        # 获取结果标签
        labelOutcome = self.label[index]
        
        # 将最后一个token设为CLS
        code[-1] = 'CLS'

        # 创建注意力掩码
        mask = np.ones(self.max_len)
        mask[:-len(code)] = 0
        mask = np.append(np.array([1]), mask)

        # 将代码转换为索引
        tokensReal, code2 = code2index(code, self.vocab)
        
        # 对序列进行padding
        year = seq_padding_reverse(year, self.max_len, token2idx=self.year2idx)
        age = seq_padding_reverse(age, self.max_len, token2idx=self.age2idx)

        # 根据MEM模式选择是否进行mask
        if self.MEM == False:
            tokens, codeMLM, labelMLM = nonMASK(code, self.vocab)
        else:
            tokens, codeMLM, labelMLM = randommaskreal(code, self.acceptableVoc)

        # 获取位置编码和分段标记
        tokens = seq_padding_reverse(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seg(tokens)

        # 对序列进行padding
        code2 = seq_padding_reverse(code2, self.max_len, symbol=self.vocab['PAD'])
        codeMLM = seq_padding_reverse(codeMLM, self.max_len, symbol=self.vocab['PAD'])
        labelMLM = seq_padding_reverse(labelMLM, self.max_len, symbol=-1)

        # 处理静态特征
        outCodeS = [int(xx) for xx in self.codeS[index]]
        fixedcovar = np.array(outCodeS)
        labelcovar = np.array(([-1] * len(outCodeS)) + [-1, -1])
        
        # MEM模式下对静态特征进行mask
        if self.MEM == True:
            fixedcovar, labelcovar = covarUnsupMaker(fixedcovar)
            
        # 合并静态特征和时序特征    
        code2 = np.append(fixedcovar, code2)
        codeMLM = np.append(fixedcovar, codeMLM)

        # 返回处理后的数据
        return torch.LongTensor(age), torch.LongTensor(code2), torch.LongTensor(codeMLM), torch.LongTensor(
            position), torch.LongTensor(segment), torch.LongTensor(year), \
               torch.LongTensor(mask), torch.LongTensor(labelMLM), torch.LongTensor(
            [labelOutcome]), treatmentOutcome,  torch.LongTensor(labelcovar)

    def __len__(self):
        """返回数据集大小"""
        return len(self.code)

