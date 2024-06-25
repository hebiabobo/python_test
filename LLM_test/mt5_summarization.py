from torch.utils.data import Dataset

max_dataset_size = 200000


class LCSTS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                items = line.strip().split('!=!')
                assert len(items) == 2
                Data[idx] = {
                    'title': items[0],
                    'content': items[1]
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_data = LCSTS(r'E:\pythonProject\dataset\lcsts_tsv\data1.tsv')
valid_data = LCSTS(r'E:\pythonProject\dataset\lcsts_tsv\data2.tsv')
test_data = LCSTS(r'E:\pythonProject\dataset\lcsts_tsv\data3.tsv')

print(f'train set size: {len(train_data)}')
print(f'valid set size: {len(valid_data)}')
print(f'test set size: {len(test_data)}')
print(next(iter(train_data)))

# 数据预处理
from transformers import AutoTokenizer

model_checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

inputs = tokenizer("我叫张三，在华东交通大学学习计算机。")
print(inputs)
print(tokenizer.convert_ids_to_tokens(inputs.input_ids))
# {'input_ids': [259, 3003, 27333, 8922, 2092, 261, 1083, 8423, 8854, 29503, 9792, 24920, 123553, 306, 1],
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
# ['▁', '我', '叫', '张', '三', ',', '在', '华', '东', '交通', '大学', '学习', '计算机', '。', '</s>']
'''
特殊的 Unicode 字符 ▁ 以及序列结束 token </s> 表明 mT5 模型采用的是基于 Unigram 切分算法的 SentencePiece 分词器。
Unigram 对于处理多语言语料库特别有用，它使得 SentencePiece 可以在不知道重音、标点符号以及没有空格分隔字符（例如中文）的情况下对文本进行分词。
'''

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM

max_input_length = 512
max_target_length = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
print(torch.version.cuda)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)


def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['content'])
        batch_targets.append(sample['title'])
    batch_data = tokenizer(
        batch_inputs,
        padding=True,  # True或'longest'：填充到批次中最长的序列（如果只提供单个序列，则不应用填充）。
        max_length=max_input_length,
        truncation=True,
        # True或'longest_first'：截断为参数指定的最大长度max_length，或模型接受的最大长度（如果未max_length提供max_length=None
        # ）。这将逐个标记截断，从对中最长的序列中删除一个标记，直到达到适当的长度。
        return_tensors="pt"  # "pt"：返回 PyTorch 张量（torch.Tensor）
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets,
            padding=True,
            max_length=max_target_length,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx + 1:] = -100
        batch_data['labels'] = labels
    return batch_data


train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collote_fn)

'''
打印出一个 batch 的数据，验证是否处理正确
'''
batch = next(iter(train_dataloader))
print(batch.keys())
print('batch shape:', {k: v.shape for k, v in batch.items()})
print(batch)

from tqdm.auto import tqdm


def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))  # 使用 tqdm 创建一个进度条
    progress_bar.set_description(f'loss: {0:>7f}')  # 设置初始描述为损失为 0。
    finish_batch_num = (epoch - 1) * len(dataloader)  # 计算在之前的 epoch 中已完成的批次数。

    model.train()
    '''
    enumerate(dataloader, start=1)：遍历 dataloader 中的每个批次数据，并从1开始为每个批次数据进行编号。
    batch 是当前批次的编号。
    batch_data 是当前批次的数据。
    '''
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(device)  # batch_data.to(device)：将当前批次的数据移动到指定的设备（通常是 GPU 或 CPU）。
        outputs = model(**batch_data)  # batch_data 是一个包含模型输入数据的字典，通过解包操作 ** 将字典中的键值对作为参数传递给模型。(值)
        loss = outputs.loss

        optimizer.zero_grad()  # optimizer.zero_grad() 是 PyTorch
        # 中用于清除优化器中累积梯度的函数。在每次反向传播之前调用这个函数是很重要的，否则梯度会累积，从而导致梯度计算错误和模型训练不稳定。
        loss.backward()
        optimizer.step()  # 利用计算得到的梯度更新模型参数。具体更新方式取决于优化器的类型（例如 SGD、Adam 等）
        lr_scheduler.step()  #

        total_loss += loss.item()  # 将当前批次的损失值累加到 total_loss，用来跟踪一个 epoch 中所有批次的总损失。
        '''
        在训练过程中动态更新进度条的描述，显示当前的平均损失。
        
        progress_bar 是一个 tqdm 进度条对象，用于在训练循环中可视化进度。
        set_description 方法用于设置进度条的描述文本。
        f'loss: {total_loss / (finish_batch_num + batch):>7f}' 是一个格式化字符串，动态计算并显示当前的平均损失。
        
        finish_batch_num: 已完成的批次数量，通常等于 (epoch-1) * len(dataloader)，表示在当前 epoch 之前已经完成的总批次数。
        
        f'loss: {total_loss / (finish_batch_num + batch):>7f}':
        使用 Python 的 f-string 格式化方法，动态插入计算得到的平均损失值。
        :>7f 指定了数值的格式，其中：
        > 表示右对齐。
        7 表示总长度至少为 7 个字符（包括小数点）。
        f 表示浮点数格式。
        '''
        progress_bar.set_description(f'loss: {total_loss / (finish_batch_num + batch):>7f}')
        progress_bar.update(1)  # 用于更新进度条。每调用一次该方法，进度条将前进指定的步数（在这里是 1 步）。这在训练循环中非常有用，因为它可以直观地显示训练过程的进度。
    return total_loss
