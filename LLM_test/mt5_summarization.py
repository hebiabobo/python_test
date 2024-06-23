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
