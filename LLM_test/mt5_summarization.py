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

    '''
    model.train() 是 PyTorch 中用于将模型设置为训练模式的方法。它的主要作用是启用在训练过程中特定层的行为，
    例如 dropout 和 batch normalization 层，这些层在训练和推理（测试）时的行为不同。

    在 PyTorch 中，模型的训练和评估模式的区别在于以下几个方面：

    Dropout:
    训练模式：在训练模式下，dropout 层会随机将一部分神经元的输出设为零，以防止过拟合。
    评估模式：在评估模式下，dropout 层会按比例缩放所有神经元的输出，不会将任何神经元的输出设为零。
    Batch Normalization:
    训练模式：在训练模式下，batch normalization 层会计算批次内的均值和方差，并更新其内部的移动平均值。
    评估模式：在评估模式下，batch normalization 层会使用训练期间计算的移动平均值进行归一化，而不是当前批次的均值和方差。
    '''

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


'''
在验证/测试循环中，我们首先通过 model.generate() 函数获取预测结果，
然后将预测结果和正确标签都处理为 rouge 库接受的文本列表格式（这里我们将标签序列中的 -100 替换为 pad token ID 以便于分词器解码），
最后送入到 rouge 库计算各项 ROUGE 值：
'''
import numpy as np
from rouge import Rouge

rouge = Rouge()  # 创建 ROUGE 计算器对象，用于计算 ROUGE 指标。


def test_loop(dataloader, model):  # 定义了一个名为 test_loop 的函数，接受数据加载器 dataloader 和模型 model 作为参数。
    preds, labels = [], []  # 创建两个空列表 preds 和 labels，用于存储生成的文本和真实标签。

    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        '''
        是 PyTorch 中的上下文管理器，用于在其内部关闭梯度计算。在这个上下文中，PyTorch 不会追踪张量的梯度，也不会进行梯度计算，
        这对于在评估模型、推断或者验证时非常有用，可以节省显存并提高计算效率。
        
        使用 with 语句，将需要关闭梯度计算的代码块包裹起来，这样在该代码块内的所有操作都不会计算梯度。
        '''
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_target_length,  # 设置生成的最大长度
                num_beams=4,
                no_repeat_ngram_size=2,
            ).cpu().numpy()
        '''
        isinstance(generated_tokens, tuple)：这是 Python 中的一个内置函数，用于检查一个对象是否是指定类型的实例。
        在这里，它检查 generated_tokens 是否是元组类型。
        
        generated_tokens[0]: 如果 generated_tokens 是元组类型，则将其转换为元组的第一个元素。
        '''
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        label_tokens = batch_data["labels"].cpu().numpy()  # 获取批次数据中的真实标签并转换为 NumPy 数组。

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        '''
        label_tokens != -100 是一个条件表达式，返回一个布尔数组，表示 label_tokens 中哪些元素不等于 -100。
        如果条件为真（即不等于 -100），则保留原值，即 label_tokens 中对应位置的元素。
        如果条件为假（即等于 -100），则用 tokenizer.pad_token_id 替换该位置的元素。
        '''
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [' '.join(pred.strip()) for pred in decoded_preds]  # 将解码后的预测文本（decoded_preds）连接成字符串，并添加到名为 preds 的列表中。
        labels += [' '.join(label.strip()) for label in decoded_labels]
    '''使用 ROUGE 计算器的 get_scores 方法计算 ROUGE 指标，传入生成文本 preds 和真实标签 labels，并指定 avg=True 以获取平均值。'''
    scores = rouge.get_scores(hyps=preds, refs=labels, avg=True)
    '''遍历 ROUGE 指标字典中的每个键值对，将键保持不变，而将 F1 分数乘以 100 后作为值。以便以百分比形式表示。'''
    result = {key: value['f'] * 100 for key, value in scores.items()}
    result['avg'] = np.mean(list(result.values()))
    print(f"Rouge1: {result['rouge-1']:>0.2f} Rouge2: {result['rouge-2']:>0.2f} RougeL: {result['rouge-l']:>0.2f}\n")
    return result


'''
保存模型
根据模型在验证集上的性能来调整超参数以及选出最好的模型，然后将选出的模型应用于测试集以评估最终的性能。
这里继续使用 AdamW 优化器，并且通过 get_scheduler() 函数定义学习率调度器:
'''
from transformers import AdamW, get_scheduler

learning_rate = 2e-5
epoch_num = 10

'''
这行代码使用了 Transformers 库中的 AdamW 优化器来初始化一个优化器对象，用于模型参数的优化更新。
AdamW 是 Transformers 库中的一个优化器类，用于实现 AdamW 优化算法。它是 Adam 优化器的一个变种，在解决权重衰减（Weight Decay）问题上表现更好。
model.parameters() 用于获取模型中需要优化的参数，这里将模型的参数传递给优化器，使优化器可以对这些参数进行优化。
'''
optimizer = AdamW(model.parameters(), lr=learning_rate)
'''
预热步数（warmup steps）是指在训练神经网络时，在开始正式训练之前进行的一些额外的步骤。预热步数通常用于调整学习率或其他优化器超参数，
以帮助模型更好地收敛或避免训练过程中的梯度爆炸或消失等问题。

作用和原理
1.调整学习率：预热步数常用于动态调整学习率。在预热阶段，学习率可能会逐渐增加，使得模型在开始时更多地探索参数空间，
而不会因为初始的高学习率导致训练不稳定或发散。
2.缓解梯度爆炸/消失：通过逐渐增加学习率或其他手段，可以缓解训练过程中的梯度爆炸或消失问题。这对于深层神经网络特别重要，因为在初始阶段可能会出现梯度异常的情况。

实现方法
预热步数的实现方法有多种，取决于具体的优化器或训练框架。以下是一些常见的实现方法:
线性预热：在预热阶段，学习率线性增加到设定的最大值。
多项式预热：学习率根据多项式函数逐渐增加到设定的最大值。
余弦预热：学习率根据余弦函数的形状逐渐增加，这种方法在一些 Transformer 模型中比较常见。
'''
lr_scheduler = get_scheduler(  # 使用 get_scheduler 函数初始化了一个学习率调度器，采用线性衰减的方式。
    "linear",  # 表示使用线性调度器，即学习率会随着训练步数线性地减小。
    optimizer=optimizer,  # 指定了优化器对象，即之前初始化的 AdamW 优化器对象。
    num_warmup_steps=0,  # 表示预热步数，即在开始训练的一段时间内，学习率会逐渐增加到初始学习率，这里设置为0表示不进行预热。
    num_training_steps=epoch_num * len(train_dataloader),  # 表示总的训练步数，即整个训练过程中的迭代次数，这里设置为训练周期数乘以训练数据集的批次数。
)

total_loss = 0.
best_avg_rouge = 0.  # 最佳平均 ROUGE 指标
for t in range(epoch_num):  # 对每个周期进行训练，调用 train_loop 函数进行模型训练，调用 test_loop 函数进行模型验证。
    print(f"Epoch {t + 1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t + 1, total_loss)
    valid_rouge = test_loop(valid_dataloader, model)
    print(valid_rouge)
    rouge_avg = valid_rouge['avg']
    if rouge_avg > best_avg_rouge:  # 每个周期结束后，根据验证集的 ROUGE 平均值来保存表现最好的模型权重。
        best_avg_rouge = rouge_avg
        print('saving new weights...\n')
        '''
        使用 PyTorch 的 torch.save 函数将模型的状态字典（即模型的参数）保存到文件中，以便后续可以加载和使用该模型的参数。

        具体解释如下：
        torch.save(obj, f)：将对象 obj 保存到文件 f 中。
        model.state_dict()：这是 PyTorch 模型的一个属性，表示模型的状态字典，包含了模型的所有可学习参数（权重和偏置）。
        
        epoch_{t+1}：表示当前训练周期数加一，例如，如果当前是第一轮训练（t=0），那么保存的文件名就是 epoch_1。
        valid_rouge_{rouge_avg:0.4f}：表示验证集上的 ROUGE 平均值，格式化为四位小数的浮点数。
        _model_weights.bin：表示文件的后缀名，这里是保存模型权重的二进制文件。
        '''
        torch.save(model.state_dict(), f'epoch_{t + 1}_valid_rouge_{rouge_avg:0.4f}_model_weights.bin')
print("Done!")
