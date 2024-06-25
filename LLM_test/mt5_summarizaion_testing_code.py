import torch
from tqdm.auto import tqdm
import numpy as np
import json
from torch.utils.data import DataLoader
from rouge import Rouge

from LLM_test.mt5_summarization_full_training_code import LCSTS, model, collote_fn, tokenizer

rouge = Rouge()  # 创建 ROUGE 计算器对象，用于计算 ROUGE 指标。

max_dataset_size = 200000
max_input_length = 512
max_target_length = 32
train_batch_size = 8
test_batch_size = 8
learning_rate = 2e-5
epoch_num = 3
beam_size = 4
no_repeat_ngram_size = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

test_data = LCSTS(r'E:\pythonProject\dataset\lcsts_tsv\data3.tsv')
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collote_fn)

model.load_state_dict(torch.load('epoch_1_valid_rouge_6.6667_model_weights.bin'))  # 加载了之前训练得到的模型的权重。

model.eval()  # 将模型设置为评估模式，这意味着在推理过程中不会进行参数更新和梯度计算。
with torch.no_grad():  # 使用上下文管理器，确保在该块中的操作不会进行梯度计算，以节省内存和计算资源。
    print('evaluating on test set...')
    sources, preds, labels = [], [], []  # sources：用于存储测试集中的源文本数据，即需要进行摘要生成的文本。
    for batch_data in tqdm(test_dataloader):
        batch_data = batch_data.to(device)
        generated_tokens = model.generate(
            batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            max_length=max_target_length,
            num_beams=beam_size,
            no_repeat_ngram_size=no_repeat_ngram_size,
        ).cpu().numpy()
        '''
            isinstance(generated_tokens, tuple)：这是 Python 中的一个内置函数，用于检查一个对象是否是指定类型的实例。
            在这里，它检查 generated_tokens 是否是元组类型。

            generated_tokens[0]: 如果 generated_tokens 是元组类型，则将其转换为元组的第一个元素。
        '''
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        label_tokens = batch_data["labels"].cpu().numpy()  # # 获取批次数据中的真实标签并转换为 NumPy 数组。

        decoded_sources = tokenizer.batch_decode(
            batch_data["input_ids"].cpu().numpy(),
            skip_special_tokens=True,
            use_source_tokenizer=True
            # 表示使用源文本的 Tokenizer 进行解码，这里假设模型在训练和推理时使用了不同的 Tokenizer，因此需要指定使用源文本的 Tokenizer 进行解码。
        )
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        '''
            label_tokens != -100 是一个条件表达式，返回一个布尔数组，表示 label_tokens 中哪些元素不等于 -100。
            如果条件为真（即不等于 -100），则保留原值，即 label_tokens 中对应位置的元素。
            如果条件为假（即等于 -100），则用 tokenizer.pad_token_id 替换该位置的元素。
        '''
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        sources += [source.strip() for source in decoded_sources]
        preds += [pred.strip() for pred in decoded_preds]  # # 将解码后的预测文本（decoded_preds）连接成字符串，并添加到名为 preds 的列表中。
        labels += [label.strip() for label in decoded_labels]
    scores = rouge.get_scores(  # rouge.get_scores(...)：使用了 ROUGE 指标计算模块中的 get_scores 方法，用于计算模型生成的摘要（hyps）与真实摘要（refs）之间的 ROUGE 指标。
        hyps=[' '.join(pred) for pred in preds],  # 将模型生成的摘要 preds 中的每个摘要列表转换为一个字符串，然后用空格连接起来，构成一个摘要列表 hyps，用于计算 ROUGE 指标。
        refs=[' '.join(label) for label in labels],  # hyps=[' '.join(pred) for pred in preds]：将模型生成的摘要 preds 中的每个摘要列表
        # 转换为一个字符串，然后用空格连接起来，构成一个摘要列表 hyps，用于计算 ROUGE 指标。
        avg=True  # 表示计算 ROUGE 指标的平均值，即计算 ROUGE-1、ROUGE-2 和 ROUGE-L 指标的平均值。
    )
    rouges = {key: value['f'] * 100 for key, value in scores.items()}
    rouges['avg'] = np.mean(list(rouges.values()))
    print(
        f"Test Rouge1: {rouges['rouge-1']:>0.2f} Rouge2: {rouges['rouge-2']:>0.2f} RougeL: {rouges['rouge-l']:>0.2f}\n")
    results = []
    print('saving predicted results...')
    '''
    使用 zip 函数将 sources、preds 和 labels 中对应位置的元素一一配对，形成一个迭代器，每次迭代取出一个元组，
    元组中的元素分别对应 sources、preds 和 labels 中对应位置的元素。
    '''
    for source, pred, label in zip(sources, preds, labels):
        results.append({  # 在每次迭代中，将配对的源文本 source、模型预测的摘要 pred 和真实的摘要 label 组成一个字典，并将这个字典添加到 results 列表中。
            "document": source,
            "prediction": pred,
            "summarization": label
        })
    '''
    将模型在测试集上生成的摘要结果按行保存到 JSON 文件 'test_data_pred.json' 中，
    每行包含一个样本的源文本、模型预测的摘要和真实的摘要，以便后续分析和查看模型生成的摘要结果。
    '''
    with open('test_data_pred.json', 'wt', encoding='utf-8') as f:  # 使用 with 上下文管理器打开文件 'test_data_pred.json'，以写入模式打开，并指定编码为 UTF-8。
        for exapmle_result in results:
            '''
            将每个字典 example_result 转换为 JSON 格式的字符串，使用 json.dumps 方法进行转换。
            ensure_ascii=False 表示不将非 ASCII 字符转义为 Unicode 转义字符。
            然后将转换后的 JSON 字符串写入文件，并在字符串末尾添加换行符。
            '''
            f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')
