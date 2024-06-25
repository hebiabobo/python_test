import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from LLM_test.mt5_summarization import LCSTS, test_loop, collote_fn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model_checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

# print(model)

# for i in model.named_parameters():
#     print(f"{i[0]} -> {i[1].device}")

article_text = """
受众在哪里，媒体就应该在哪里，媒体的体制、内容、技术就应该向哪里转变。
媒体融合关键是以人为本，即满足大众的信息需求，为受众提供更优质的服务。
这就要求媒体在融合发展的过程中，既注重技术创新，又注重用户体验。
"""

input_ids = tokenizer(
    article_text,
    return_tensors="pt",
    truncation=True,
    max_length=512
).to(device)
generated_tokens = model.generate(
    input_ids["input_ids"],
    attention_mask=input_ids["attention_mask"],
    max_length=32,
    no_repeat_ngram_size=2,
    num_beams=4
)
summary = tokenizer.decode(
    generated_tokens[0],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
print(summary)

article_texts = [
    """
    受众在哪里，媒体就应该在哪里，媒体的体制、内容、技术就应该向哪里转变。
    媒体融合关键是以人为本，即满足大众的信息需求，为受众提供更优质的服务。
    这就要求媒体在融合发展的过程中，既注重技术创新，又注重用户体验。
    """,
    """
    新华社受权于18日全文播发修改后的《中华人民共和国立法法》，
    修改后的立法法分为“总则”“法律”“行政法规”“地方性法规、
    自治条例和单行条例、规章”“适用与备案审查”“附则”等6章，共计105条。
    """
]

input_ids = tokenizer(
    article_texts,
    padding=True,
    return_tensors="pt",
    truncation=True,
    max_length=512
).to(device)
generated_tokens = model.generate(
    input_ids["input_ids"],
    attention_mask=input_ids["attention_mask"],
    max_length=32,
    no_repeat_ngram_size=2,
    num_beams=4
)
summarys = tokenizer.batch_decode(
    generated_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
print(summarys)

'''在开始训练之前，先评估一下没有经过微调的模型在 LCSTS 测试集上的性能。'''
test_data = LCSTS(r'E:\pythonProject\dataset\lcsts_tsv\data3.tsv')
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collote_fn)

test_loop(test_dataloader, model)
