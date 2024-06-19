import textattack
from textattack.attack_recipes import BERTAttackLi2020
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper

# 加载预训练的BERT模型和tokenizer
model_name = 'textattack/bert-base-uncased-imdb'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 创建模型包装器
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# 使用TextAttack提供的BERT-Attack攻击算法
attack_recipe = BERTAttackLi2020.build(model_wrapper)

# 示例文本
original_text = "This is a simple example to demonstrate adversarial sample generation."
true_label = 1  # 假设正确标签为1（积极）

# 创建TextAttack的攻击目标
attack_goal = textattack.goal_functions.ClassificationGoalFunction(model_wrapper)
attack_input = textattack.shared.AttackedText(original_text)
attack_target = textattack.shared.AttackTarget(attack_input, true_label)

# 运行攻击并生成对抗样本
attack_result = attack_recipe.attack(attack_target)

print("Original Text: ", original_text)
print("Adversarial Text: ", attack_result.perturbed_text())
