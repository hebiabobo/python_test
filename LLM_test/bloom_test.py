# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="bigscience/bloom-560m")

# 加载BLOOM模型和分词器
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

# 编码输入文本
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
print("inputs:", inputs)

# 生成文本
output_sequences = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1, penalty_alpha=0.6,
                                  repetition_penalty=5.0, do_sample=True, top_k=50,
                                  top_p=0.8)

# 解码输出文本
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(generated_text)
