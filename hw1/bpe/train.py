from tokenizer import Tokenizer
from transformers import GPT2Tokenizer

# Load GPT-2 tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('/data/align-anything/hantao/models/gpt2')

# Your BPE tokenizer setup
with open("manual.txt", "rb") as f:
    data_bytes = f.read()

tokenizer = Tokenizer()
tokenizer.train(data_bytes, vocab_size=1024)

with open("manual.txt", "r", encoding="utf-8") as f:
    text = f.read()
# Test encode/decode
ids = tokenizer.encode(text)
decoded_text = tokenizer.decode(ids)
print("Reconstruction successful:", text == decoded_text)

# Test sentences
english_text = "Originated as the Imperial University of Peking in 1898, Peking University was China’s first national comprehensive university and the supreme education authority at the time. Since the founding of the People’s Republic of China in 1949, it has developed into a comprehensive university with fundamental education and research in both humanities and science. The reform and opening-up of China in 1978 has ushered in a new era for the University unseen in history. And its merger with Beijing Medical University in 2000 has geared itself up for all-round and vibrant growth in such fields as science, engineering, medicine, agriculture, humanities and social sciences. Supported by the “211 Project” and the “985 Project”, the University has made remarkable achievements, such as optimizing disciplines, cultivating talents, recruiting high-caliber teachers, as well as teaching and scientific research, which paves the way for a world-class university."
chinese_text = "博士学位论文应当表明作者具有独立从事科学研究工作的能力，并在科学或专门技术上做出创造性的成果。博士学位论文或摘要，应当在答辩前三个月印送有关单位，并经同行评议。学位授予单位应当聘请两位与论文有关学科的专家评阅论文，其中一位应当是外单位的专家。评阅人应当对论文写详细的学术评语，供论文答辩委员会参考。"

# Compare tokenization
bpe_eng_tokens = tokenizer.encode(english_text)
gpt2_eng_tokens = gpt2_tokenizer.encode(english_text)

bpe_cn_tokens = tokenizer.encode(chinese_text)
gpt2_cn_tokens = gpt2_tokenizer.encode(chinese_text)

print(f"English text tokenization comparison:")
print(f"BPE tokens: {bpe_eng_tokens}")
print(f"GPT-2 tokens: {gpt2_eng_tokens}")

print(f"Chinese text tokenization comparison:")
print(f"BPE tokens: {bpe_cn_tokens}")
print(f"GPT-2 tokens: {gpt2_cn_tokens}")

print("English text tokenization comparison:")
print(f"BPE tokens length: {len(bpe_eng_tokens)}")
print(f"GPT-2 tokens length: {len(gpt2_eng_tokens)}")

print("\nChinese text tokenization comparison:")
print(f"BPE tokens length: {len(bpe_cn_tokens)}")
print(f"GPT-2 tokens length: {len(gpt2_cn_tokens)}")
