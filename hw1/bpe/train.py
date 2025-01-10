from tokenizer import Tokenizer
from transformers import GPT2Tokenizer

# Load GPT-2 tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('/home/pku0018/models/gpt2')

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

"""
Vocab size: 1024
First 10 tokens: [b'\n', b' ', b'%', b'&', b'(', b')', b'+', b',', b'-', b'.']
Last 10 tokens: [b'\xe6\x9c\x9f\xe9\x99\x90', b'\xe5\x8c\xbb\xe9\x99\xa2', b'\xe9\x9c\x80\xe8\xa6\x81', b'\xe5\x9d\x9a', b'\xe6\x95\xb4', b'\xe6\x9c\xaf', b'\xe4\xbf\xae\xe8\xaf\xbe', b'\xe5\xbf\x85\xe9\xa1\xbb', b'\xe6\xaf\x8f', b'\xe4\xbc\x9a\xe8\xae\xae']
Reconstruction successful: True
English text tokenization comparison:
BPE tokens: [38, 67, 60, 58, 60, 64, 52, 69, 56, 55, 1, 52, 68, 1, 69, 59, 56, 1, 32, 63, 66, 56, 67, 60, 52, 62, 1, 44, 64, 60, 71, 56, 67, 68, 60, 69, 74, 1, 65, 57, 489, 56, 61, 60, 64, 58, 1, 60, 64, 596, 19, 20, 19, 7, 489, 56, 61, 60, 64, 58, 1, 44, 64, 60, 71, 56, 67, 68, 60, 69, 74, 1, 72, 52, 68, 1, 26, 59, 60, 64, 52, 294, 101, 68, 1, 57, 60, 67, 68, 69, 1, 64, 52, 69, 60, 65, 64, 52, 62, 1, 54, 65, 63, 66, 67, 56, 59, 56, 64, 68, 60, 71, 56, 1, 70, 64, 60, 71, 56, 67, 68, 60, 69, 74, 1, 52, 64, 55, 1, 69, 59, 56, 1, 68, 70, 66, 67, 56, 63, 56, 1, 56, 55, 70, 54, 52, 69, 60, 65, 64, 1, 52, 70, 69, 59, 65, 67, 60, 69, 74, 1, 52, 69, 1, 69, 59, 56, 1, 69, 60, 63, 56, 463, 42, 60, 64, 54, 56, 1, 69, 59, 56, 1, 57, 65, 70, 64, 55, 60, 64, 58, 1, 65, 57, 1, 69, 59, 56, 489, 56, 65, 66, 62, 56, 294, 101, 68, 1, 41, 56, 66, 70, 53, 62, 60, 54, 1, 65, 57, 1, 26, 59, 60, 64, 52, 1, 60, 64, 596, 20, 15, 20, 7, 1, 60, 69, 1, 59, 52, 68, 1, 55, 56, 71, 56, 62, 65, 66, 56, 55, 1, 60, 64, 69, 65, 1, 52, 1, 54, 65, 63, 66, 67, 56, 59, 56, 64, 68, 60, 71, 56, 1, 70, 64, 60, 71, 56, 67, 68, 60, 69, 74, 1, 72, 60, 69, 59, 1, 57, 70, 64, 55, 52, 63, 56, 64, 69, 52, 62, 1, 56, 55, 70, 54, 52, 69, 60, 65, 64, 1, 52, 64, 55, 1, 67, 56, 68, 56, 52, 67, 54, 59, 1, 60, 64, 1, 53, 65, 69, 59, 1, 59, 70, 63, 52, 64, 60, 69, 60, 56, 68, 1, 52, 64, 55, 1, 68, 54, 60, 56, 64, 54, 56, 463, 43, 59, 56, 1, 67, 56, 57, 65, 67, 63, 1, 52, 64, 55, 1, 65, 66, 56, 64, 60, 64, 58, 8, 70, 66, 1, 65, 57, 1, 26, 59, 60, 64, 52, 1, 60, 64, 596, 20, 18, 19, 1, 59, 52, 68, 1, 70, 68, 59, 56, 67, 56, 55, 1, 60, 64, 1, 52, 1, 64, 56, 72, 1, 56, 67, 52, 1, 57, 65, 67, 1, 69, 59, 56, 1, 44, 64, 60, 71, 56, 67, 68, 60, 69, 74, 1, 70, 64, 68, 56, 56, 64, 1, 60, 64, 1, 59, 60, 68, 69, 65, 67, 74, 463, 24, 64, 55, 1, 60, 69, 68, 1, 63, 56, 67, 58, 56, 67, 1, 72, 60, 69, 59, 1, 25, 56, 60, 60, 64, 58, 1, 36, 56, 55, 60, 54, 52, 62, 1, 44, 64, 60, 71, 56, 67, 68, 60, 69, 74, 1, 60, 64, 581, 11, 11, 11, 1, 59, 52, 68, 1, 58, 56, 52, 67, 56, 55, 1, 60, 69, 68, 56, 62, 57, 1, 70, 66, 1, 57, 65, 67, 1, 52, 62, 62, 8, 67, 65, 70, 64, 55, 1, 52, 64, 55, 1, 71, 60, 53, 67, 52, 64, 69, 1, 58, 67, 65, 72, 69, 59, 1, 60, 64, 1, 68, 70, 54, 59, 1, 57, 60, 56, 62, 55, 68, 1, 52, 68, 1, 68, 54, 60, 56, 64, 54, 56, 7, 1, 56, 64, 58, 60, 64, 56, 56, 67, 60, 64, 58, 7, 1, 63, 56, 55, 60, 54, 60, 64, 56, 7, 1, 52, 58, 67, 60, 54, 70, 62, 69, 70, 67, 56, 7, 1, 59, 70, 63, 52, 64, 60, 69, 60, 56, 68, 1, 52, 64, 55, 1, 68, 65, 54, 60, 52, 62, 1, 68, 54, 60, 56, 64, 54, 56, 68, 463, 42, 70, 66, 66, 65, 67, 69, 56, 55, 1, 53, 74, 1, 69, 59, 56, 1, 451, 13, 12, 12, 489, 67, 65, 56, 54, 69, 455, 1, 52, 64, 55, 1, 69, 59, 56, 1, 451, 20, 19, 16, 489, 67, 65, 56, 54, 69, 455, 7, 1, 69, 59, 56, 1, 44, 64, 60, 71, 56, 67, 68, 60, 69, 74, 1, 59, 52, 68, 1, 63, 52, 55, 56, 1, 67, 56, 63, 52, 67, 61, 52, 53, 62, 56, 1, 52, 54, 59, 60, 56, 71, 56, 63, 56, 64, 69, 68, 7, 1, 68, 70, 54, 59, 1, 52, 68, 1, 65, 66, 69, 60, 63, 60, 60, 64, 58, 1, 55, 60, 68, 54, 60, 66, 62, 60, 64, 56, 68, 7, 1, 54, 70, 62, 69, 60, 71, 52, 69, 60, 64, 58, 1, 69, 52, 62, 56, 64, 69, 68, 7, 1, 67, 56, 54, 67, 70, 60, 69, 60, 64, 58, 1, 59, 60, 58, 59, 8, 54, 52, 62, 60, 53, 56, 67, 1, 69, 56, 52, 54, 59, 56, 67, 68, 7, 1, 52, 68, 1, 72, 56, 62, 62, 1, 52, 68, 1, 69, 56, 52, 54, 59, 60, 64, 58, 1, 52, 64, 55, 1, 68, 54, 60, 56, 64, 69, 60, 57, 60, 54, 1, 67, 56, 68, 56, 52, 67, 54, 59, 7, 1, 72, 59, 60, 54, 59, 1, 66, 52, 71, 56, 68, 1, 69, 59, 56, 1, 72, 52, 74, 1, 57, 65, 67, 1, 52, 1, 72, 65, 67, 62, 55, 8, 54, 62, 52, 68, 68, 1, 70, 64, 60, 71, 56, 67, 68, 60, 69, 74, 9]
GPT-2 tokens: [11610, 3898, 355, 262, 11773, 2059, 286, 350, 18754, 287, 46244, 11, 350, 18754, 2059, 373, 2807, 447, 247, 82, 717, 2260, 9815, 6403, 290, 262, 17700, 3707, 4934, 379, 262, 640, 13, 4619, 262, 16636, 286, 262, 4380, 447, 247, 82, 2066, 286, 2807, 287, 24977, 11, 340, 468, 4166, 656, 257, 9815, 6403, 351, 7531, 3707, 290, 2267, 287, 1111, 47824, 290, 3783, 13, 383, 4975, 290, 4756, 12, 929, 286, 2807, 287, 15524, 468, 47098, 287, 257, 649, 6980, 329, 262, 2059, 29587, 287, 2106, 13, 843, 663, 24589, 351, 11618, 8366, 2059, 287, 4751, 468, 31394, 2346, 510, 329, 477, 12, 744, 290, 21266, 3349, 287, 884, 7032, 355, 3783, 11, 8705, 11, 9007, 11, 14510, 11, 47824, 290, 1919, 19838, 13, 36848, 416, 262, 564, 250, 21895, 4935, 447, 251, 290, 262, 564, 250, 42250, 4935, 447, 251, 11, 262, 2059, 468, 925, 11004, 16970, 11, 884, 355, 45780, 29861, 11, 45414, 18054, 11, 16517, 1029, 12, 43288, 7799, 11, 355, 880, 355, 7743, 290, 5654, 2267, 11, 543, 279, 3080, 262, 835, 329, 257, 995, 12, 4871, 6403, 13]
Chinese text tokenization comparison:
BPE tokens: [351, 405, 417, 607, 516, 238, 350, 936, 148, 87, 120, 619, 876, 447, 806, 191, 316, 164, 919, 587, 223, 806, 246, 1002, 1005, 1019, 390, 848, 328, 652, 751, 673, 164, 847, 169, 351, 405, 246, 147, 93, 100, 301, 965, 345, 223, 689, 606, 340, 434, 602, 180, 124, 800, 549, 394, 587, 378, 431, 229, 273, 391, 169, 996, 417, 457, 100, 931, 154, 112, 187, 354, 305, 549, 588, 164, 864, 841, 305, 158, 314, 282, 230, 187, 417, 466, 443, 394, 164, 864, 169, 841, 211, 417, 330, 305, 720, 165, 114, 845, 164, 156, 1019, 273, 645, 158, 452, 103, 305, 689, 342, 439, 304, 169]
GPT-2 tokens: [39355, 248, 18803, 27764, 99, 19526, 235, 164, 106, 118, 23877, 229, 41753, 242, 37605, 241, 26193, 101, 23626, 236, 43291, 38519, 17739, 115, 17312, 231, 45379, 105, 44165, 233, 20015, 236, 12859, 233, 163, 100, 239, 27764, 99, 163, 254, 242, 163, 102, 114, 32432, 98, 43291, 21410, 47797, 121, 27950, 249, 171, 120, 234, 33176, 114, 28839, 101, 163, 100, 239, 27764, 99, 22755, 244, 10310, 241, 29785, 101, 162, 232, 222, 17312, 107, 41468, 161, 223, 248, 49035, 118, 26344, 249, 34460, 254, 45250, 100, 21410, 22755, 238, 162, 252, 250, 16764, 39355, 248, 18803, 27764, 99, 19526, 235, 164, 106, 118, 23877, 229, 22755, 244, 162, 239, 246, 17358, 223, 171, 120, 234, 41753, 242, 37605, 241, 28839, 101, 163, 18433, 164, 122, 102, 30298, 235, 49011, 10310, 103, 17312, 230, 39355, 108, 34460, 223, 17312, 231, 17739, 111, 39355, 243, 19526, 235, 171, 120, 234, 33176, 114, 163, 119, 237, 28938, 234, 26193, 234, 46237, 226, 164, 106, 106, 16764, 27764, 99, 19526, 235, 162, 236, 230, 12859, 230, 39355, 243, 19526, 235, 41753, 242, 37605, 241, 164, 223, 246, 46237, 115, 10310, 97, 19526, 235, 10310, 236, 164, 106, 118, 23877, 229, 17312, 231, 17739, 111, 27764, 99, 163, 100, 239, 21410, 10310, 241, 22522, 114, 46237, 226, 165, 11805, 164, 106, 118, 23877, 229, 171, 120, 234, 17739, 114, 40792, 31660, 19526, 235, 41753, 242, 37605, 241, 42468, 13783, 244, 39355, 243, 19526, 235, 21410, 10310, 241, 22522, 114, 16764, 46237, 226, 165, 11805, 21689, 41753, 242, 37605, 241, 43380, 117, 164, 106, 118, 23877, 229, 37863, 247, 46237, 99, 163, 119, 228, 21410, 27764, 99, 17312, 107, 46237, 226, 46237, 255, 171, 120, 234, 160, 122, 249, 164, 106, 118, 23877, 229, 163, 18433, 164, 122, 102, 34650, 242, 37772, 246, 27670, 248, 20998, 224, 32003, 225, 16764]
English text tokenization comparison:
BPE tokens length: 938
GPT-2 tokens length: 185

Chinese text tokenization comparison:
BPE tokens length: 113
GPT-2 tokens length: 306
"""