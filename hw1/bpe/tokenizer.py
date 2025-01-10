import collections
import tqdm
import json

class Tokenizer:
    def __init__(self):
        # token2id/id2token 用于存储最终的 byte-level token
        # 注意，这里的 "token" 并不是单个字节，而可能是多个字节拼接
        # 故，我们将 token 表示为 bytes 类型
        self.token2id = {}
        self.id2token = []
        
        # 记录合并规则： (tokenA_id, tokenB_id) -> new_token_id
        self.bpe_ranks = {}

        # 目标词表大小
        self.vocab_size = 0

    def train(self, data_bytes, vocab_size):
        """
        训练 byte-level BPE Tokenizer
        参数:
            data_bytes (bytes): 训练文本对应的原始字节序列
            vocab_size (int): 目标词表大小
        """
        self.vocab_size = vocab_size

        # 1. 找到数据中实际出现过的所有字节，初始化 token2id, id2token
        counter = collections.Counter(data_bytes)
        unique_bytes = sorted(counter.keys())  # 按字节值从小到大排序
        
        # 初始化最初的 token2id, id2token
        # 每个单字节都是一个 token（以 bytes([b]) 表示）
        for i, b in enumerate(unique_bytes):
            token = bytes([b])  
            self.token2id[token] = i
            self.id2token.append(token)
        
        current_vocab_size = len(self.id2token)

        # 2. 将文本由字节转换为 token_id 列表，例如 b'\xe5\x8c\x97' -> [id(\xe5), id(\x8c), id(\x97), ...]
        token_id_data = [self.token2id[bytes([b])] for b in data_bytes]

        # 定义一个函数: 用于统计所有相邻 token 对出现的频率
        def get_pair_stats(token_id_list):
            pair_stats = collections.Counter()
            for i in range(len(token_id_list) - 1):
                pair = (token_id_list[i], token_id_list[i + 1])
                pair_stats[pair] += 1
            return pair_stats

        # use tqdm to show the progress
        pbar = tqdm.tqdm(total=vocab_size, desc="Training BPE")
        pbar.update(current_vocab_size)
        # 3. 进入合并循环
        while current_vocab_size < vocab_size:
            pbar.update(1)
            pair_stats = get_pair_stats(token_id_data)
            if not pair_stats:
                # 没有可合并对
                break

            best_pair, best_count = pair_stats.most_common(1)[0]
            if best_count < 1:
                # 无法继续合并
                break

            # 构造新 token (字节拼接)
            tokenA_id, tokenB_id = best_pair
            tokenA = self.id2token[tokenA_id]
            tokenB = self.id2token[tokenB_id]
            new_token = tokenA + tokenB  # bytes拼接

            # 如果已经存在则不重复添加（极少情况下可能出现重复）
            if new_token in self.token2id:
                # 已存在则跳过
                # (理论上这一步通常不会发生，但实现里最好做个保护)
                break

            new_token_id = current_vocab_size
            self.token2id[new_token] = new_token_id
            self.id2token.append(new_token)
            self.bpe_ranks[best_pair] = new_token_id
            current_vocab_size += 1

            # 4. 在 token_id_data 中，把所有 pair 替换成 new_token
            new_token_id_data = []
            i = 0
            while i < len(token_id_data):
                if i < len(token_id_data) - 1:
                    current_pair = (token_id_data[i], token_id_data[i+1])
                    if current_pair == best_pair:
                        # 合并
                        new_token_id_data.append(new_token_id)
                        i += 2
                        continue
                new_token_id_data.append(token_id_data[i])
                i += 1

            token_id_data = new_token_id_data
        pbar.close()

        print(f"Vocab size: {len(self.id2token)}")
        # print the first 10 and last 10 tokens
        print(f"First 10 tokens: {self.id2token[:10]}")
        print(f"Last 10 tokens: {self.id2token[-10:]}")

    def encode(self, text):
        """
        把字符串编码成 token_id 列表 (byte-level BPE)
        1) 先将 text 转成 UTF-8 bytes
        2) 把每个单字节转成 token_id（若不存在就视作 unk，这里可以自行处理）
        3) 根据训练时学到的 bpe_ranks 不断做 pair 合并
        """
        text_bytes = text.encode('utf-8', errors='replace')
        
        # 先把每个字节转换为 token_id
        token_ids = []
        for b in text_bytes:
            bt = bytes([b])
            if bt in self.token2id:
                token_ids.append(self.token2id[bt])
            else:
                # 如果出现了训练集中从未见过的字节，可视作 UNK
                # 这里演示中直接跳过
                pass

        # 然后根据 bpe_ranks 做合并 (贪心)
        # 跟训练时做法一样：循环直到无法再合并
        changed = True
        while changed:
            changed = False
            new_token_ids = []
            i = 0
            while i < len(token_ids):
                if i < len(token_ids) - 1:
                    pair = (token_ids[i], token_ids[i+1])
                    if pair in self.bpe_ranks:
                        new_token_ids.append(self.bpe_ranks[pair])
                        i += 2
                        changed = True
                        continue
                new_token_ids.append(token_ids[i])
                i += 1

            token_ids = new_token_ids
        return token_ids

    def decode(self, token_ids):
        """
        根据 token_id 列表还原出原始字节，再用 utf-8 解码得到字符串
        """
        output_bytes = bytearray()
        for tid in token_ids:
            if 0 <= tid < len(self.id2token):
                output_bytes += self.id2token[tid]
            else:
                # 超出词表，视作 UNK 或忽略
                pass
        # 再将字节数组解码为字符串 (utf-8)
        return output_bytes.decode('utf-8', errors='replace')