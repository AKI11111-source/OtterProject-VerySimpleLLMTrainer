# 第二段代码：SFT微调训练 (train_sft.py)
import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import Dataset, DataLoader
import os

class CharVocab:
    def __init__(self):
        self.char2idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.idx2char = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}
        self.vocab_size = 4
        
    def build_vocab(self, texts):
        chars = set()
        for text in texts:
            chars.update(text)
        
        for char in sorted(chars):
            self.char2idx[char] = self.vocab_size
            self.idx2char[self.vocab_size] = char
            self.vocab_size += 1
            
    def encode(self, text):
        return [self.char2idx.get(c, self.char2idx['<unk>']) for c in text]

# 使用与预训练相同的模型架构
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3, max_seq_len=256):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        x_emb = self.embedding(x)
        pos_emb = self.pos_embedding(positions)
        x = x_emb + pos_emb
        
        # 因果掩码
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        x = self.transformer(x, memory=x, tgt_mask=mask)
        return self.fc(x)

# SFT数据集类
class SFTDataset(Dataset):
    def __init__(self, file_path, vocab, max_length=256):
        self.vocab = vocab
        self.max_length = max_length
        self.data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                conversations = item['conversations']
                
                # 将对话拼接成一个序列
                full_text = ""
                for turn in conversations:
                    role = "用户: " if turn['role'] == 'user' else "助手: "
                    full_text += role + turn['content'] + "\n"
                
                encoded = vocab.encode(full_text)
                if len(encoded) > max_length - 2:  # 为BOS和EOS留空间
                    encoded = encoded[:max_length-2]
                encoded = [vocab.char2idx['<bos>']] + encoded + [vocab.char2idx['<eos>']]
                self.data.append(encoded)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        if len(item) < self.max_length:
            item = item + [self.vocab.char2idx['<pad>']] * (self.max_length - len(item))
        return torch.tensor(item[:-1]), torch.tensor(item[1:])

# SFT训练函数
def train_sft():
    # 加载预训练模型和词汇表
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('pretrain_model.pt',map_location=device, weights_only=False)
    vocab = checkpoint['vocab']
    
    # 初始化模型
    model = SimpleTransformer(vocab.vocab_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建SFT数据集和数据加载器
    dataset = SFTDataset('sft.jsonl', vocab)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 训练配置
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.char2idx['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 训练循环
    model.train()
    for epoch in range(1):  # 微调x个epoch
        total_loss = 0
        for i, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')
        
        print(f'Epoch {epoch}, Average Loss: {total_loss / len(dataloader)}')
    
    # 保存SFT模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab
    }, 'sft_model.pt')
    
    print("SFT训练完成，模型已保存为 sft_model.pt")

if __name__ == '__main__':
    train_sft()