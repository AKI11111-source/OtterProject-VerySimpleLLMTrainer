# 第三段代码：对话推理 (chat.py)
import torch
import torch.nn as nn
import json

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

# 使用相同的模型架构
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

# 生成函数
def generate_response(model, vocab, prompt, max_length=100, temperature=0.8):
    device = next(model.parameters()).device
    
    # 编码输入
    input_text = "用户: " + prompt + "\n助手: "
    encoded = [vocab.char2idx['<bos>']] + vocab.encode(input_text)
    input_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    
    # 生成响应
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_tensor)
            next_token_logits = output[0, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
            
            if next_token.item() == vocab.char2idx['<eos>']:
                break
                
            encoded.append(next_token.item())
            input_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    
    # 解码响应
    response_ids = encoded[len(vocab.encode("用户: " + prompt + "\n助手: ")) + 1:]  # +1 for BOS
    response = ''.join([vocab.idx2char.get(idx, '') for idx in response_ids])
    
    # 移除可能的多余内容
    if '\n' in response:
        response = response.split('\n')[0]
    
    return response

# 主对话循环
def chat():
    # 加载SFT模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('sft_model.pt',map_location=device,weights_only=False)
    vocab = checkpoint['vocab']
    
    model = SimpleTransformer(vocab.vocab_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("对话模型已加载，开始对话吧！(输入'退出'结束对话)")
    
    while True:
        user_input = input("用户: ")
        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("助手: 再见！")
            break
        
        response = generate_response(model, vocab, user_input)
        print(f"助手: {response}")

if __name__ == '__main__':
    chat()