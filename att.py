import numpy as np

batch_size=100

def score_data():
    data = np.load("unsw.npy")
    score = data[0:5, :]
    label = data[5, :]

    return (score, label)


class att_model(nn.Module):
    def __init__(self):
        super(att_model, self).__init__()
        
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)


    def attention_net(self, x):       #x:[batch, seq_len, hidden_dim*2]

        u = torch.tanh(torch.matmul(x, self.w_omega))         #[batch, seq_len, hidden_dim*2]
        att = torch.matmul(u, self.u_omega)                   #[batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)

        scored_x = x * att_score                              #[batch, seq_len, hidden_dim*2]

        context = torch.sum(scored_x, dim=1)                  #[batch, hidden_dim*2]
        return context


    def forward(self, x):
        embedding = self.dropout(self.embedding(x))       #[seq_len, batch, embedding_dim]

        # output: [seq_len, batch, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        output = output.permute(1, 0, 2)                  #[batch, seq_len, hidden_dim*2]

        attn_output = self.attention_net(output)
        logit = self.fc(attn_output)
        return logit

    def forward(self, x):


def train(epoch, train_loader, allow_zz=True):

    model.train()
    loss_encoder_s, loss_decoder_s, loss_xz_s, loss_xx_s, loss_zz_s = 0.0, 0.0, 0.0, 0.0, 0.0
    for idx, (score, label) in enumerate(tqdm(train_loader)):
        score


x, y = score_data()
loader = FastTensorDataLoader(x, y, batch_size=batch_size, shuffle=True)

model = att_model()
model.cuda()

for epoch in range(50):
    train(epoch, loader)




