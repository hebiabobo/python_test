#!/usr/bin/env python
# coding: utf-8

# Based on code from: http://nlp.seas.harvard.edu/2018/04/03/attention.html

# # Prelims

# In[ ]:


# from google.colab import drive
# from tqdm.autonotebook import get_ipython
#
# drive.mount('/content/drive')

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

seaborn.set_context(context="talk")
# get_ipython().run_line_magic('matplotlib', 'auto')

# In[ ]:


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
if cuda:
    print("Using CUDA from GPU")


# get_ipython().system('nvidia-smi')


# # Model Architecture

# In[ ]:


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# In[ ]:


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# ## Encoder and Decoder Stacks

# In[ ]:


def clones(module, N):
    '''
    创建 N 个相同的神经网络层实例。具体来说，它接收一个神经网络模块 module 和一个整数 N，返回一个包含 N 个相同模块的 nn.ModuleList。

    在实现像 Transformer 这样的模型时，需要堆叠多个相同的子层。例如，在 Transformer 编码器中，会堆叠多层自注意力层和前馈神经网络层。
    通过 clones 函数，可以轻松地创建这些重复的层，而不需要手动实例化每一层。
    '''
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# In[ ]:


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# In[ ]:


class LayerNorm(nn.Module):  # 这个 LayerNorm 类实现了一个自定义的层归一化（Layer Normalization）模块。层归一化是一种正则化技术，可以在训练神经网络时帮助稳定和加速训练过程。
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):  # features：输入张量的最后一维的大小（即特征的数量）。eps：一个很小的数值，用于防止除以零的情况，默认为 1e-6。
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # 一个可学习的参数，初始化为全 1 向量，形状为 (features,)。
        self.b_2 = nn.Parameter(torch.zeros(features))  # 另一个可学习的参数，初始化为全 0 向量，形状为 (features,)。
        self.eps = eps

    def forward(self, x):
        '''
        keepdim=True 保持输出张量的维度与输入一致，这样在后续计算时可以进行广播操作。

        广播操作（Broadcasting）是一种便捷的方式，用于在不同形状的数组之间执行算术运算。广播机制会自动扩展较小的数组，使其与较大的数组形状匹配，
        从而使得运算可以在相同形状的数组之间进行。
        '''
        mean = x.mean(-1, keepdim=True)  # 计算输入 x 在最后一个维度上的均值。
        std = x.std(-1, keepdim=True)  # 计算输入 x 在最后一个维度上的标准差。
        '''
        使用可学习参数 self.a_2 和 self.b_2 对标准化结果进行线性变换。
        
        对输入 x 进行标准化，即减去均值 mean，再除以标准差 std 加上一个很小的值 eps。
        归一化后的结果乘以可学习参数 self.a_2，再加上另一个可学习参数 self.b_2。
        '''
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# In[ ]:


class SublayerConnection(nn.Module):  # 实现了一个子层连接模块，其中包含残差连接和层归一化（Layer Normalization）。
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):  # size：输入张量的最后一维的大小（即特征的数量）。
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)  # dropout 层的丢弃概率。

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."

        '''
        残差连接（Residual Connection）能够缓解梯度消失问题的原因在于它改变了网络中梯度的传播路径，从而使得梯度能够更有效地传递。
        具体来说，残差连接通过在每一层添加一个恒等映射（identity mapping），使得信息能够更直接地从输入层传递到输出层。这有助于保持梯度的大小，
        避免在反向传播过程中梯度过小或过大。
        
        在残差网络中，每一层的输出不仅依赖于该层的非线性变换（𝐹(𝑥)），还包括直接通过恒等映射传递的输入（𝑥）。这种直接路径使得梯度可以绕过非线性变换层，
        直接传播到前面的层，从而减小了梯度消失的风险。
        
        假设我们有一个没有残差连接的深度网络和一个有残差连接的深度网络。对于没有残差连接的深度网络，假如某一层的梯度很小，
        那么这一层之前的所有梯度都会受到影响，梯度会逐层减小，最终导致梯度消失。而在有残差连接的深度网络中，即使某一层的梯度很小，
        由于有恒等映射的存在，梯度能够通过恒等映射直接传递回前面的层，保持梯度的大小，从而缓解梯度消失问题。
        '''
        return x + self.dropout(sublayer(self.norm(x)))  # 残差连接


# In[ ]:


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn  # 自注意力机制模块
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout),
                               2)  # self.sublayer 是两个 SublayerConnection 实例的克隆，分别用于自注意力机制和前馈神经网络。
        self.size = size  # 保存了隐藏层的大小。

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        '''
        第一个子层 (self.sublayer[0]) 对输入 x 应用自注意力机制，并且使用残差连接和层归一化。
        self_attn(x, x, x, mask)：这是一个自注意力机制，三个 x 分别代表 query、key 和 value，并且使用了掩码 mask。
        lambda x: self.self_attn(x, x, x, mask)：这是一个匿名函数，用于包装自注意力机制，以适应 SublayerConnection 的接口。
        '''
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        return self.sublayer[1](x, self.feed_forward)  # 第二个子层 (self.sublayer[1]) 对第一个子层的输出应用前馈神经网络，并且使用残差连接和层归一化。


# In[ ]:


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# In[ ]:


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# In[ ]:


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# ### Attention

# In[ ]:


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention 缩放点积注意力'"
    '''
    # query: 查询张量，形状为 [batch_size, seq_len_q, d_k]。key: 键张量，形状为 [batch_size, seq_len_k, d_k]。
    value: 值张量，形状为 [batch_size, seq_len_v, d_v]，其中 seq_len_k 和 seq_len_v 可以不同。
    mask: 可选的注意力掩码，形状为 [batch_size, seq_len_q, seq_len_k]，掩盖了不需要关注的位置。
    '''
    d_k = query.size(-1)  # 获取查询向量的最后一个维度大小，通常是注意力的维度。
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
        d_k)  # 计算缩放后的点积注意力。将查询和键的转置相乘，然后除以缩放系数 sqrt(d_k)。这一步计算了每对 query 和 key 之间的相关性。
    if mask is not None:  # : 如果存在掩码，则将掩码为 0 的位置的分数设置为一个非常大的负数（接近负无穷），这样在 softmax 计算时这些位置的注意力权重会接近于 0。
        scores = scores.masked_fill(mask == 0, -1e9)
    '''
    p_attn = F.softmax(scores, dim=-1): 对分数进行 softmax 归一化，得到注意力权重。
    dim=-1 表示在最后一个维度上进行 softmax 操作，通常是 seq_len_k 的维度。
    '''
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn  # 将注意力权重 p_attn 与值张量 value 相乘，得到加权后的值。同时返回注意力权重 p_attn，以便后续可视化或分析。


# In[ ]:


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):  # h: 注意力头的数量。d_model: 模型的维度。
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # 确保 d_model 可以被 h 整除。
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # 每个头的维度。
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # 创建四个线性层，用于将输入投影到多个子空间中。
        self.attn = None  # 用于存储注意力权重。
        self.dropout = nn.Dropout(p=dropout)  # dropout 层。

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # 如果提供了掩码，将其扩展一个维度，以便应用于所有头。
        nbatches = query.size(0)  # 获取批次大小 nbatches:

        # 1) Do all the linear projections in batch from d_model => h x d_k
        '''
        for l, x in zip(self.linears, (query, key, value))：迭代 zip 生成的元组，每次迭代得到一个线性层 l 和对应的输入 x（查询、键或值）
        
        l(x)：将输入 x 通过线性层 l 进行线性变换。
        
        view(nbatches, -1, self.h, self.d_k)：将线性变换后的结果重整形，nbatches 是批次大小，-1 表示自动计算的维度，
        self.h 是头的数量，self.d_k 是每个头的维度。
        
        .transpose(1, 2)：将维度进行转置，使得结果的形状从 [nbatches, seq_len, heads, head_dim] 变为 [nbatches, heads, seq_len, head_dim]。
        '''
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,  # 计算注意力得分和输出。
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        '''
        从多头注意力的输出中，重新组合成原始输入的形状并应用最后一个线性层，得到最终输出。下面是详细解释:
        
        x.transpose(1, 2):
        输入 x 的形状是 [batch_size, heads, seq_len, head_dim]。
        通过 x.transpose(1, 2) 将 heads 和 seq_len 维度互换，结果的形状为 [batch_size, seq_len, heads, head_dim]。
        
        .contiguous():
        contiguous 是一个 PyTorch 方法，用于返回一个内存连续的张量副本。这是因为 transpose 操作不会改变底层数据的内存布局，
        使用 contiguous 确保后续操作能够顺利进行。
        
        .view(nbatches, -1, self.h * self.d_k):
        view 方法重新调整张量的形状。
        [batch_size, seq_len, heads, head_dim] 变为 [batch_size, seq_len, heads * head_dim]。
        heads * head_dim 就是 self.h * self.d_k，其中 self.h 是注意力头的数量，self.d_k 是每个头的维度。
        '''
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        '''
        self.linears[-1](x):

        self.linears 是一个包含 4 个线性层的列表。
        self.linears[-1] 获取列表中的最后一个线性层。
        x 通过最后一个线性层进行线性变换，得到最终的输出。
        '''
        return self.linears[-1](x)


# ## Position-wise Feed-Forward Networks

# In[ ]:


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation.该类实现了 Transformer 中的前馈神经网络 (Feed Forward Network, FFN)，这个 FFN 在每个位置上独立应用。"

    def __init__(self, d_model, d_ff, dropout=0.1):  # d_model：模型的维度。d_ff：前馈神经网络的隐藏层维度。
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # self.w_1：从 d_model 映射到 d_ff 的线性层。
        self.w_2 = nn.Linear(d_ff, d_model)  # self.w_2：从 d_ff 映射回 d_model 的线性层。
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ## Embeddings and Softmax

# In[ ]:


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    '''
    在 Transformer 模型中，缩放词向量有助于稳定训练过程，特别是在使用自注意力机制时。具体原因如下:

    词嵌入矩阵中的初始值通常较小，如果不进行缩放，可能会导致模型的梯度更新较小，从而使训练过程变慢。
    通过乘以 math.sqrt(self.d_model)，可以确保输入到后续层的词向量具有合适的幅度，从而提高模型的数值稳定性。
    '''

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# ## Positional Encoding
#
#

# In[ ]:


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # 创建一个形状为 (max_len, d_model) 的零张量 pe，用于存储位置编码。
        '''
        创建一个形状为 (max_len, 1) 的张量 position，表示位置索引。
        
        torch.arange 函数生成一个从 0 开始到 max_len-1 结束的一维张量（即一个向量），数据类型默认为浮点数。
        unsqueeze(1):
        unsqueeze 方法用于在指定的维度上插入一个维度。在这里，参数 1 表示在第一维度（即索引为0的维度）前插入一个维度，
        将一维张量（形状为 (max_len,)）变为二维张量（形状为 (max_len, 1)）。
        '''
        position = torch.arange(0., max_len).unsqueeze(1)
        '''
        这行代码的作用是计算位置编码中的除数项 div_term。是一个固定值，用于位置编码的计算。
        
        torch.arange(0., d_model, 2):
        torch.arange 函数生成一个从 0 开始，以步长 2 递增的一维张量（即一个向量）。例如，如果 d_model 是 512，则这个张量包含了从 0 到 510 的偶数。
        
        -(math.log(10000.0) / d_model):
        math.log(10000.0) 计算以自然对数为底的 10000 的对数值。
        然后这个值被除以 d_model，并取负号。这一步计算得到了一个标量，用于在后续计算中缩放位置编码的频率。
        
        torch.exp 函数计算输入张量的每个元素的指数函数值。在这里，输入是一个张量，其元素是前一步计算得到的标量乘以从 0 到 d_model-1 的偶数。
        
        为了更好地理解这段代码，我们可以通过一个简单的例子来演示：

        d_model = 512
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        print(div_term)
        输出结果大致如下（具体数值可能有所不同）:

        tensor([1.0000e+00, 7.0711e-01, 5.0000e-01, 3.5355e-01, ..., 1.9659e-05])
        '''
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        '''
        position * div_term 执行逐元素乘法，生成一个形状为 (max_len, d_model / 2) 的张量，其中每一列都是位置索引乘以对应的 div_term。
        
        在值上position * div_term 相当于 max_len * div_term
        
        因为 pe 的第一个维度是 max_len，所以 pe[:, 0::2] 选择了 pe 中所有行的偶数列，将其填充为正弦函数的值。
              
        输出结果大致如下（具体数值可能有所不同）:
        tensor([[ 0.0000e+00,  1.0000e+00,  0.0000e+00, ...,  1.0000e+00],
        [ 8.4147e-01,  9.1545e-01,  8.6623e-01, ...,  9.9998e-01],
        [ 9.0929e-01,  4.1615e-01,  9.5106e-01, ...,  9.9992e-01],
        ...,
        [ 1.7453e-01, -9.8481e-01,  1.7097e-01, ...,  9.9970e-01],
        [ 9.6940e-01, -2.5202e-01,  9.6975e-01, ...,  9.9962e-01],
        [ 3.4202e-01, -9.3969e-01,  3.3623e-01, ...,  9.9951e-01]])
        
        即:
        tensor([[ sin(max_len * div_term),  cos(max_len * div_term),  sin, ...,  cos],
        [ sin(max_len * div_term),  cos(max_len * div_term),  sin, ...,  cos],
        [ sin(max_len * div_term),  cos(max_len * div_term),  sin, ...,  cos],
        ...,
        [ sin(max_len * div_term),  cos(max_len * div_term),  sin, ...,  cos],
        [ sin(max_len * div_term),  cos(max_len * div_term),  sin, ...,  cos],
        [ sin(max_len * div_term),  cos(max_len * div_term),  sin, ...,  cos]])
        '''
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)  # 同样的逻辑，但是选择了 pe 中所有行的奇数列，将其填充为余弦函数的值。
        '''
        unsqueeze(0) 方法在索引 0 的位置上添加一个维度，将原来的形状 (max_len, d_model) 扩展为 (1, max_len, d_model)。
        新添加的维度是用于表示批次大小的维度，虽然当前只有一个样本，但这种表示方式适用于处理多个样本的情况。
        '''
        pe = pe.unsqueeze(0)
        '''        
        self.register_buffer('pe', pe) 是将一个张量 pe 注册为模型的缓冲区（buffer）。在 PyTorch 中，模型的状态可以分为参数（parameters）和缓冲区（buffers）两种类型。

        作用和解释:
        
        注册为缓冲区:
        当调用 register_buffer 方法时，PyTorch 会将 pe 张量注册为模型的一部分，但它不是模型的参数。
        注册为缓冲区意味着 pe 不会被优化器更新，也不会出现在 model.parameters() 的输出中。
        这对于模型的固定部分非常有用，例如位置编码、Batch Normalization 的 running_mean 和 running_var 等。
        
        为什么使用 register_buffer:
        在深度学习模型中，有些张量是与模型结构紧密相关的，但它们不是模型的可学习参数。这些张量通常是模型在训练过程中需要使用的固定数据，
        例如位置编码或者在推理过程中需要保持不变的统计量。
        将这些张量注册为缓冲区可以确保它们被保存和加载到模型的状态字典中，同时在推理时不会被修改。
        '''
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


# ## Full Model

# In[ ]:


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    model = model.cuda()
    return model


# # Training
#

# ## Batches and Masking

# In[ ]:


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        tgt_mask = tgt_mask.cuda()
        return tgt_mask


# ## Training Loop

# In[ ]:


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        batch.src = batch.src.to(device)
        batch.trg = batch.trg.to(device)
        batch.src_mask = batch.src_mask.to(device)
        batch.trg_mask = batch.trg_mask.to(device)
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens).cuda()
        total_loss += loss
        total_loss = total_loss.cuda()
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


# ## Training Data and Batching

# In[ ]:


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    maximum = max(src_elements, tgt_elements)
    return maximum


# ## Optimizer

# In[ ]:


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# ## Regularization
#
# ### Label Smoothing
#

# In[ ]:


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.criterion = self.criterion.to(device)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        target.data = target.data.to(device)
        true_dist = x.data.clone()
        true_dist = true_dist.to(device)
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        mask = mask.to(device)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        self.true_dist = self.true_dist.to(device)
        final_criterion = self.criterion(x, Variable(true_dist, requires_grad=False)).to(device)
        return final_criterion


# ## Loss Computation

# In[ ]:


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x).cuda()
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss = loss.cuda()
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        final_loss = (loss.data * norm).cuda()
        return final_loss


# ## Greedy Decoding

# In[ ]:


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src_mask = src_mask.to(device)
    model = model.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


# # Data Preparation

# In[ ]:


# For data loading.
import os
from torchtext import data, datasets

# from torchtext.data import Field, BucketIterator, TabularDataset

# main_path = "/content/drive/My Drive/"
main_path = r"D:\PycharmProjects\SMERTI-master"

corpus_name = "News_Dataset"
corpus = os.path.join(main_path, corpus_name)
save_dir = os.path.join(corpus, r"output\transformer")
data_path = corpus
# os.chdir(data_path)

masked_train_path = os.path.join(data_path, "masked_train_headlines.txt")
unmasked_train_path = os.path.join(data_path, "train_headlines.txt")

masked_val_path = os.path.join(data_path, "masked_val_headlines.txt")
unmasked_val_path = os.path.join(data_path, "val_headlines.txt")

masked_test_path = os.path.join(data_path, "masked_test_headlines.txt")
unmasked_test_path = os.path.join(data_path, "test_headlines.txt")

# In[ ]:


from string import punctuation
import re

masked_train = open(masked_train_path, encoding='utf-8').read().split('\n')
unmasked_train = open(unmasked_train_path, encoding='utf-8').read().split('\n')

masked_val = open(masked_val_path, encoding='utf-8').read().split('\n')
unmasked_val = open(unmasked_val_path, encoding='utf-8').read().split('\n')

masked_test = open(masked_test_path, encoding='utf-8').read().split('\n')
unmasked_test = open(unmasked_test_path, encoding='utf-8').read().split('\n')


def process_text(s):
    s = s.lower().strip()
    s = re.sub('\!+', '!', s)
    s = re.sub('\,+', ',', s)
    s = re.sub('\?+', '?', s)
    s = re.sub('\.+', '.', s)
    s = re.sub("[^a-zA-Z.!?,\[\]'']+", ' ', s)
    for p in punctuation:
        if p not in ["'", "[", "]"]:
            s = s.replace(p, " " + p + " ")
    s = re.sub(' +', ' ', s)
    s = s.strip()
    return s


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
SRC = data.Field(pad_token=BLANK_WORD)
TGT = data.Field(init_token=BOS_WORD,
                 eos_token=EOS_WORD, pad_token=BLANK_WORD)

# In[ ]:


import pandas as pd

train_data = {'src': [line for line in masked_train], 'trg': [line for line in unmasked_train]}
train = pd.DataFrame(train_data, columns=["src", "trg"])

val_data = {'src': [line for line in masked_val], 'trg': [line for line in unmasked_val]}
val = pd.DataFrame(val_data, columns=["src", "trg"])

test_data = {'src': [line for line in masked_test], 'trg': [line for line in unmasked_test]}
test = pd.DataFrame(test_data, columns=["src", "trg"])

train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)

# In[ ]:


data_fields = [('src', SRC), ('trg', TGT)]
train, val, test = data.TabularDataset.splits(path=data_path, train='train.csv', validation='val.csv', test='test.csv',
                                              format='csv', fields=data_fields)

# In[ ]:


SRC.build_vocab(train, val)
TGT.build_vocab(train, val)


# ## Iterators

# In[ ]:


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    src, trg = src.cuda(), trg.cuda()
    return Batch(src, trg, pad_idx)


# ## Training

# In[ ]:


if True:
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model = model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion = criterion.cuda()
    BATCH_SIZE = 4096
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)

# ## Training the System

# In[ ]:


import time

start = time.time()

# model = torch.load('news_transformer_10_full.pt') #uncomment to continue training from a saved model (e.g. news_transformer_10_full.pt, set start=10)
start = 0  # change this depending on model loaded on previous line
log_f = open("SMERTI_loss_log.txt", 'a')
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
total_epochs = 15

for epoch in range(total_epochs):
    print("Beginning epoch ", epoch + 1 + start)
    model.train()
    run_epoch((rebatch(pad_idx, b) for b in train_iter),
              model,
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    with torch.no_grad():
        test_loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model,
                              SimpleLossCompute(model.generator, criterion, None))
        print("validation_loss: ", test_loss)
    log_f.write(str(epoch + 1 + start) + ' | ' + str(test_loss) + '\n')

    path = "news_transformer_{}.pt".format((epoch + 1 + start))
    path2 = "news_transformer_{}_full.pt".format((epoch + 1 + start))
    torch.save(model.state_dict(), path)
    torch.save(model, path2)

log_f.close()

end = time.time()
print(end - start)

# <div id="disqus_thread"></div>
# <script>
#     /**
#      *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
#      *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables
#      */
#     /*
#     var disqus_config = function () {
#         this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
#         this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
#     };
#     */
#     (function() {  // REQUIRED CONFIGURATION VARIABLE: EDIT THE SHORTNAME BELOW
#         var d = document, s = d.createElement('script');
#
#         s.src = 'https://EXAMPLE.disqus.com/embed.js';  // IMPORTANT: Replace EXAMPLE with your forum shortname!
#
#         s.setAttribute('data-timestamp', +new Date());
#         (d.head || d.body).appendChild(s);
#     })();
# </script>
# <noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript" rel="nofollow">comments powered by Disqus.</a></noscript>
