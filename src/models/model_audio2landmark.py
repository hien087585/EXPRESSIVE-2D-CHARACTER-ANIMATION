"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import math
import torch.nn.functional as F
import copy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_FEAT_SIZE = 161
FACE_ID_FEAT_SIZE = 204
Z_SIZE = 16
EPSILON = 1e-40


class Audio2landmark_content(nn.Module):

    def __init__(self, num_window_frames=18, in_size=80, lstm_size=AUDIO_FEAT_SIZE, use_prior_net=False, hidden_size=256, num_layers=3, drop_out=0, bidirectional=False):
        super(Audio2landmark_content, self).__init__()

        self.fc_prior = self.fc = nn.Sequential(
            nn.Linear(in_features=in_size, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, lstm_size),
        )

        self.use_prior_net = use_prior_net
        if(use_prior_net):
            self.bilstm = nn.LSTM(input_size=lstm_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout=drop_out,
                                  bidirectional=bidirectional,
                                  batch_first=True, )
        else:
            self.bilstm = nn.LSTM(input_size=in_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout=drop_out,
                                  bidirectional=bidirectional,
                                  batch_first=True, )

        self.in_size = in_size
        self.lstm_size = lstm_size
        self.num_window_frames = num_window_frames

        self.fc_in_features = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.fc_in_features + FACE_ID_FEAT_SIZE, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 204),
        )



    def forward(self, au, face_id):

        inputs = au
        if(self.use_prior_net):
            inputs = self.fc_prior(inputs.contiguous().view(-1, self.in_size))
            inputs = inputs.view(-1, self.num_window_frames, self.lstm_size)

        output, (hn, cn) = self.bilstm(inputs)
        output = output[:, -1, :]

        if(face_id.shape[0] == 1):
            face_id = face_id.repeat(output.shape[0], 1)
        output2 = torch.cat((output, face_id), dim=1)

        output2 = self.fc(output2)
        # output += face_id

        return output2, face_id



class Embedder(nn.Module):
    def __init__(self, feat_size, d_model):
        super().__init__()
        self.embed = nn.Linear(feat_size, d_model)
    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len].clone().detach().to(device)
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

    # build a decoder layer with two multi-head attention layers and
    # one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)

        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

    # We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, in_size):
        super().__init__()
        self.N = N
        self.embed = Embedder(in_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, in_size):
        super().__init__()
        self.N = N
        self.embed = Embedder(in_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, x, e_outputs, src_mask=None, trg_mask=None):
        x = self.embed(x)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Audio2landmark_pos(nn.Module):

    def __init__(self, audio_feat_size=80, c_enc_hidden_size=256, num_layers=3, drop_out=0,
                 spk_feat_size=256, spk_emb_enc_size=128, lstm_g_win_size=64, add_info_size=6,
                 transformer_d_model=32, N=2, heads=2, z_size=128, audio_dim=256):
        super(Audio2landmark_pos, self).__init__()

        self.lstm_g_win_size = lstm_g_win_size
        self.add_info_size = add_info_size
        comb_mlp_size = c_enc_hidden_size * 2

        self.audio_content_encoder = nn.LSTM(input_size=audio_feat_size,
                                             hidden_size=c_enc_hidden_size,
                                             num_layers=num_layers,
                                             dropout=drop_out,
                                             bidirectional=False,
                                             batch_first=True)

        self.use_audio_projection = not (audio_dim == c_enc_hidden_size)
        if(self.use_audio_projection):
            self.audio_projection = nn.Sequential(
                nn.Linear(in_features=c_enc_hidden_size, out_features=256),
                nn.LeakyReLU(0.02),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.02),
                nn.Linear(128, audio_dim),
            )


        ''' original version '''
        self.spk_emb_encoder = nn.Sequential(
            nn.Linear(in_features=spk_feat_size, out_features=256),
            nn.LeakyReLU(0.02),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.02),
            nn.Linear(128, spk_emb_enc_size),
        )
        # self.comb_mlp = nn.Sequential(
        #     nn.Linear(in_features=audio_dim + spk_emb_enc_size, out_features=comb_mlp_size),
        #     nn.LeakyReLU(0.02),
        #     nn.Linear(comb_mlp_size, comb_mlp_size // 2),
        #     nn.LeakyReLU(0.02),
        #     nn.Linear(comb_mlp_size // 2, 180),
        # )

        d_model = transformer_d_model * heads
        N = N
        heads = heads

        self.encoder = Encoder(d_model, N, heads, in_size=audio_dim + spk_emb_enc_size + z_size)
        self.decoder = Decoder(d_model, N, heads, in_size=204)
        self.out = nn.Sequential(
            nn.Linear(in_features=d_model + z_size, out_features=512),
            nn.LeakyReLU(0.02),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02),
            nn.Linear(256, 204),
        )


    def forward(self, au, emb, face_id, fls, z, add_z_spk=False, another_emb=None):

        # audio
        audio_encode, (_, _) = self.audio_content_encoder(au)
        audio_encode = audio_encode[:, -1, :]

        if(self.use_audio_projection):
            audio_encode = self.audio_projection(audio_encode)

        # spk
        spk_encode = self.spk_emb_encoder(emb)
        if(add_z_spk):
            z_spk = torch.tensor(torch.randn(spk_encode.shape)*0.01, requires_grad=False, dtype=torch.float).to(device)
            spk_encode = spk_encode + z_spk

        # comb
        # comb_input = torch.cat((audio_encode, spk_encode), dim=1)
        # comb_encode = self.comb_mlp(comb_input)
        comb_encode = torch.cat((audio_encode, spk_encode, z), dim=1)
        src_feat = comb_encode.unsqueeze(0)

        e_outputs = self.encoder(src_feat)[0]

        e_outputs = torch.cat((e_outputs, z), dim=1)

        fl_pred = self.out(e_outputs)

        return fl_pred, face_id[0:1, :], spk_encode




def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = torch.tensor(torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.to(device)
    return np_mask


def create_masks(src, trg):
    src_mask = (src != torch.zeros_like(src, requires_grad=False))

    if trg is not None:
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size)
        np_mask = np_mask.to(device)
        trg_mask = np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask


class TalkingToon_spk2res_lstmgan_DL(nn.Module):
    def __init__(self, comb_emb_size=256, input_size=6):
        super(TalkingToon_spk2res_lstmgan_DL, self).__init__()

        self.fl_D = nn.Sequential(
            nn.Linear(in_features=FACE_ID_FEAT_SIZE, out_features=512),
            nn.LeakyReLU(0.02),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02),
            nn.Linear(256, 1),
        )

    def forward(self, feat):
        d = self.fl_D(feat)
        # d = torch.sigmoid(d)
        return d


class Transformer_DT(nn.Module):
    def __init__(self, transformer_d_model=32, N=2, heads=2, spk_emb_enc_size=128):
        super(Transformer_DT, self).__init__()
        d_model = transformer_d_model * heads
        self.encoder = Encoder(d_model, N, heads, in_size=204 + spk_emb_enc_size)
        self.out = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=512),
            nn.LeakyReLU(0.02),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02),
            nn.Linear(256, 1),
        )

    def forward(self, fls, spk_emb, win_size=64, win_step=1):
        feat = torch.cat((fls, spk_emb), dim=1)

        win_size = feat.shape[0]-1 if feat.shape[0] <= win_size else win_size
        D_input = [feat[i:i+win_size:win_step] for i in range(0, feat.shape[0]-win_size)]
        D_input = torch.stack(D_input, dim=0)
        D_output = self.encoder(D_input)
        D_output = torch.max(D_output, dim=1, keepdim=False)[0]
        d = self.out(D_output)
        # d = torch.sigmoid(d)
        return d


class TalkingToon_spk2res_lstmgan_DT(nn.Module):
    def __init__(self, comb_emb_size=256, lstm_g_hidden_size=256, num_layers=3, drop_out=0, input_size=6):
        super(TalkingToon_spk2res_lstmgan_DT, self).__init__()

        self.fl_DT = nn.GRU(input_size=comb_emb_size + FACE_ID_FEAT_SIZE,
                                             hidden_size=lstm_g_hidden_size,
                                             num_layers=3,
                                             dropout=0,
                                             bidirectional=False,
                                             batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(in_features=lstm_g_hidden_size, out_features=512),
            nn.LeakyReLU(0.02),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02),
            nn.Linear(256, 1),
        )

        self.maxpool = nn.MaxPool1d(4, 1)

    def forward(self, comb_encode, fls, win_size=32, win_step=1):
        feat = torch.cat((comb_encode, fls), dim=1)
        # v
        # feat = torch.cat((comb_encode[0:-1], fls[1:] - fls[0:-1]), dim=1)

        # max pooling
        feat = feat.transpose(0, 1).unsqueeze(0)
        feat = self.maxpool(feat)
        feat = feat[0].transpose(0, 1)

        win_size = feat.shape[0] - 1 if feat.shape[0] <= win_size else win_size
        D_input = [feat[i:i+win_size:win_step] for i in range(0, feat.shape[0]-win_size)]
        D_input = torch.stack(D_input, dim=0)
        D_output, _ = self.fl_DT(D_input)
        D_output = D_output[:, -1, :]
        d = self.projection(D_output)
        # d = torch.sigmoid(d)
        return d


# ==== BEGIN StudentAudio2Landmark ====
# Light-weight student network: Audio (mel 80D) -> 68x2 = 136 landmarks (2D, head-pose aligned).
# Designed to replace the heavy Transformer/LSTM/GAN stack of MakeItTalk and to
# run on weak GPUs / CPU / edge devices.
#
# Phase 1 : train end-to-end with `emotion_dim=0`.
# Phase 2 : call `set_emotion_dim(d)` (or rebuild with `from_checkpoint(..., emotion_dim=d)`)
#           to fuse an extra emotion vector. The Phase-1 convolutional weights are
#           invariant to that change, and the newly added FC columns are zero-initialised,
#           so a Phase-1 checkpoint keeps producing identical outputs right after the upgrade.
#
# NOTE: the notebook imports this block from the repo (it is not copied into the notebook).
# 2D: landmarks are aligned to a canonical head pose and the z axis is dropped (see make_aligned.py).

STUDENT_AUDIO_DIM = 80
STUDENT_NUM_POINTS = 68
STUDENT_COORDS = 2                                            # 2D = (x, y); set 3 to bring z back
STUDENT_LANDMARK_DIM = STUDENT_NUM_POINTS * STUDENT_COORDS    # 136
STUDENT_MOUTH_IDX = list(range(48, 68))  # 20 mouth landmarks (outer 48-59 + inner 60-67)


class _ConvBlock1d(nn.Module):
    """Conv1d -> BatchNorm -> GELU, twice, with a stride/dilation aware residual path."""

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 dilation=1, drop_out=0.0, use_norm=True):
        super(_ConvBlock1d, self).__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm1d(out_channels) if use_norm else nn.Identity()
        self.norm2 = nn.BatchNorm1d(out_channels) if use_norm else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_out)
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.drop(self.act(self.norm1(self.conv1(x))))
        out = self.act(self.norm2(self.conv2(out)))
        return self.drop(out + residual)


class _TemporalRefine1d(nn.Module):
    """Depthwise residual block that widens the temporal receptive field cheaply."""

    def __init__(self, channels, kernel_size=3, dilation=2, drop_out=0.0, use_norm=True):
        super(_TemporalRefine1d, self).__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=padding,
                                   dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.BatchNorm1d(channels) if use_norm else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_out)

    def forward(self, x):
        out = self.act(self.norm(self.depthwise(x)))
        out = self.pointwise(out)
        return self.drop(out + x)


class _TemporalUpsampler(nn.Module):
    """Restore the downsampled temporal axis back to the original frame count.

    mode='interp'  -> parameter free (default, best for edge devices)
    mode='learned' -> nearest-neighbour upscale followed by a Conv1d per stage
    """

    def __init__(self, channels, strides, mode='interp'):
        super(_TemporalUpsampler, self).__init__()
        assert mode in ('interp', 'learned'), 'upsample mode must be interp|learned'
        self.mode = mode
        blocks = []
        if mode == 'learned':
            for s in strides:
                if s == 1:
                    continue
                blocks.append(nn.Sequential(
                    nn.Upsample(scale_factor=s, mode='nearest'),
                    nn.Conv1d(channels, channels, 3, padding=1),
                    nn.GELU(),
                ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, target_len=None):
        if self.mode == 'learned':
            x = self.blocks(x)
        if target_len is not None and x.shape[-1] != target_len:
            x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        return x


class StudentAudio2Landmark(nn.Module):
    """Light-weight 1D-CNN + FC student: audio mel-spectrogram -> 2D landmarks (136 = 68 x 2).

    Input
    -----
    mel      : (B, T, audio_dim) when input_layout='bot' (default), or (B, audio_dim, T) for 'bft'.
    emotion  : (B, emotion_dim) optional Phase-2 emotion vector.

    Output
    ------
    landmarks : (B, T, landmark_dim) with landmark_dim = 68 * STUDENT_COORDS (= 136 for 2D),
                laid out as 68 consecutive (x, y) pairs (or (x, y, z) triplets when COORDS = 3).

    Modularity for Phase 2
    ----------------------
    The emotion vector is fused *after* the convolutional stack and *after* the
    temporal upsampling, i.e. exactly at the boundary of the FC head. Consequently
    all convolutional kernels keep their Phase-1 shape, and only the first FC layer
    grows. `set_emotion_dim()` performs that surgery while copying every Phase-1
    weight it can and zero-initialising the new emotion columns.
    """

    def __init__(self,
                 audio_dim=STUDENT_AUDIO_DIM,
                 landmark_dim=STUDENT_LANDMARK_DIM,
                 emotion_dim=0,
                 cnn_channels=(128, 256, 256, 384),
                 kernel_size=5,
                 strides=(2, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 refine_dilation=2,
                 fc_hidden=512,
                 drop_out=0.2,
                 upsample='interp',
                 emotion_fuse='concat',
                 emotion_bottleneck=None,
                 use_norm=True):
        super(StudentAudio2Landmark, self).__init__()

        cnn_channels = tuple(cnn_channels)
        strides = tuple(strides)
        dilations = tuple(dilations)
        assert len(cnn_channels) == len(strides) == len(dilations), \
            'cnn_channels / strides / dilations must have the same length'
        assert emotion_fuse in ('concat', 'add'), 'emotion_fuse must be concat|add'
        assert upsample in ('interp', 'learned'), 'upsample must be interp|learned'

        self.audio_dim = int(audio_dim)
        self.landmark_dim = int(landmark_dim)
        self.cnn_channels = cnn_channels
        self.kernel_size = int(kernel_size)
        self.strides = strides
        self.dilations = dilations
        self.refine_dilation = int(refine_dilation)
        self.fc_hidden = int(fc_hidden)
        self.drop_out = float(drop_out)
        self.upsample = upsample
        self.emotion_fuse = emotion_fuse
        self.emotion_dim = int(emotion_dim)
        self.emotion_bottleneck = int(emotion_bottleneck) if emotion_bottleneck \
            else int(cnn_channels[-1])
        self.use_norm = bool(use_norm)

        blocks = []
        in_ch = self.audio_dim
        for out_ch, stride, dilation in zip(self.cnn_channels, self.strides, self.dilations):
            blocks.append(_ConvBlock1d(in_ch, out_ch, kernel_size=self.kernel_size,
                                       stride=stride, dilation=dilation,
                                       drop_out=self.drop_out, use_norm=self.use_norm))
            in_ch = out_ch
        self.audio_encoder = nn.Sequential(*blocks)
        self.encoder_out_dim = in_ch

        self.temporal_refine = _TemporalRefine1d(self.encoder_out_dim,
                                                 kernel_size=3,
                                                 dilation=self.refine_dilation,
                                                 drop_out=self.drop_out,
                                                 use_norm=self.use_norm)
        self.upsampler = _TemporalUpsampler(self.encoder_out_dim, self.strides,
                                            mode=self.upsample)
        self.smooth = nn.Sequential(
            nn.Conv1d(self.encoder_out_dim, self.encoder_out_dim, 3, padding=1),
            nn.GELU(),
        )

        self._build_emotion_and_head()

    # ------------------------------------------------------------------ builders

    def _build_emotion_and_head(self):
        """(Re)build the emotion projection and the FC head for the current config."""
        if self.emotion_dim > 0:
            self.emotion_proj = nn.Linear(self.emotion_dim, self.emotion_bottleneck)
            # A trainable per-channel gate is what makes the fusion a strict no-op on
            # top of the Phase-1 weights, instead of zero-initialising the projection.
            # Zero-initialising BOTH the projection and the new head columns would
            # multiply to zero and the emotion branch would never receive a gradient.
            # Here the gate starts at 1 while the *head columns* it feeds start at 0,
            # so: output delta is exactly 0 at init, yet the head columns already get
            # a non-zero gradient on the first backward pass.
            self.emotion_gate = nn.Parameter(torch.ones(self.emotion_bottleneck))
        else:
            self.emotion_proj = None
            self.emotion_gate = None

        if self.emotion_dim > 0 and self.emotion_fuse == 'concat':
            head_in = self.encoder_out_dim + self.emotion_bottleneck
        else:
            head_in = self.encoder_out_dim
        self.head_in_dim = head_in

        self.fc = nn.Sequential(
            nn.Linear(head_in, self.fc_hidden),
            nn.GELU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.fc_hidden, max(self.fc_hidden // 2, self.landmark_dim)),
            nn.GELU(),
            nn.Linear(max(self.fc_hidden // 2, self.landmark_dim), self.landmark_dim),
        )

        if self.emotion_dim > 0 and self.emotion_fuse == 'concat':
            with torch.no_grad():
                self.fc[0].weight[:, self.encoder_out_dim:].zero_()

    # ------------------------------------------------------------------- helpers

    @staticmethod
    def _as_bct(mel, layout):
        """Return audio features as (B, channels=audio_dim, T)."""
        if layout == 'bot':                      # (B, T, F)
            return mel.transpose(1, 2)
        if layout == 'bft':                      # (B, F, T)
            return mel
        raise ValueError("input_layout must be 'bot' (B,T,F) or 'bft' (B,F,T)")

    # ------------------------------------------------------------------- forward

    def forward(self, mel, emotion=None, input_layout='bot'):
        x = self._as_bct(mel, input_layout)
        num_frames = x.shape[-1]

        feat = self.audio_encoder(x)                     # (B, C, T')
        feat = self.temporal_refine(feat)                # (B, C, T')
        feat = self.upsampler(feat, num_frames)          # (B, C, T)
        feat = self.smooth(feat)                         # (B, C, T)

        if self.emotion_dim > 0:
            if emotion is None:
                emotion = feat.new_zeros(feat.shape[0], self.emotion_dim)
            emo = self.emotion_proj(emotion.to(feat.dtype))          # (B, eb)
            emo = emo * self.emotion_gate
            emo = emo.unsqueeze(-1).expand(-1, -1, num_frames)       # (B, eb, T)
            if self.emotion_fuse == 'concat':
                feat = torch.cat([feat, emo], dim=1)
            else:
                feat = feat + emo

        out = self.fc(feat.transpose(1, 2))              # (B, T, landmark_dim)
        return out

    # ------------------------------------------------------- Phase-2 upgrade API

    def set_emotion_dim(self, emotion_dim, emotion_fuse=None, emotion_bottleneck=None):
        """Grow / shrink the emotion branch while preserving every Phase-1 weight.

        NOTE: the rebuilt layers are created on the CPU. If the model already lives on the GPU,
        call ``.to(device)`` afterwards and rebuild the optimizer (new parameters are not in the
        old one). The output stays identical to Phase 1 only for ``emotion_fuse='concat'``.
        """
        emotion_fuse = emotion_fuse or self.emotion_fuse
        emotion_bottleneck = int(emotion_bottleneck) if emotion_bottleneck \
            else self.emotion_bottleneck
        if (int(emotion_dim) == self.emotion_dim and emotion_fuse == self.emotion_fuse
                and emotion_bottleneck == self.emotion_bottleneck):
            return self

        previous_state = self.state_dict()
        self.emotion_dim = int(emotion_dim)
        self.emotion_fuse = emotion_fuse
        self.emotion_bottleneck = emotion_bottleneck
        self._build_emotion_and_head()
        self.load_state_dict_flexible(previous_state)
        return self

    def load_state_dict_flexible(self, state_dict, verbose=False):
        """Load a checkpoint, adapting the FC head when the emotion branch grew.

        Tensors whose name is unknown or whose shape changed for any reason other
        than the documented head growth are silently left at their fresh init.
        """
        own = self.state_dict()
        adapted, skipped = {}, []
        for key, value in state_dict.items():
            if key not in own:
                skipped.append((key, 'unknown'))
                continue
            if own[key].shape == value.shape:
                adapted[key] = value
            elif key == 'emotion_gate' and value.shape != own[key].shape:
                # keep whatever the previous run had learned for the shared channels
                k = min(value.shape[0], own[key].shape[0])
                grown = own[key].clone()
                grown[:k] = value[:k]
                adapted[key] = grown
            elif key == 'fc.0.weight' and value.shape[1] < own[key].shape[1]:
                # Copy every Phase-1 column verbatim; the extra emotion columns keep
                # their dedicated zero-init (see `_build_emotion_and_head`), which is
                # what makes the upgrade a strict no-op. They still receive gradient
                # on the first backward pass thanks to the non-zero `emotion_gate`.
                grown = own[key].clone()
                grown[:, :value.shape[1]] = value
                adapted[key] = grown
            else:
                skipped.append((key, 'shape {} -> {}'.format(tuple(value.shape),
                                                             tuple(own[key].shape))))
        missing, unexpected = self.load_state_dict(adapted, strict=False)
        if verbose and (skipped or missing or unexpected):
            print('[StudentAudio2Landmark] kept %d tensors, %d skipped, %d missing'
                  % (len(adapted), len(skipped), len(missing)))
            for key, why in skipped:
                print('   skipped {} ({})'.format(key, why))
        return adapted

    def set_encoder_trainable(self, flag):
        """Freeze / unfreeze the convolutional backbone (Phase-2 fine-tuning)."""
        for module in (self.audio_encoder, self.temporal_refine, self.upsampler, self.smooth):
            for param in module.parameters():
                param.requires_grad = bool(flag)
        return self

    @property
    def downsample_factor(self):
        return int(np.prod(self.strides))

    def num_parameters(self, trainable_only=False):
        params = [p for p in self.parameters() if (p.requires_grad or not trainable_only)]
        return int(sum(p.numel() for p in params))

    def config(self):
        return {
            'audio_dim': self.audio_dim,
            'landmark_dim': self.landmark_dim,
            'emotion_dim': self.emotion_dim,
            'cnn_channels': list(self.cnn_channels),
            'kernel_size': self.kernel_size,
            'strides': list(self.strides),
            'dilations': list(self.dilations),
            'refine_dilation': self.refine_dilation,
            'fc_hidden': self.fc_hidden,
            'drop_out': self.drop_out,
            'upsample': self.upsample,
            'emotion_fuse': self.emotion_fuse,
            'emotion_bottleneck': self.emotion_bottleneck,
            'use_norm': self.use_norm,
        }

    @classmethod
    def from_config(cls, cfg, **overrides):
        merged = dict(cfg)
        merged['cnn_channels'] = tuple(merged.get('cnn_channels', (128, 256, 256, 384)))
        merged['strides'] = tuple(merged.get('strides', (2, 2, 2, 2)))
        merged['dilations'] = tuple(merged.get('dilations', (1, 1, 1, 1)))
        merged.update(overrides)
        return cls(**merged)

    @classmethod
    def from_checkpoint(cls, ckpt_path, map_location='cpu', emotion_dim=None,
                        emotion_fuse=None, emotion_bottleneck=None, verbose=False,
                        eval_mode=True):
        """Rebuild the student and load Phase-1 weights, optionally upgrading to Phase 2.

        The returned model is put in ``eval()`` mode by default (``eval_mode=True``).
        This matters because the student carries a dropout head: a freshly loaded model
        still in training mode would produce non-deterministic predictions. Pass
        ``eval_mode=False`` if you intend to keep training the returned model and will
        call ``model.train()`` yourself.
        """
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        cfg = dict(ckpt.get('config') or {})
        if emotion_dim is not None:
            cfg['emotion_dim'] = int(emotion_dim)
        if emotion_fuse is not None:
            cfg['emotion_fuse'] = emotion_fuse
        if emotion_bottleneck is not None:
            cfg['emotion_bottleneck'] = int(emotion_bottleneck)
        model = cls.from_config(cfg)
        state = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt.get('model')
        if state is None:
            raise KeyError('checkpoint has no model_state_dict / state_dict / model entry')
        model.load_state_dict_flexible(state, verbose=verbose)
        if eval_mode:
            model.eval()
        return model

    def save_checkpoint(self, path, optimizer=None, scheduler=None, epoch=None,
                        step=None, best_val=None, extra=None):
        payload = {
            'config': self.config(),
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            'step': step,
            'best_val': best_val,
        }
        if optimizer is not None:
            payload['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            payload['scheduler_state_dict'] = scheduler.state_dict()
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path


# ==== END StudentAudio2Landmark ====