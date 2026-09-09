'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza
Apache 2.0 License
2023
'''

import torch
import torch.nn.functional as F
from torch import nn
from text.symbols import symbols
from einops import repeat






class DynamicTanh(nn.Module):
    def __init__(
        self,
        normalized_shape,
        channels_last=True,
        alpha_init_value=0.5,
        activation="tanh",
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last
        self.activation = activation

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def splitanh(self, x):
        a = 0.75
        B = 1.25

        z = self.alpha * x
        u = torch.abs(z)
        sign = torch.sign(z)

        t = (u - a) / (B - a)
        y_turn = a + (1 - a) * (2 * t - t * t)

        y_abs = torch.where(u <= a, u, y_turn)
        y_abs = torch.where(u >= B, torch.ones_like(u), y_abs)

        return sign * y_abs


    def forward(self, x):
        if self.activation == "splitanh":
            x = self.splitanh(x)
        else:
            x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x




class Encoder(nn.Module):
    """ Phoneme Encoder """

    def __init__(self):
        super().__init__()

        embed_dim=80
        kernel_size=5
        dim_out = 32
        self.embed = nn.Embedding(len(symbols) + 1, embed_dim, padding_idx=0)
        self.b1_conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.b1_dyt1 = DynamicTanh(embed_dim)
        self.b1_conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.b1_dyt2 = DynamicTanh(embed_dim)
        self.b2_conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.b2_dyt1 = DynamicTanh(embed_dim)
        self.b2_conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.b2_dyt2 = DynamicTanh(embed_dim)
        self.linear = nn.Linear(embed_dim, dim_out)


    def forward(self, phoneme, mask=None):

        skip = self.embed(phoneme)

        x = skip.permute(0, 2, 1)
        x = self.b1_conv1(x)
        x = x.permute(0, 2, 1)
        x = self.b1_dyt1(x)
        x = x.permute(0, 2, 1)
        x = self.b1_conv2(x)
        x = x.permute(0, 2, 1)
        x = self.b1_dyt2(x+skip)

        skip = x
        x = skip.permute(0, 2, 1)
        x = self.b2_conv1(x)
        x = x.permute(0, 2, 1)
        x = self.b2_dyt1(x)
        x = x.permute(0, 2, 1)
        x = self.b2_conv2(x)
        x = x.permute(0, 2, 1)
        x = self.b2_dyt2(x+skip)
        x_32 = self.linear(x)

        return x_32


class AcousticDecoder(nn.Module):
    """ Pitch, Duration, Energy Predictor """

    def __init__(self,
                 pitch_stats=None,
                 energy_stats=None,
                 duration_stats=None,
                 duration=False):
        super().__init__()
        
        dim=32
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.dyt1 = DynamicTanh(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.dyt2 = DynamicTanh(dim)
        self.linear = nn.Linear(dim, 1)
        self.duration = duration
        
        if pitch_stats is not None:
            pitch_min, pitch_max = pitch_stats
            self.pitch_bins = nn.Parameter(torch.linspace(pitch_min, pitch_max, dim - 1),\
                                           requires_grad=False,)
            self.pitch_embedding = nn.Embedding(dim, dim)
        else:
            self.pitch_bins = None
            self.pitch_embedding = None

        if energy_stats is not None:
            energy_min, energy_max = energy_stats
            self.energy_bins = nn.Parameter(torch.linspace(energy_min, energy_max, dim - 1), \
                                            requires_grad=False,)
            self.energy_embedding = nn.Embedding(dim, dim)
        else:
            self.energy_bins = None
            self.energy_embedding = None

        if duration_stats is not None:
            duration_min, duration_max = duration_stats
            self.duration_bins = nn.Parameter(torch.linspace(duration_min, duration_max, dim - 1), \
                                              requires_grad=False,)
            self.duration_embedding = nn.Embedding(dim, dim)
        else:
            self.duration_bins = None
            self.duration_embedding = None


    def get_pitch_embedding(self, pred, target, mask, control=1.):
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            #pred = pred * control
            embedding = self.pitch_embedding(torch.bucketize(pred, self.pitch_bins))
        return embedding

    def get_energy_embedding(self, pred, target, mask, control=1.):
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            #pred = pred * control
            embedding = self.energy_embedding(torch.bucketize(pred, self.energy_bins))
        return embedding

    def get_duration_embedding(self, pred, target, mask, control=1.):
        if target is not None:
            embedding = self.duration_embedding(torch.bucketize(target.float(), self.duration_bins))
        else:
            embedding = self.duration_embedding(torch.bucketize(pred.float(), self.duration_bins))
        return embedding

    def get_embedding(self, pred, target, mask, control=1.):
        if self.pitch_embedding is not None:
            return self.get_pitch_embedding(pred, target, mask, control)
        elif self.energy_embedding is not None:
            return self.get_energy_embedding(pred, target, mask, control)
        elif self.duration_embedding is not None:
            return self.get_duration_embedding(pred, target, mask, control)
        return None

    def forward(self, features):

        y = features.permute(0, 2, 1)
        y = self.conv1(y)
        y = y.permute(0, 2, 1)
        y = self.dyt1(y)        
        y = y.permute(0, 2, 1)
        y = self.conv2(y)
        y = y.permute(0, 2, 1)
        y = self.dyt2(y)
        y = self.linear(y)
        if self.duration:
            y = nn.ReLU()(y)+1
        return y



class FeatureUpsampler(nn.Module):
    """ Upsample fused features using target or predicted duration"""

    def __init__(self):
        super().__init__()

    def forward(self, fused_features, fused_masks, duration, max_mel_len=None):
        mel_len = list()
        features = list()
        masks = list()

        for feature, mask, repetition in zip(fused_features, fused_masks, duration):
            repetition = repetition.squeeze().int()
            feature = feature.repeat_interleave(repetition, dim=0)
            mask = mask.repeat_interleave(repetition, dim=0)
            mel_len.append(feature.shape[0])
            if max_mel_len is not None:
                feature = F.pad(feature, (0, 0, 0, max_mel_len -
                                feature.shape[0]), "constant", 0.0)
                mask = F.pad(mask, (0, 0, 0,  max_mel_len -
                             mask.shape[0]), "constant", True)
            features.append(feature)
            masks.append(mask)

        if max_mel_len is None:
            max_mel_len = max(mel_len)
            features = [F.pad(feature, (0, 0, 0, max_mel_len - feature.shape[0]),
                              "constant", 0.0) for feature in features]
            masks = [F.pad(mask, (0, 0, 0, max_mel_len - mask.shape[0]),
                           "constant", True) for mask in masks]

        features = torch.stack(features)
        masks = torch.stack(masks)
        len_pred = torch.IntTensor(mel_len).to(features.device)
        #len_pred = torch.LongTensor(mel_len).to(features.device)

        return features, masks, len_pred



class MelDecoder(nn.Module):
    """ Mel Spectrogram Decoder """

    def __init__(self, dim, kernel_size=3, n_mel_channels=80,
                 n_blocks=2, block_depth=2,):
        super().__init__()

        dim_mel=80

        self.proj = nn.Conv1d(dim_mel, dim_mel, kernel_size=9, padding=4, dilation=1, groups=dim_mel)

        self.block1_dyt1 = DynamicTanh(dim_mel)
        self.block1_conv1_dw = nn.Conv1d(dim_mel, dim_mel, kernel_size=3, padding=1, dilation=1, groups=dim_mel)
        self.block1_conv1_pw = nn.Conv1d(dim_mel, dim_mel, kernel_size=1)
        
        self.block1_dyt2 = DynamicTanh(dim_mel)
        self.block1_conv2_dw = nn.Conv1d(dim_mel, dim_mel, kernel_size=3, padding=1, dilation=1, groups=dim_mel)
        self.block1_conv2_pw = nn.Conv1d(dim_mel, dim_mel, kernel_size=1)
        
        self.block2_dyt1 = DynamicTanh(dim_mel)
        self.block2_conv1_dw = nn.Conv1d(dim_mel, dim_mel, kernel_size=3, padding=3, dilation=3, groups=dim_mel)
        self.block2_conv1_pw = nn.Conv1d(dim_mel, dim_mel, kernel_size=1)
        self.block2_dyt2 = DynamicTanh(dim_mel)
        self.block2_conv2_dw = nn.Conv1d(dim_mel, dim_mel, kernel_size=3, padding=1, dilation=1, groups=dim_mel)
        self.block2_conv2_pw = nn.Conv1d(dim_mel, dim_mel, kernel_size=1)

        self.block3_dyt1 = DynamicTanh(dim_mel)
        self.block3_conv1_dw = nn.Conv1d(dim_mel, dim_mel, kernel_size=3, padding=5, dilation=5, groups=dim_mel)
        self.block3_conv1_pw = nn.Conv1d(dim_mel, dim_mel, kernel_size=1)
        self.block3_dyt2 = DynamicTanh(dim_mel)
        self.block3_conv2_dw = nn.Conv1d(dim_mel, dim_mel, kernel_size=3, padding=1, dilation=1, groups=dim_mel)
        self.block3_conv2_pw = nn.Conv1d(dim_mel, dim_mel, kernel_size=1)

        self.mel_linear_up = nn.Linear(dim_mel, 4*dim_mel)
        self.mel_dyt = DynamicTanh(4*dim_mel)
        self.mel_linear_down = nn.Linear(4*dim_mel, dim_mel)


    def forward(self, features):
        x = features
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = x.permute(0, 2, 1)
        
        skip = x
        x = self.block1_dyt1(skip)
        x = x.permute(0, 2, 1)
        x = self.block1_conv1_dw(x)
        x = self.block1_conv1_pw(x)
        x = x.permute(0, 2, 1)
        x = self.block1_dyt2(x)
        x = x.permute(0, 2, 1)
        x = self.block1_conv2_dw(x)
        x = self.block1_conv2_pw(x)
        x = x.permute(0, 2, 1)

        skip = x + skip
        x = self.block2_dyt1(skip)
        x = x.permute(0, 2, 1)
        x = self.block2_conv1_dw(x)
        x = self.block2_conv1_pw(x)
        x = x.permute(0, 2, 1)
        x = self.block2_dyt2(x)
        x = x.permute(0, 2, 1)
        x = self.block2_conv2_dw(x)
        x = self.block2_conv2_pw(x)
        x = x.permute(0, 2, 1)

        skip = x + skip
        x = self.block3_dyt1(skip)
        x = x.permute(0, 2, 1)
        x = self.block3_conv1_dw(x)
        x = self.block3_conv1_pw(x)
        x = x.permute(0, 2, 1)
        x = self.block3_dyt2(x)
        x = x.permute(0, 2, 1)
        x = self.block3_conv2_dw(x)
        x = self.block3_conv2_pw(x)
        x = x.permute(0, 2, 1)
        
        mel = self.mel_linear_up(x + skip)
        mel = self.mel_dyt(mel)
        mel = self.mel_linear_down(mel)

        return mel


class PhonemeEncoder(nn.Module):
    """ Encodes phonemes to acoustic features """

    def __init__(self,
                 pitch_stats=None, 
                 energy_stats=None, 
                 depth=2, 
                 reduction=4, 
                 head=1, 
                 embed_dim=128, 
                 kernel_size=3, 
                 expansion=1):
        super().__init__()

        self.encoder = Encoder()
        
        
        self.feature_upsampler = FeatureUpsampler()
        self.pitch_decoder = AcousticDecoder(pitch_stats=pitch_stats)
        self.energy_decoder = AcousticDecoder(energy_stats=energy_stats)
        self.duration_decoder = AcousticDecoder(duration=True, duration_stats=(2, 34))

        self.fusion_linear = nn.Linear(32*4, 80)
        

    def forward(self, x, train=False):
        phoneme = x["phoneme"]
        phoneme_mask = x["phoneme_mask"] if phoneme.shape[0] > 1 else None

        pitch_target = x["pitch"] if train else None
        energy_target = x["energy"] if train  else None
        duration_target = x["duration"] if train  else None
        mel_len = x["mel_len"] if train  else None
        max_mel_len = torch.max(mel_len).item() if train else None

        features = self.encoder(phoneme)

        mask = None
        if phoneme_mask is not None:
            mask = repeat(phoneme_mask, 'b n -> b n a', a=features.shape[-1])
            features = features.masked_fill(mask, 0)
        
        pitch_pred = self.pitch_decoder(features)
        pitch_features = self.pitch_decoder.get_embedding(pitch_pred, pitch_target, mask)
        pitch_features = pitch_features.squeeze()
        if mask is not None:
            pitch_features = pitch_features.masked_fill(mask, 0)
        elif pitch_features.dim() != 3:
            pitch_features = pitch_features.unsqueeze(0)

        energy_pred = self.energy_decoder(features)
        energy_features = self.energy_decoder.get_embedding(energy_pred, energy_target, mask)
        energy_features = energy_features.squeeze()

        if mask is not None:
            energy_features = energy_features.masked_fill(mask, 0)
        elif energy_features.dim() != 3:
            energy_features = energy_features.unsqueeze(0)

        duration_pred = self.duration_decoder(features)
        duration_features = self.duration_decoder.get_embedding(duration_pred, duration_target, mask)
        duration_features = duration_features.squeeze()
        if mask is not None:
            duration_features = duration_features.masked_fill(mask, 0)
        elif duration_features.dim() != 3:
            duration_features = duration_features.unsqueeze(0)
       
        fused_features = torch.cat([features, pitch_features, \
                                    energy_features, duration_features], dim=-1)

        fused_features = self.fusion_linear(fused_features)

        if phoneme_mask is not None:
            fused_masks = repeat(phoneme_mask, 'b n -> b n a', a=fused_features.shape[-1])
            fused_features = fused_features.masked_fill(fused_masks, 0)
        else:
            fused_masks = torch.zeros_like(fused_features).bool()
        
        if duration_target is None:
            duration_target = torch.round(duration_pred).squeeze()
        if phoneme_mask is not None:
            duration_target = duration_target.masked_fill(phoneme_mask, 0).clamp(min=0)
        else:
            duration_target = duration_target.unsqueeze(0)

        features, masks, mel_len_pred = self.feature_upsampler(fused_features,
                                                               fused_masks,
                                                               duration=duration_target,
                                                               max_mel_len=max_mel_len,)
    
        if mask is None:
            masks = None

        y = {"pitch": pitch_pred,
             "energy": energy_pred,
             "duration": duration_pred,
             "mel_len": mel_len_pred,
             "features": features,
             "masks": masks, }

        return y

        
class Phoneme2Mel(nn.Module):
    """ From Phoneme Sequence to Mel Spectrogram """

    def __init__(self,
                 encoder,
                 decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, train=False):
        # Dirty trick to enable ONNX compilation.
        # Else, the torch.to_onnx complains about missing input in the forward method.
        if isinstance(x, list):
            x = x[0]
            
        pred = self.encoder(x, train=train)
        mel = self.decoder(pred["features"]) 
        
        mask = pred["masks"]
        if mask is not None and mel.size(0) > 1:
            mask = mask[:, :, :mel.shape[-1]]
            mel = mel.masked_fill(mask, 0)
        
        pred["mel"] = mel

        if train: 
            return pred

        return mel, pred["mel_len"], pred["duration"]
