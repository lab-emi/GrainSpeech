'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza
Apache 2.0 License
2023
'''

import os
import json
import hifigan
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.networks import PhonemeEncoder, MelDecoder, Phoneme2Mel
from lightning import LightningModule
from torch.optim import AdamW
from utils.tools import write_to_file
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


def get_hifigan(checkpoint="hifigan/LJ_V2/generator_v2", infer_device=None, verbose=False):
    # get the main path
    main_path = os.path.dirname(os.path.abspath(checkpoint))
    json_config = os.path.join(main_path, "config.json")
    if verbose:
        print("Using config: ", json_config)
        print("Using hifigan checkpoint: ", checkpoint)
    with open(json_config, "r") as f:
        config = json.load(f)

    config = hifigan.AttrDict(config)
    torch.manual_seed(config.seed)
    vocoder = hifigan.Generator(config)
    if infer_device is not None:
        vocoder.to(infer_device)
        ckpt = torch.load(checkpoint, map_location=torch.device(infer_device))
    else:
        ckpt = torch.load(checkpoint)
        #ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    for p in vocoder.parameters():
        p.requires_grad = False
    
    return vocoder

# bard
def linear_warmup_cosine_annealing_lr(optimizer, num_warmup_steps, num_training_steps, max_lr):
    """
    Implements a learning rate scheduler with linear warm up and then cosine learning rate decay.

    Args:
        optimizer: The optimizer to use.
        num_warmup_steps: The number of steps to use for linear warm up.
        num_training_steps: The total number of training steps.
        max_lr: The maximum learning rate.

    Returns:
        A learning rate scheduler.
    """
    scheduler = CosineAnnealingLR(optimizer, num_training_steps, eta_min=0)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return 0.5 * (1.0 + math.cos(math.pi * (current_step - num_warmup_steps) / float(num_training_steps - num_warmup_steps)))

    scheduler.set_lambda(lr_lambda)

    return scheduler

# chatgpt
def get_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr=0):
    """
    Create a learning rate scheduler with linear warm-up and cosine learning rate decay.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to create the scheduler.
        warmup_steps (int): The number of warm-up steps.
        total_steps (int): The total number of steps.
        min_lr (float, optional): The minimum learning rate at the end of the decay. Default: 0.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warm-up
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine learning rate decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


class MaskedSSIMLoss(nn.Module):
    """Gaussian-window SSIM loss for padded mel spectrograms."""

    def __init__(
        self,
        kernel_size=(11, 5),
        sigma=(1.5, 1.0),
        k1=0.01,
        k2=0.03,
        eps=1e-6,
    ):
        super().__init__()
        if any(size % 2 == 0 for size in kernel_size):
            raise ValueError("SSIM kernel dimensions must be odd.")
        if any(size <= 0 for size in kernel_size):
            raise ValueError("SSIM kernel dimensions must be positive.")
        if len(sigma) != 2 or any(value <= 0 for value in sigma):
            raise ValueError("SSIM Gaussian sigmas must be positive.")

        self.kernel_size = tuple(int(size) for size in kernel_size)
        self.sigma = tuple(float(value) for value in sigma)
        self.k1 = k1
        self.k2 = k2
        self.eps = eps

        kt, kf = self.kernel_size
        sigma_t, sigma_f = self.sigma
        time = torch.arange(kt, dtype=torch.float32) - (kt - 1) / 2
        freq = torch.arange(kf, dtype=torch.float32) - (kf - 1) / 2
        time_kernel = torch.exp(-0.5 * (time / sigma_t).square())
        freq_kernel = torch.exp(-0.5 * (freq / sigma_f).square())
        gaussian_kernel = time_kernel[:, None] * freq_kernel[None, :]
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        # This kernel is deterministic and should not make old checkpoints
        # report a missing state-dict key.
        self.register_buffer(
            "gaussian_kernel",
            gaussian_kernel[None, None],
            persistent=False,
        )

    def _masked_pool(self, value, mask):
        kt, kf = self.kernel_size
        padding = (kt // 2, kf // 2)
        kernel = self.gaussian_kernel.to(device=value.device, dtype=value.dtype)
        valid_weight = F.conv2d(mask, kernel, stride=1, padding=padding)
        weighted_value = F.conv2d(
            value * mask,
            kernel,
            stride=1,
            padding=padding,
        )
        return weighted_value / valid_weight.clamp_min(self.eps)

    def _target_data_range(self, target, mask):
        valid = mask > 0
        target_min = target.masked_fill(~valid, float("inf")).flatten(1).amin(dim=1)
        target_max = target.masked_fill(~valid, float("-inf")).flatten(1).amax(dim=1)
        valid_count = valid.flatten(1).sum(dim=1)
        data_range = torch.where(
            valid_count > 0,
            target_max - target_min,
            torch.ones_like(target_min),
        )
        return data_range.clamp_min(self.eps).detach().view(-1, 1, 1, 1)

    def forward(self, pred_mel, target_mel, valid_mask=None):
        if pred_mel.shape != target_mel.shape:
            raise ValueError("pred_mel and target_mel must have the same shape.")

        batch, time, freq = pred_mel.shape
        # Keep local moment calculations in FP32 under mixed-precision training.
        pred = pred_mel.unsqueeze(1).float()
        target = target_mel.unsqueeze(1).float()

        if valid_mask is None:
            mask = torch.ones_like(pred)
        elif valid_mask.dim() == 2:
            mask = valid_mask[:, None, :, None].expand(batch, 1, time, freq)
        elif valid_mask.dim() == 3:
            mask = valid_mask[:, None, :, :]
        else:
            raise ValueError("valid_mask must be [B,T] or [B,T,F].")
        mask = mask.to(device=pred.device, dtype=pred.dtype)

        mu_pred = self._masked_pool(pred, mask)
        mu_target = self._masked_pool(target, mask)
        var_pred = (self._masked_pool(pred * pred, mask) - mu_pred.square()).clamp_min(0.0)
        var_target = (
            self._masked_pool(target * target, mask) - mu_target.square()
        ).clamp_min(0.0)
        covariance = self._masked_pool(pred * target, mask) - mu_pred * mu_target

        data_range = self._target_data_range(target, mask)
        c1 = (self.k1 * data_range).square()
        c2 = (self.k2 * data_range).square()
        numerator = (2.0 * mu_pred * mu_target + c1) * (2.0 * covariance + c2)
        denominator = (
            (mu_pred.square() + mu_target.square() + c1)
            * (var_pred + var_target + c2)
        ).clamp_min(self.eps)
        ssim = (numerator / denominator).clamp(min=-1.0, max=1.0)

        loss_map = (1.0 - ssim) * mask
        return loss_map.sum() / mask.sum().clamp_min(1.0)


class EfficientSpeech(LightningModule):
    def __init__(self,
                 preprocess_config, 
                 lr=1e-3,
                 weight_decay=1e-6, 
                 max_epochs=5000,
                 depth=1, 
                 n_blocks=2, 
                 block_depth=2, 
                 reduction=4, 
                 head=1,
                 embed_dim=128, 
                 kernel_size=3, 
                 decoder_kernel_size=3, 
                 expansion=1,
                 wav_path="wavs",
                 hifigan_checkpoint="hifigan/LJ_V2/generator_v2",
                 infer_device=None,
                 verbose=False,
                 constant_lr=False,
                 ssim_kernel_time=11,
                 ssim_kernel_freq=5,
                 ssim_sigma_time=1.5,
                 ssim_sigma_freq=1.0,
                 ssim_k1=0.01,
                 ssim_k2=0.03,
                 mel_weight=5.0,
                 l1_weight=1.0,
                 ssim_weight=1.0,
                 pitch_weight=2.0,
                 energy_weight=2.0,
                 duration_weight=1.0):
        super(EfficientSpeech, self).__init__()

        self.save_hyperparameters()

        
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
            stats = json.load(f)
            pitch_stats = stats["pitch"][:2]
            energy_stats = stats["energy"][:2]

        phoneme_encoder = PhonemeEncoder(pitch_stats=pitch_stats,
                                         energy_stats=energy_stats,
                                         depth=depth,
                                         reduction=reduction,
                                         head=head,
                                         embed_dim=embed_dim,
                                         kernel_size=kernel_size,
                                         expansion=expansion)

        mel_decoder = MelDecoder(dim=embed_dim//reduction, 
                                 kernel_size=decoder_kernel_size,
                                 n_blocks=n_blocks, 
                                 block_depth=block_depth)

        self.phoneme2mel = Phoneme2Mel(encoder=phoneme_encoder,
                                       decoder=mel_decoder)

        self.hifigan = get_hifigan(checkpoint=hifigan_checkpoint,
                                   infer_device=infer_device, verbose=verbose)

        self.ssim_loss_fn = MaskedSSIMLoss(
            kernel_size=(ssim_kernel_time, ssim_kernel_freq),
            sigma=(ssim_sigma_time, ssim_sigma_freq),
            k1=ssim_k1,
            k2=ssim_k2,
        )
        self.mel_weight = float(mel_weight)
        self.l1_weight = float(l1_weight)
        self.ssim_weight = float(ssim_weight)
        self.pitch_weight = float(pitch_weight)
        self.energy_weight = float(energy_weight)
        self.duration_weight = float(duration_weight)
        self.hparams["mel_objective"] = "1.0_l1_plus_1.0_ssim"

        self.training_step_outputs = []


    def forward(self, x):
        return self.phoneme2mel(x, train=True) if self.training else self.predict_step(x)


    def predict_step(self, batch, batch_idx=0,  dataloader_idx=0):
        mel, mel_len, duration = self.phoneme2mel(batch, train=False)
        mel_hifigan = mel.transpose(1, 2)  # (B, n_mels, T) for HiFiGAN
        wav = self.hifigan(mel_hifigan).squeeze(1)
        return wav, mel_len, mel  # mel: (B, T, n_mels)


    def loss(self, y_hat, y, x):
        pitch_pred = y_hat["pitch"]
        energy_pred = y_hat["energy"]
        duration_pred = y_hat["duration"]
        mel_pred = y_hat["mel"]

        phoneme_mask = x["phoneme_mask"]
        mel_mask = x["mel_mask"]

        pitch = x["pitch"]
        energy = x["energy"]
        duration = x["duration"]
        mel = y["mel"]

        mel_pred = mel_pred[:, :mel.shape[1], :mel.shape[2]]
        valid_mel_mask = ~mel_mask[:, :mel.shape[1]]  # [B, T], True means valid

        mel_l1_loss = F.l1_loss(
            mel_pred.masked_select(valid_mel_mask.unsqueeze(-1)),
            mel.masked_select(valid_mel_mask.unsqueeze(-1)),
        )
        ssim_loss = self.ssim_loss_fn(
            pred_mel=mel_pred,
            target_mel=mel,
            valid_mask=valid_mel_mask,
        )
        mel_loss = (
            self.l1_weight * mel_l1_loss
            + self.ssim_weight * ssim_loss
        )

        phoneme_mask = ~phoneme_mask

        pitch_pred = pitch_pred[:,:pitch.shape[-1]]
        pitch_pred = pitch_pred.squeeze(-1)
        pitch = pitch.masked_select(phoneme_mask)
        pitch_pred = pitch_pred.masked_select(phoneme_mask)
        pitch_loss = nn.MSELoss()(pitch_pred, pitch)

        energy_pred = energy_pred[:,:energy.shape[-1]]
        energy_pred = energy_pred.squeeze(-1)
        energy      = energy.masked_select(phoneme_mask)
        energy_pred = energy_pred.masked_select(phoneme_mask)
        energy_loss = nn.MSELoss()(energy_pred, energy)

        duration_pred = duration_pred[:,:duration.shape[-1]]
        duration_pred = duration_pred.squeeze(-1)
        duration      = duration.masked_select(phoneme_mask)
        duration_pred = duration_pred.masked_select(phoneme_mask)
        duration      = torch.log(duration.float() + 1)
        duration_pred = torch.log(duration_pred.float() + 1)
        duration_loss = nn.MSELoss()(duration_pred, duration)

        return (
            mel_loss,
            mel_l1_loss,
            ssim_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )
 

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        (
            mel_loss,
            mel_l1_loss,
            ssim_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        ) = self.loss(y_hat, y, x)

        loss = (
            self.mel_weight * mel_loss
            + self.pitch_weight * pitch_loss
            + self.energy_weight * energy_loss
            + self.duration_weight * duration_loss
        )
        
        losses = {"loss": loss, 
                  "mel_loss": mel_loss,
                  "mel_l1_loss": mel_l1_loss,
                  "ssim_loss": ssim_loss,
                  "pitch_loss": pitch_loss,
                  "energy_loss": energy_loss, 
                  "duration_loss": duration_loss}
        self.training_step_outputs.append(losses)
        
        return loss


    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        avg_mel_loss = torch.stack([x["mel_loss"] for x in self.training_step_outputs]).mean()
        avg_mel_l1_loss = torch.stack(
            [x["mel_l1_loss"] for x in self.training_step_outputs]
        ).mean()
        avg_ssim_loss = torch.stack([x["ssim_loss"] for x in self.training_step_outputs]).mean()
        avg_pitch_loss = torch.stack([x["pitch_loss"] for x in self.training_step_outputs]).mean()
        avg_energy_loss = torch.stack(
            [x["energy_loss"] for x in self.training_step_outputs]).mean()
        avg_duration_loss = torch.stack(
            [x["duration_loss"] for x in self.training_step_outputs]).mean()
        self.log("mel", avg_mel_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("l1", avg_mel_l1_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("ssim", avg_ssim_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("pitch", avg_pitch_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("energy", avg_energy_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("dur", avg_duration_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss", avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("lr", self.scheduler.get_last_lr()[0], on_epoch=True, prog_bar=True, sync_dist=True)
        self.training_step_outputs.clear()


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.phoneme2mel(x, train=True)
        (
            mel_loss,
            mel_l1_loss,
            ssim_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        ) = self.loss(y_hat, y, x)
        val_loss = (
            self.mel_weight * mel_loss
            + self.pitch_weight * pitch_loss
            + self.energy_weight * energy_loss
            + self.duration_weight * duration_loss
        )
        self.log_dict(
            {
                "val_loss": val_loss,
                "val_mel": mel_loss,
                "val_l1": mel_l1_loss,
                "val_ssim": ssim_loss,
                "val_pitch": pitch_loss,
                "val_energy": energy_loss,
                "val_duration": duration_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=y["mel"].shape[0],
        )

        if batch_idx==0 and self.current_epoch>=1 :
            wavs, lengths, _ = self.forward(x)
            wavs = wavs.to(torch.float).cpu().numpy()[:5]
            write_to_file(wavs, self.hparams.preprocess_config, lengths=lengths.cpu().numpy()[:5], \
                wav_path=self.hparams.wav_path, filename="prediction")

            mel = y["mel"]
            mel = mel.transpose(1, 2)
            lengths = x["mel_len"]
            with torch.no_grad():
                wavs = self.hifigan(mel).squeeze(1)
                wavs = wavs.to(torch.float).cpu().numpy()[:5]

            write_to_file(wavs, self.hparams.preprocess_config, lengths=lengths.cpu().numpy()[:5],\
                    wav_path=self.hparams.wav_path, filename="reconstruction")

            # write the text to be converted to file
            path = os.path.join(self.hparams.wav_path, "prediction.txt")
            with open(path, "w") as f:
                text = x["text"] 
                for i in range(len(text)):
                    f.write(text[i] + "\n")
            
    def on_test_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.constant_lr:
            # No warmup/decay: keep lr fixed at self.hparams.lr for the whole run.
            self.scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        else:
            self.scheduler = get_lr_scheduler(optimizer, 50, self.hparams.max_epochs, min_lr=0)

        return [optimizer], [self.scheduler]
