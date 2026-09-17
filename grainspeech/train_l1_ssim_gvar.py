"""Train or resume the 11x5 SSIM + GVar GrainSpeech model."""

import datetime
import os
from pathlib import Path

import torch
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from datamodule import LJSpeechDataModule
from layers import networks as networks_module
from model_l1_ssim_gvar import GrainSpeech
from utils.tools import get_args


if __name__ == "__main__":
    args = get_args()
    if args.run_name is None:
        raise ValueError("--run-name is required")
    if args.checkpoint is not None and not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {args.checkpoint}")

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    with open(args.preprocess_config, "r", encoding="utf-8") as stream:
        preprocess_config = yaml.load(stream, Loader=yaml.FullLoader)

    datamodule = LJSpeechDataModule(
        preprocess_config=preprocess_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = GrainSpeech(
        preprocess_config=preprocess_config,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        wav_path=args.out_folder,
        hifigan_checkpoint=args.hifigan_checkpoint,
        infer_device=args.infer_device,
        verbose=args.verbose,
        constant_lr=False,
        mel_weight=5.0,
        l1_weight=1.0,
        ssim_weight=1.0,
        gvar_weight=0.5,
    )

    acoustic_params = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("phoneme2mel.") and parameter.requires_grad
    )
    if args.checkpoint is None:
        print("Training from random initialization")
    else:
        print(f"Resuming full training state from: {args.checkpoint}")
    print(f"  network file:       {networks_module.__file__}")
    print(f"  acoustic params:    {acoustic_params:,}")
    print(f"  SSIM kernel:        {tuple(model.ssim_loss_fn.kernel_size)}")
    print(f"  SSIM sigma:         {tuple(model.ssim_loss_fn.sigma)}")
    print(f"  GVar kernel:        {tuple(model.gvar_loss_fn.kernel_size)}")
    print(
        "  mel loss:           "
        f"{model.l1_weight:g} * L1 + {model.ssim_weight:g} * SSIM "
        f"+ {model.gvar_weight:g} * GVar"
    )
    print(f"  outer mel weight:   {model.mel_weight:g}")
    print(f"  mel objective:      {model.hparams['mel_objective']}")
    print(f"  max epochs:         {args.max_epochs}")

    logger = TensorBoardLogger(
        save_dir=".", name="lightning_logs", version=args.run_name
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="{epoch}-{step}",
        every_n_epochs=10,
        save_top_k=1,
        save_last=True,
    )
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        check_val_every_n_epoch=10,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    if args.compile:
        print("Compiling model with torch.compile")
        model = torch.compile(model)

    start_time = datetime.datetime.now()
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.checkpoint)
    print(f"Training time: {datetime.datetime.now() - start_time}")
