"""Train the 11x5 L1 + Gaussian SSIM model from random initialization."""

import datetime
import os

import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from datamodule import LJSpeechDataModule
from layers import networks as networks_module
from model_l1_ssim import EfficientSpeech
from utils.tools import get_args


if __name__ == "__main__":
    args = get_args()
    if args.checkpoint is not None:
        raise ValueError("From-scratch training does not accept --checkpoint")
    if args.run_name is None:
        raise ValueError("--run-name is required")

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    with open(args.preprocess_config, "r", encoding="utf-8") as stream:
        preprocess_config = yaml.load(stream, Loader=yaml.FullLoader)

    datamodule = LJSpeechDataModule(
        preprocess_config=preprocess_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = EfficientSpeech(
        preprocess_config=preprocess_config,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        depth=args.depth,
        n_blocks=args.n_blocks,
        block_depth=args.block_depth,
        reduction=args.reduction,
        head=args.head,
        embed_dim=args.embed_dim,
        kernel_size=args.kernel_size,
        decoder_kernel_size=args.decoder_kernel_size,
        expansion=args.expansion,
        wav_path=args.out_folder,
        hifigan_checkpoint=args.hifigan_checkpoint,
        infer_device=args.infer_device,
        verbose=args.verbose,
        constant_lr=False,
        mel_weight=5.0,
        l1_weight=1.0,
        ssim_weight=1.0,
    )

    acoustic_params = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("phoneme2mel.") and parameter.requires_grad
    )
    print("Training from random initialization")
    print(f"  network file:       {networks_module.__file__}")
    print(f"  acoustic params:    {acoustic_params:,}")
    print(f"  SSIM kernel:        {tuple(model.ssim_loss_fn.kernel_size)}")
    print(f"  SSIM sigma:         {tuple(model.ssim_loss_fn.sigma)}")
    print(
        "  mel loss:           "
        f"{model.l1_weight:g} * L1 + {model.ssim_weight:g} * SSIM"
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

    start_time = datetime.datetime.now()
    trainer.fit(model, datamodule=datamodule)
    print(f"Training time: {datetime.datetime.now() - start_time}")
