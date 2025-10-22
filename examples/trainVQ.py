# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import random
import shutil
import sys
import os
import logging
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import image_models
from torch.utils.tensorboard import SummaryWriter


def setup_logging(save_dir):
    log_file = os.path.join(save_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


class VQRateDistortionLoss(nn.Module):
    """VQ系モデル向けの簡易RD損失
       - out に rd_loss があればそれを優先
       - なければ total = rate_weight*rate + lmbda*mse + vq_weight*vq_loss
       - rate が無ければ 0 とみなす（likelihoods があっても確率モデルに依らない設計なので使わない）
    """
    def __init__(self, lmbda=1e-2, rate_weight=1.0, vq_weight=1.0):
        super().__init__()
        self.lmbda = lmbda
        self.rate_weight = rate_weight
        self.vq_weight = vq_weight

    def forward(self, out, target):
        losses = {}

        # x_hat は必須
        x_hat = out.get("x_hat", None)
        if x_hat is None:
            raise ValueError("Model output must contain 'x_hat'.")

        mse = F.mse_loss(x_hat, target)
        losses["mse_loss"] = mse

        # 優先：モデルが rd_loss を出すならそれを使う
        if "rd_loss" in out and isinstance(out["rd_loss"], torch.Tensor):
            total = out["rd_loss"]
            # 可能なら補助ログも拾う
            rate = out.get("rate", torch.zeros_like(mse))
            vq_loss = out.get("vq_loss", torch.zeros_like(mse))
            losses["bpp_loss"] = rate
            losses["vq_loss"] = vq_loss
            losses["loss"] = total
            return losses

        # それ以外：rate / vq_loss を可能な限り拾って合成
        rate = out.get("rate", None)
        if rate is None:
            rate = torch.zeros_like(mse)
        vq_loss = out.get("vq_loss", None)
        if vq_loss is None:
            vq_loss = torch.zeros_like(mse)

        total = self.rate_weight * rate + self.lmbda * mse + self.vq_weight * vq_loss

        losses["bpp_loss"] = rate
        losses["vq_loss"] = vq_loss
        losses["loss"] = total
        return losses


def configure_optimizers_vq(net, args):
    # メイン最適化
    base_params = [p for p in net.parameters() if p.requires_grad]
    if len(base_params) == 0:
        raise ValueError("No trainable parameters in the model.")

    optimizer = optim.Adam(base_params, lr=args.learning_rate)

    # 補助（存在する場合のみ）
    aux_params = []
    if hasattr(net, "aux_parameters") and callable(getattr(net, "aux_parameters")):
        aux_params = list(net.aux_parameters())

    aux_optimizer = None
    if len(aux_params) > 0:
        aux_optimizer = optim.Adam(aux_params, lr=args.aux_learning_rate)

    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer,
    epoch, clip_max_norm, logger, writer=None
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(out_net, d)

        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        # 補助損（存在する場合のみ）
        aux_loss_tensor = torch.tensor(0.0, device=device)
        if aux_optimizer is not None and hasattr(model, "aux_loss"):
            aux_loss = model.aux_loss()
            if isinstance(aux_loss, torch.Tensor):
                aux_loss_tensor = aux_loss
                aux_loss_tensor.backward()
                aux_optimizer.step()

        if i % 10 == 0:
            msg = (
                f"Train epoch {epoch}: [{i*len(d)}/{len(train_dataloader.dataset)} "
                f"({100. * i / len(train_dataloader):.0f}%)] "
                f"Loss: {out_criterion['loss'].item():.4f} | "
                f"MSE: {out_criterion['mse_loss'].item():.4f} | "
                f"Rate: {out_criterion.get('bpp_loss', torch.tensor(0.)).item():.4f} | "
                f"VQ: {out_criterion.get('vq_loss', torch.tensor(0.)).item():.4f} | "
                f"Aux: {aux_loss_tensor.item():.4f}"
            )
            logger.info(msg)

            if writer:
                step = epoch * len(train_dataloader) + i
                writer.add_scalar("train/total_loss", out_criterion["loss"].item(), step)
                writer.add_scalar("train/mse_loss", out_criterion["mse_loss"].item(), step)
                writer.add_scalar("train/rate", out_criterion.get("bpp_loss", torch.tensor(0.)).item(), step)
                writer.add_scalar("train/vq_loss", out_criterion.get("vq_loss", torch.tensor(0.)).item(), step)
                writer.add_scalar("train/aux_loss", aux_loss_tensor.item(), step)


def test_epoch(epoch, test_dataloader, model, criterion, logger, writer=None):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    rate_loss = AverageMeter()
    mse_loss = AverageMeter()
    vq_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            # モデルが aux_loss を持つ場合のみ記録（勾配なし）
            aux_val = 0.0
            if hasattr(model, "aux_loss"):
                aux = model.aux_loss()
                if isinstance(aux, torch.Tensor):
                    aux_val = aux.item()

            loss.update(out_criterion["loss"], d.size(0))
            mse_loss.update(out_criterion["mse_loss"], d.size(0))
            rate_loss.update(out_criterion.get("bpp_loss", 0.0), d.size(0))
            vq_loss.update(out_criterion.get("vq_loss", 0.0), d.size(0))
            aux_loss.update(aux_val, d.size(0))

    logger.info(
        f"Test epoch {epoch}: "
        f"Loss: {loss.avg:.4f} | "
        f"MSE: {mse_loss.avg:.4f} | "
        f"Rate: {rate_loss.avg:.4f} | "
        f"VQ: {vq_loss.avg:.4f} | "
        f"Aux: {aux_loss.avg:.4f}"
    )

    if writer:
        writer.add_scalar("test/loss", loss.avg, epoch)
        writer.add_scalar("test/mse_loss", mse_loss.avg, epoch)
        writer.add_scalar("test/rate", rate_loss.avg, epoch)
        writer.add_scalar("test/vq_loss", vq_loss.avg, epoch)
        writer.add_scalar("test/aux_loss", aux_loss.avg, epoch)

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="VQ compression training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="nvtc",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Main learning rate")
    parser.add_argument("--aux-learning-rate", type=float, default=1e-3, help="Auxiliary learning rate (if any)")
    parser.add_argument("-n", "--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--quality", type=int, default=3, help="Quality index for zoo config (if supported)")
    parser.add_argument("--lambda", dest="lmbda", type=float, default=1e-2, help="Distortion weight")
    parser.add_argument("--rate-weight", type=float, default=1.0, help="Rate weight for VQ models")
    parser.add_argument("--vq-weight", type=float, default=1.0, help="VQ regularization weight")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--test-batch-size", type=int, default=64, help="Test batch size")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256), help="Crop size (H W)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--save", action="store_true", default=True, help="Save checkpoints")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--clip_max_norm", default=1.0, type=float, help="Grad clipping max norm")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--save_dir", type=str, default="./runs/")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    # --- output directory ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.model}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # --- logger & tensorboard setup ---
    logger = setup_logging(save_dir)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))

    logger.info("==== Training Configuration ====")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # zoo からモデル生成（NVTC など）
    # quality は cfgs にある場合のみ使用される
    net_ctor = image_models[args.model]
    # どのモデルでも受けやすいように kwargs 経由に
    try:
        net = net_ctor(quality=args.quality)
    except TypeError:
        # quality を受けないモデルに備えて退避
        net = net_ctor()

    net = net.to(device)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, maybe_aux_optimizer = configure_optimizers_vq(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    criterion = VQRateDistortionLoss(
        lmbda=args.lmbda, rate_weight=args.rate_weight, vq_weight=args.vq_weight
    )

    last_epoch = 0
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint.get("epoch", -1) + 1
        if "state_dict" in checkpoint:
            net.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if maybe_aux_optimizer is not None and "aux_optimizer" in checkpoint:
            maybe_aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        logger.info(f"=== Epoch {epoch}/{args.epochs} - LR: {optimizer.param_groups[0]['lr']:.6f} ===")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            maybe_aux_optimizer,
            epoch,
            args.clip_max_norm,
            logger,
            writer
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, logger, writer)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            ckpt = {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }
            if maybe_aux_optimizer is not None:
                ckpt["aux_optimizer"] = maybe_aux_optimizer.state_dict()
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch}_checkpoint.pth.tar")
            save_checkpoint(ckpt, is_best, filename=ckpt_path)

    writer.close()
    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main(sys.argv[1:])
