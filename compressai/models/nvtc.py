from compressai.models import CompressionModel
from compressai.registry import register_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .modules.nvtc.model.nvtc import NVTC  # あなたの Lightning 用 NVTC 実装を流用

@register_model("nvtc")
class NVTCCompressAI(nn.Module):
    # def __init__(self, lmbda=512, **kwargs):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Lightningの部分を除いた NVTC core を初期化
        # self.nvtc = NVTC(lmbda=lmbda, **kwargs)
        self.nvtc = NVTC(*args, **kwargs)

    def forward(self, x):
        result = self.nvtc(x)
        # CompressAI標準フォーマットに整形
        return {
            "x_hat": result["x_hat"],
            "likelihoods": {
                # CompressAIでは確率値を扱うので exp(-rate*log(2)) に変換
                "stage_all": torch.exp(-result["rate"] * math.log(2))
            },
            "loss": result["loss"],
            "rd_loss": result["rd_loss"],
            "vq_loss": result["vq_loss"],
        }

    def compress(self, x):
        # NVTCのpre_padding, quantizer.compressを利用してbitstream生成
        x, shape = self.nvtc.pre_padding(x)
        bitstreams = []
        for s in range(self.nvtc.n_stage):
            for l in range(self.nvtc.n_layer[s]):
                quant = self.nvtc.quantizer[s][l]
                bitstreams.append(
                    quant.compress(x, None, self.nvtc.lmbda)[1]  # string 部分を取得
                )
        return {"strings": bitstreams, "shape": shape}

    def decompress(self, strings, shape):
        # 符号列リストをdecodeして再構成画像を返す
        x_hat = None
        idx = 0
        for s in reversed(range(self.nvtc.n_stage)):
            for l in reversed(range(self.nvtc.n_layer[s])):
                quant = self.nvtc.quantizer[s][l]
                x_rec = quant.decompress(strings[idx], vq_shape=None)
                x_hat = x_rec if x_hat is None else x_hat + x_rec
                idx += 1
        x_hat = self.nvtc.post_cropping(x_hat, shape)
        return {"x_hat": x_hat}
