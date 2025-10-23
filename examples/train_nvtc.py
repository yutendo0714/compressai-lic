# train_nvtc.py
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import RichProgressBar
import torch

class GlobalStepRichProgressBar(RichProgressBar):
    @property
    def total_train_batches(self):
        return self.trainer.max_steps - self.trainer.global_step


class MyLightningCLI(LightningCLI):
    def after_fit(self):
        # 学習後に自動でテストを実行
        self.trainer.test(self.model, self.datamodule)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    cli = MyLightningCLI(
        model_class=pl.LightningModule,
        datamodule_class=pl.LightningDataModule,
        subclass_mode_model=True,       # モデルをサブクラスとして扱うか
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            "callbacks": [GlobalStepRichProgressBar()],
        },
    )
