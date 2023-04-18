import argparse
import os
from datetime import datetime
from pprint import pprint

import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping
from pytorch_lightning.core.saving import load_hparams_from_yaml
from pytorch_lightning.loggers import TensorBoardLogger
from rich.console import Console

from src.datasets import NAryDataset
from src.models import BERT


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hp", help="path of the hparam file",default="hparams/bert.yaml")
    parser.add_argument("--exp", help="name of the experiments", choices=("bert",), default="bert")

    parser.add_argument("--do_evaluate", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--do_debug", action="store_true")

    parser.add_argument("--ckpt", help="path of the checkpoint", default=None)
    return parser.parse_args()


def main(args):
    console = Console()
    # noinspection PyTypeChecker
    hp: DictConfig = load_hparams_from_yaml(args.hp)
    hp.seed = seed_everything(hp.seed)

    console.log(f"Seed is set to [bold green]{hp.seed}[/bold green].")

    # ----------------------------------
    # 1. INIT LIGHTNING MODEL AND DATA
    # ----------------------------------
    console.log("Preparing [bold green]dataset[/bold green]...")
    data = NAryDataset(
        hp.data_dirpath,
        hp.max_seq_length,
        hp.train_batch_size,
        hp.eval_batch_size,
        force_not_shuffle=args.do_debug
    )
    console.log("Building [bold green]model[/bold green]...")
    model = BERT(hp)

    # ----------------------------------
    # 2. INIT EARLY STOPPING
    #在训练过程中，神经网络中的weights会更新，以使模型在训练数据上的表现更好。
    # 一段时间以来，训练集上的改进与测试集上的改进呈正相关。
    # 但是，有时会开始过度拟合训练数据，进一步的“改进”将导致泛化性能降低。这称为过度拟合。
    # Earlystopping是一种用于在过度拟合发生之前终止训练的技术。
    # 关键要点是使用tf.keras.EarlyStopping回调。
    # 通过监视某个值（例如，验证准确性）在最近一段时间内是否有所改善（由patience参数控制）来触发提前停止。
    # ----------------------------------
    early_stop_callback = EarlyStopping(
        monitor=hp.monitor,
        min_delta=1e-5,
        patience=hp.patience,
        verbose=True,
        mode=hp.metric_mode,
    )

    # ----------------------------------
    # 3. INIT LOGGERS
    # 输出结果到Tensorboard
    # 方便使用tensorboard查看
    # ----------------------------------
    version_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logger = TensorBoardLogger(
        save_dir="experiments",
        name=args.exp,
        version=version_name
    )

    # ----------------------------------
    # 4. INIT MODEL CHECKPOINT CALLBACK
    # 作用是以一定的频率保存keras模型或参数，通常是和model.compile()、model.fit()结合使用的，
    # 可以在训练过程中保存模型，也可以再加载出来训练一般的模型接着训练。
    # 具体的讲，可以理解为在每一个epoch训练完成后，可以根据参数指定保存一个效果最好的模型。
    # ----------------------------------
    # Model Checkpoint Callback
    ckpt_dir = os.path.join("experiments", "ckpt", args.exp, version_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{step:d}-{train_f1:.4f}-{val_f1:.4f}",
        save_top_k=hp.save_top_k,
        verbose=True,
        monitor=hp.monitor,
    )

    #print(model)

    # ----------------------------------
    # 5. INIT TRAINER
    # ----------------------------------
    trainer = Trainer(
        logger=tb_logger if (not args.do_evaluate and not args.do_predict) else False,
        enable_checkpointing=True,
        gradient_clip_val=getattr(hp, "gradient_clip_val", None),
        deterministic=True,
        check_val_every_n_epoch=1,
        fast_dev_run=args.do_debug,
        accumulate_grad_batches=getattr(hp, "accumulate_grad_batches", None),
        max_epochs=getattr(hp, "max_epochs", None),
        min_epochs=getattr(hp, "min_epochs", None),
        max_steps=getattr(hp, "max_steps", -1),
        val_check_interval=getattr(hp, "val_check_interval", None),
        callbacks=[
            cb for cb in [RichProgressBar(leave=False), early_stop_callback, checkpoint_callback]
            if cb is not None
        ]
    )

    if args.do_evaluate:
        metrics = trainer.test(model, data, ckpt_path=args.ckpt_path, verbose=True)
        pprint(metrics[0])
        return

    if args.do_predict:
        results = trainer.predict(
            model, dataloaders=[data.test_dataloader()],
            ckpt_path=args.ckpt, return_predictions=True
        )
        pred_dir = os.path.dirname(args.ckpt_path)
        for name, result in zip(("pred_test.pkl",), results):
            torch.save(result, os.path.join(pred_dir, name))



    # ----------------------------------
    # 6. START TRAINING
    # ----------------------------------
    trainer.fit(model, data)


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
