import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
import wandb
import os
import sys

# from scripts.attr_pred.attr_module_cnn import FurnitureAttributesModel
from semlayoutdiff.apm.attr_module import FurnitureAttributesModel
from semlayoutdiff.apm.furniture_data_loader import FurnitureDataset


class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


@hydra.main(version_base="1.2")
def main(cfg: DictConfig) -> None:
    if cfg.debug:
        os.environ["WANDB_MODE"] = "dryrun"
    # Initialize W&B logger
    # wandb.init(project="attr_cnn_v3", name="resume from previous checkpoint")
    wandb.init(project=cfg.project, name=f"{cfg.name}_lr={cfg.trainer.lr}_l1={cfg.trainer.l1_lambda}_l2={cfg.trainer.l2_lambda}")
    wandb_logger = WandbLogger()

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.config.update(cfg_dict)

    # Load processed front3d data
    train_dataset = FurnitureDataset(root_dir=cfg.train_data_dir, num_categories=cfg.model.num_categories,
                                     num_orientation_class=cfg.model.num_orientation_class, floor_id=cfg.model.floor_id,
                                     # transform=FurnitureDataset.transform_pair
                                     )
    val_dataset = FurnitureDataset(root_dir=cfg.val_data_dir, num_categories=cfg.model.num_categories,
                                   num_orientation_class=cfg.model.num_orientation_class, floor_id=cfg.model.floor_id,
                                   # transform=FurnitureDataset.transform_pair
                                   )

    # # Split dataset into training and validation sets
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=8)
    # train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size)
    # val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size)
    # train_dataloader = DataLoader(train_dataset, batch_size=8)
    # val_dataloader = DataLoader(val_dataset, batch_size=3)

    # train_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=custom_collate_fn)

    # Instantiate the model using the provided configuration
    model = FurnitureAttributesModel(cfg)

    # # Setup Model Checkpointing
    # checkpoint_callback = ModelCheckpoint(dirpath=cfg.checkpoint_dir,
    #                                       save_top_k=cfg.checkpoint_save_top_k,
    #                                       filename='{epoch}-{val_orient_acc:.2f}',
    #                                       monitor='val_orient_acc', mode='max')

    # Setup Model Checkpointing for validation accuracy
    checkpoint_callback_val = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        save_top_k=cfg.checkpoint_save_top_k,
        filename='val-{epoch}-{val_orient_acc:.2f}',
        monitor='val_orient_acc',
        mode='max'
    )

    # Setup Model Checkpointing for training accuracy
    checkpoint_callback_train = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        save_top_k=cfg.checkpoint_save_top_k,
        filename='train-{epoch}-{train_orient_acc:.2f}',
        monitor='train_orient_acc',
        mode='max'
    )

    # Check for existing checkpoints
    checkpoint_path = None
    if cfg.resume_training:
        checkpoint_path = cfg.checkpoint_path
        print("Loading checkpoints")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = None
            print("Checkpoint path not found, start from scratch")

    # Set up the PyTorch Lightning trainer with W&B logger
    trainer = pl.Trainer(logger=wandb_logger,
                         max_epochs=cfg.trainer.max_epochs,
                         callbacks=[checkpoint_callback_val, checkpoint_callback_train],
                         overfit_batches=0)

    # Train the model
    # trainer.fit(model, train_dataloader)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    main()
