
from SemiSupCon.models.semisupcon import SemiSupCon
from SemiSupCon.dataloading.datamodules import MixedDataModule
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.cli import SaveConfigCallback
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
import os

class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.logger is not None:
            experiment_name = trainer.logger.experiment.name
            # Required for proper reproducibility
            config = self.parser.dump(self.config, skip_none=False)
            with open(self.config_filename, "r") as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                trainer.logger.experiment.config.update(config, allow_val_change=True)
            with open(os.path.join(os.path.join(self.config['ckpt_path'], experiment_name), "config.yaml"), 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False)
                
            #instanciate a ModelCheckpoint saving the model every epoch
            
            recent_callback = ModelCheckpoint(
                dirpath=os.path.join(self.config['ckpt_path'], experiment_name),
                filename='checkpoint-epoch{epoch}',  # This means all checkpoints are saved, not just the top k
                every_n_epochs=100,  # Replace with your desired value
                save_top_k = -1
            )
            callbacks = [recent_callback]
            
            
            trainer.callbacks = trainer.callbacks[:-1]+callbacks


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--log", default=False)
        parser.add_argument("--log_model", default=False)
        parser.add_argument("--ckpt_path", default="checkpoints")
        parser.add_argument("--resume_id", default=None)
        parser.add_argument("--resume_from_checkpoint", default=None)


if __name__ == "__main__":

    cli = MyLightningCLI(model_class=SemiSupCon, datamodule_class=MixedDataModule, seed_everything_default=123,
                         run=False, save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"overwrite": True},)
    
    cli.instantiate_classes()

    if cli.config.log:
        logger = WandbLogger(project="SemiSupCon", id=cli.config.resume_id)

        experiment_name = logger.experiment.name
        ckpt_path = cli.config.ckpt_path
    else:
        logger = None

    cli.trainer.logger = logger

    try:
        if not os.path.exists(os.path.join(ckpt_path, experiment_name)):
            os.makedirs(os.path.join(ckpt_path, experiment_name))
    except:
        pass

    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.resume_from_checkpoint)
