
from SemiSupCon.models.finetuning import FinetuneSemiSupCon
from SemiSupCon.models.semisupcon import SemiSupCon
from SemiSupCon.dataloading.datamodules import MixedDataModule
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.cli import SaveConfigCallback
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml
import os
import torch
from tqdm import tqdm

from torchmetrics.functional import auroc, average_precision
import wandb



class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.logger is not None:
            experiment_name = trainer.logger.experiment.name
            # Required for proper reproducibility
            config = self.parser.dump(self.config, skip_none=False)
            with open(self.config_filename, "r") as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                # trainer.logger.experiment.config.update(config, allow_val_change=True)
                
            if config['model']['checkpoint'] is None:
                previous_experiment_name = 'from_scratch'
            else:
                previous_experiment_name = config['model']['checkpoint'].split('/')[-2]
            if not config['test'] and config['resume_id'] is None:
                new_experiment_name = experiment_name+f'_zeroshot_{previous_experiment_name}_{config["data"]["task"]}'
            else:
                new_experiment_name = experiment_name
                
            # with open(os.path.join(os.path.join(self.config['ckpt_path'], new_experiment_name), "config.yaml"), 'w') as outfile:
            #     yaml.dump(config, outfile, default_flow_style=False)

            
            # trainer.logger.experiment.name
            
            #add a checkpoint callback that saves the model every epoch
            ## and that saves the best model based on validation loss
            
    
    



class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--log", default=False)
        parser.add_argument("--log_model", default=False)
        parser.add_argument("--ckpt_path", default="SemiSupCon-finetuning")
        parser.add_argument("--head_checkpoint", default=None)
        parser.add_argument("--resume_from_checkpoint", default=None)
        parser.add_argument("--resume_id", default=None)
        parser.add_argument("--checkpoint", default=None)
        parser.add_argument('--test', default=False)
        parser.add_argument('--prototype_p', default=0.5)

if __name__ == "__main__":
    

    cli = MyLightningCLI(model_class=SemiSupCon, datamodule_class=MixedDataModule, seed_everything_default=123,
                         run=False, save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"overwrite": True})

    cli.instantiate_classes()
    
    # get the name of the model loaded from checkpoint
    if cli.config.checkpoint is not None and cli.config.test==False:
        previous_experiment_name = cli.config.checkpoint.split('/')[-2]
    else:
        previous_experiment_name = ''

    if cli.config.log:
        logger = WandbLogger(project="SemiSupCon",id = cli.config.resume_id)
        experiment_name = logger.experiment.name+f"_zeroshot_{previous_experiment_name}_{cli.config['data']['task']}"
        ckpt_path = cli.config.ckpt_path
    else:
        logger = None

    cli.trainer.logger = logger

    try:
        if not os.path.exists(os.path.join(ckpt_path, experiment_name)):
            os.makedirs(os.path.join(ckpt_path, experiment_name))
    except:
        pass
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mdm = cli.datamodule
    mdm.setup()
    model = cli.model
    model.to(device)
    
    if cli.config.checkpoint is not None:
        state_dict = torch.load(cli.config.checkpoint)['state_dict']
        model.load_state_dict(state_dict)
    
    model.freeze()
    model.eval()
    
    
    
    
    # train_dataloader = mdm.train_dataloader()
    table = None
    
    train_dataloader = mdm.train_dataloader()
    
    for prototype_p in [0.01,0.02,0.05,0.1,0.2,0.5,0.9,1]:
    
        if prototype_p == 1:
            n = None
        else:
            print(f'Using {prototype_p} of the training data as prototypes')
            n = int(float(prototype_p) * len(mdm.train_supervised_dataset))
        # prototypes = mdm.train_supervised_dataset.get_prototypes(model, n = int(cli.config.prototype_p * len(mdm.train_supervised_dataset)))
        # shuffle train_dataloader
        prototypes = train_dataloader.get_prototypes(model, n)

        test_dataset = mdm.test_supervised_dataset

        preds = []
        labels = []
        for idx in tqdm(range(len(test_dataset))):
            data = test_dataset[idx]
            audio = data['audio'].unsqueeze(1).to(model.device)
            label = data['labels'].to(model.device)
            batch_size= audio.shape[0]

            
            
            
            
            sim = model.prototype_inference(prototypes=prototypes, wav=audio).unsqueeze(0)
            
            # print(sim.shape)
            # print(label.shape)
            
            preds.append(sim)
            labels.append(label)
            
        torchpreds = torch.cat(preds)
        torchlabels = torch.cat(labels)
        
        if mdm.task == 'mtat_top50':
            aurocs = auroc(torchpreds, torchlabels, task = 'multilabel', num_labels = 50)
            aps = average_precision(torchpreds, torchlabels, task = 'multilabel', num_labels = 50)
        
        
        # pretty print results to console
        print(f'Prototype_p: {prototype_p}')
        print(f'AUROC: {aurocs}')
        print(f'AP: {aps}')
        
        if logger is not None:
            if table is None:
                table = wandb.Table(data = [[mdm.task, prototype_p, aurocs, aps]], columns = ['task','prototype_p','auroc','ap'])
            else:
                table.add_data(*[mdm.task, prototype_p, aurocs, aps])
        # log a wandb table with [task,prototype_p, auroc, ap] with name 'zero shot results'
        
    wandb.log({'zero shot results': table})
        
    
