import os
import sys
import wandb

def delete_folders(checkpoint_folders = ['Checkpoints'], wandb_projects = ['SemiSupCon']):
    runs = []
    for project_name in wandb_projects:
        runs.extend(wandb.Api().runs(f'jul-guinot/{project_name}'))

    run_names = [run.name for run in runs]
    
    for folder in checkpoint_folders:
        for subdir in os.listdir(folder):
            subdir_path = os.path.join(folder, subdir)
            if os.path.isdir(subdir_path):
                subdir_name = os.path.basename(subdir_path)
                if subdir_name not in run_names:
                    os.system(f"rm -rf {subdir_path}")
                    print(f"Removed folder: {subdir_path}")  # Pretty print removed folder
                    
 
                    

if __name__ == "__main__":
   

    delete_folders()
