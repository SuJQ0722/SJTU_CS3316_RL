# utils/logger.py
import wandb
import os
from torch.utils.tensorboard import SummaryWriter

os.environ["WANDB_MODE"] = "offline"

class Logger:
    def __init__(
        self,
        log_dir: str,
        project_name: str,
        run_name: str,
        config: dict,
        use_tensorboard: bool = True,
        use_wandb: bool = False
    ):
        self.writer = None
        if use_tensorboard:
            if SummaryWriter is None:
                raise ImportError("TensorBoard not available. Please install it with 'pip install tensorboard'")
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
            print(f"TensorBoard logging to: {os.path.join(log_dir, 'tb')}")

        self.wandb_run = None
        if use_wandb:
            if wandb is None:
                raise ImportError("WandB not available. Please install it with 'pip install wandb'")
            self.wandb_run = wandb.init(
                project=project_name,
                name=run_name,
                config=config,
                sync_tensorboard=True,
                reinit=True
            )
            print("WandB logging enabled.")
            
        if not use_tensorboard and not use_wandb:
            print("Warning: No logger is enabled (TensorBoard and WandB are both disabled).")
    
    def log(self, data, step: int):
        """log data to wandb or tensorboard
        Args:
            data: Can be either a dict of {tag: value} pairs or a single tag string
            step: Current step number
        """
        if isinstance(data, dict):
            # Handle dictionary input
            for tag, value in data.items():
                if self.writer:
                    self.writer.add_scalar(tag, value, step)
            if self.wandb_run:
                self.wandb_run.log(data, step=step)
        else:
            pass
        
    def finish(self):
        """Finish logging and close resources."""
        if self.writer:
            self.writer.close()
        if self.wandb_run:
            self.wandb_run.finish()