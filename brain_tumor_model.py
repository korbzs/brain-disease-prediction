from dataclasses import dataclass
import pandas as pd

@dataclass
class Brain_Model():
    train_dataset: pd.DataFrame
    test_dataset: pd.DataFrame
    method: str
    hyperparams: list[dict[str, str]]
    metrics: list[str]      #in more depth like auc, roc, conf_mx  
    accuracy: float = 0.9798712310
    
    def __str__(self) -> str:
        return (f"Brain Model Summary:\n"
                f"--------------------\n"
                f"Method: {self.method}\n"
                f"Train Dataset Shape: {self.train_dataset.shape}\n"
                f"Test Dataset Shape: {self.test_dataset.shape}\n"
                f"Hyperparameters: {self.hyperparams}\n"
                f"Accuracy: {self.accuracy}\n"
                f"Metrics: {', '.join(self.metrics)}\n")