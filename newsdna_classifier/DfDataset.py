import pandas as pd
from torch.utils.data import Dataset


class DfDataset(Dataset):
    def __init__(self, fin, sep='\t'):
        df = pd.read_csv(fin, sep=sep)
        df = df[['id', 'label', 'text']]
        df = df.astype({'id': int})

        self._df = df
        self.num_entries = len(self._df.index)

    def __getitem__(self, idx):
        return self._df.iloc[idx].to_dict()

    def __len__(self):
        return self.num_entries
