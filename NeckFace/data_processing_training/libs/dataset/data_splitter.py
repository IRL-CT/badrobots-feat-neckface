import random
from torch.utils.data import DataLoader
from .dataset import CNNDataset

class DataSplitter:
    train_loader: DataLoader
    val_loader: DataLoader

    def __init__(self, train_data, test_data, BATCH_SIZE, WORKER_NUM, shuffle, input_config):

        if shuffle:
            random.shuffle(train_data)
        train_data = train_data
        val_data = test_data

        print('train length', len(train_data))
        print('test length', len(val_data))
        # convert to 'Dataloader'
        if len(train_data):
            self.train_loader = DataLoader(
                CNNDataset(train_data, input_config=input_config, is_train=True),
                batch_size=BATCH_SIZE,
                shuffle=shuffle,
                num_workers=WORKER_NUM
            )
        else:
            self.train_loader = None
        self.test_loader = DataLoader(
            CNNDataset(val_data, input_config=input_config, is_train=False),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=WORKER_NUM
        )