import numpy as np
import pandas as pd


class DataGenerator2(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """

    def __init__(self, ):
        """
        Args:
            num_samples_per_class: num samples to generate per task
            batch_size: size of meta batch size (e.g. number of tasks)
        """
        self.df_train = pd.read_csv('model_def/data_train.csv')
        self.df_test = pd.read_csv('model_def/data_test.csv')

    def generate_train(self):
        datas = [self.df_train[self.df_train['task_id'] == i] for i in range(64)]
        datas_x = [np.array(data['X']) for data in datas]
        datas_x = np.expand_dims(np.stack(datas_x), axis=2)

        datas_y = [np.array(data['y']) for data in datas]
        datas_y = np.expand_dims(np.stack(datas_y), axis=2)
        while True:
            yield datas_x, datas_y

    def generate_test(self):
        datas = [self.df_test[self.df_test['task_id'] == i] for i in range(1)]
        datas = datas*64
        datas_x = [np.array(data['X']) for data in datas]
        datas_x = np.expand_dims(np.stack(datas_x), axis=2)

        datas_y = [np.array(data['y']) for data in datas]
        datas_y = np.expand_dims(np.stack(datas_y), axis=2)
        while True:
            yield datas_x, datas_y