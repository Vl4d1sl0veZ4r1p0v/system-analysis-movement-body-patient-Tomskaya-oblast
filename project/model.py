from sklearn.externals import joblib
from scipy.stats import skew, kurtosis
import pickle
import os
import pandas as pd
import numpy as np

class Model(object):
    def __init__(self, filename, scaler=None):
        format_file = filename[filename.rfind('.')+1:]
        self.has_scaler = False
        self.model = joblib.load(filename)
        if scaler is not None:
            self.has_scaler = True
            self.scaler = pickle.load(open(scaler, 'rb'))

    def predict(self, file):
        sample = pd.read_csv(file)
        if sample.shape[1] < 6:
            sample.to_csv('temp.csv', index=False)
            table = pd.read_csv('temp.csv', sep=";", names=["time", "parts", "crd1", "crd2", "crd3"])
            new_table = pd.DataFrame(columns=["time"])
            new_line = {'time': table['time'][0]}
            for i in range(table.shape[0]):
                line = table.loc[i]
                if line[0] == new_line['time']:
                    new_line[line[1] + '_1'] = line[2]
                    new_line[line[1] + '_2'] = line[3]
                    new_line[line[1] + '_3'] = line[4]
                else:
                    new_table = new_table.append(new_line, ignore_index=True)
                    new_line = {'time': line[0]}
            new_table = new_table.drop(columns={"Hip_center_1", "Hip_center_2", "Hip_center_3"})
            new_table = new_table.fillna(method='ffill')
            sample = new_table[2:]
            os.remove('temp.csv')
        true_columns = sample.columns.tolist()[1:]
        sample = gen_features(sample[true_columns].values)
        if self.has_scaler:
            sample = self.scaler.transform(sample)
        pred = self.model.predict(sample.flatten().reshape(1, -1))
        return pred[0]

def gen_features(X):
    strain = []
    strain.append(np.mean(X, axis=0))
    strain.append(np.std(X, axis=0))
    strain.append(np.min(X, axis=0))
    strain.append(np.max(X, axis=0))
    strain.append(kurtosis(X))
    strain.append(skew(X))
    strain.append(np.percentile(X, 0.25, axis=0))
    strain.append(np.percentile(X, 0.5, axis=0))
    strain.append(np.percentile(X, 0.75, axis=0))
    return np.array(strain)
