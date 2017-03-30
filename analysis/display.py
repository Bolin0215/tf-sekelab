import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # dir = os.path.join('..', 'out', '10_100_56_0.9_adamax_0.002')
    dir = os.path.join('..', 'out')
    out_file = '30_100_100_1.0_adamax'
    # out_file = 'out'
    filename = os.path.join(dir, out_file)
    train_acc, val_acc = read_result(filename)
    plot(train_acc, val_acc)

def read_result(filename):
    train_accs = []
    val_accs = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('epoch'):
            parts = line.split(',')
            train_acc_text = parts[3].strip()
            val_acc_text = parts[4].strip()
            train_acc = train_acc_text.split('=')[1].strip()
            val_acc = val_acc_text.split('=')[1].strip()
            train_accs.append(float(train_acc))
            val_accs.append(float(val_acc))
    return train_accs, val_accs

def plot(train_acc, val_acc):
    assert len(train_acc) == len(val_acc), 'lists to be shown are not equal!'

    df = pd.DataFrame({'train_acc':np.array(train_acc, dtype='float32'),
                       'val_acc':np.array(val_acc, dtype='float32')},
                      index=list(range(1,len(train_acc)+1)))
    plt.show(df.plot(title='no-attention'))


if __name__ == "__main__":
    main()