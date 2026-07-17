import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Hyperparameters
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Create multi-class data

X, y = make_blobs(  # type:ignore
    n_samples=1000,
    n_features=NUM_FEATURES,
    centers=NUM_CLASSES,
    cluster_std=1.5,
    random_state=RANDOM_SEED,
)

# 2. Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.long)


# 3. Split data into train and test sets
def random_split():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test


# X_train, X_test, y_train, y_test = random_split()


# plt.figure(figsize=(10, 7))
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()
