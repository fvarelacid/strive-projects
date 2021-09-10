#Define a train and validation function
import data_handler as dh
import models as mod

X_train, X_test, y_train, y_test = dh.build_dataset("data/insurance.csv", 8)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

for epochs in range(2):
    X_train_batch, X_test_batch, y_train_batch, y_test_batch = dh.to_batches(X_train, X_test, y_train, y_test, 8)

    mod.MLP.forward(X_train_batch)
    

