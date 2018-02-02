from sklearn.model_selection import train_test_split

from models import *

def Volatile(x):
    return Variable(x, volatile=True)


batch_size = 8
total_minibatches = Data.create_minibatches('resnet_v1_50', 'block3', batch_size)

X_train, X_test, y_train, y_test = train_test_split([i[1] for i in total_minibatches], [i[0] for i in total_minibatches], test_size=0.2)


net = Fetures2ECoGTrans(features_dim=X_test[0].shape[3], hidden_dim=y_test[0].shape[1])
opt = torch.optim.RMSprop(net.parameters(), lr=0.001)
mse = torch.nn.MSELoss()

n_train = len(y_train)
epoch = 0
max_epochs = 15
while epoch < max_epochs:
    sum_loss = 0
    for idx, t in enumerate(X_train):
        net.zero_grad()
        output = net(Variable(t))
        loss = mse(output, Variable(y_train[idx]))
        loss.backward()
        opt.step()
        sum_loss += loss.data[0]
    epoch += 1
    print('[{:2d}] {:5.3f}'.format(epoch, sum_loss / n_train))



n_valid = len(y_test)
sum_loss = 0
for idx, t in enumerate(X_test):
    output = net(Volatile(t))
    loss = mse(output, Volatile(y_test[idx]))
    sum_loss += loss.data[0]


print('valid loss: {:5.3f}'.format(sum_loss / n_valid))
