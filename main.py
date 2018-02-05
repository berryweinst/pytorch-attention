from sklearn.model_selection import train_test_split
from models import *
import gc

def Volatile(x):
    return Variable(x, volatile=True)


batch_size = 8
total_minibatches = Data.create_minibatches('resnet_v1_50', 'block3', batch_size)

X_train, X_test, y_train, y_test = train_test_split([i[0] for i in total_minibatches], [i[1] for i in total_minibatches], test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15)



# net = Fetures2ECoGTrans(features_dim=X_test[0].shape[3], hidden_dim=y_test[0].shape[1])
net = Fetures2ECoGTrans(features_dim=X_test[0].shape[3], hidden_dim=int(X_test[0].shape[3]/8))
opt = torch.optim.RMSprop(net.parameters(), lr=0.0005)
mse = torch.nn.MSELoss()

n_train = len(y_train)
epoch = 0
max_epochs = 50
min_loss = np.inf
n_valid = len(y_valid)
while epoch < max_epochs:
    sum_loss = 0
    for idx, t in enumerate(X_train):
        net.zero_grad()
        w, output = net(Variable(t))
        loss = mse(output, Variable(y_train[idx]))
        loss.backward()
        opt.step()
        sum_loss += loss.data[0]
    epoch += 1
    print('[{:2d}] {:5.10f}'.format(epoch, sum_loss / n_train))
    sum_loss = 0
    for idx, t in enumerate(X_valid):
        w, output = net(Volatile(t))
        loss = mse(output, Volatile(y_valid[idx]))
        sum_loss += loss.data[0]

    valid_loss = sum_loss / n_valid
    print('valid loss: {:5.10f}'.format(valid_loss))
    if (min_loss > valid_loss):
        min_loss = valid_loss
    else:
        break




n_valid = len(y_test)
sum_loss = 0
# per_elec_target = np.empty((0, y_test[0].shape[1]))
# per_elec_output = np.empty((0, y_test[0].shape[1]))
per_elec_target = []
per_elec_output = []
for s in y_test:
    per_elec_target.extend(list(s))
    # per_elec_target = np.append(per_elec_target, np.array(s), axis=0)

for idx, t in enumerate(X_test):
    w, output = net(Volatile(t))
    per_elec_output.extend(list(output.data))
    loss = mse(output, Volatile(y_test[idx]))
    sum_loss += loss.data[0]


print('valid loss: {:5.10f}'.format(sum_loss / n_valid))
gc.collect()

print ("Pearson correleation (r, p)")
print (stats.pearsonr(per_elec_target, per_elec_output))

print("Visual electrode 100 samples. Target in blue. Net output in green")
plt.plot(per_elec_target[0:100])
plt.plot(per_elec_output[0:100])
plt.show()
