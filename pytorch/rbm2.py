# Restricted Boltzmann Machine

# Importing the libraries
import numpy as np
import pandas as pd
import csv

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python',encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python',encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python',encoding = 'latin-1')

# Preparing the training set and test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,1][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

#Converting the data into torch tensor
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#Converting the ratings into binary ratings 1(Liked) or 0(Not Liked)
training_set[training_set == 0] == -1
training_set[training_set == 1] == 0
training_set[training_set == 2] == 0
training_set[training_set >= 3] == 1
test_set[test_set == 0] == -1
test_set[test_set == 1] == 0
test_set[test_set == 2] == 0
test_set[test_set >= 3] == 1    

#Converting the architecture of the Neural Networks
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - 100, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range (10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM    
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))

prediction = rbm.predict(vk)
print(prediction)

with open('output2.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(prediction)
    
csv_file.close()

#Converting torch tensor to array
prediction = torch.Tensor.numpy(prediction)

#Deleting 1st Column
prediction = np.delete(prediction,np.s_[0:1], axis=1)

#movies title
movie_title = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python',encoding = 'latin-1')
title = movie_title.iloc[0:1681, 1].values
print (title)

#Converting to dataframe
title = pd.DataFrame(title)
title = np.reshape(title, 1681)
pred_result = pd.DataFrame(prediction)

#Converting to numpy array to make row as column and again coverting to dataframe
title=title.values
#array to 1D
title=title.reshape((1, -1))
title = pd.DataFrame(title)

#Merging title and pred_result
frame = [title,pred_result]
result = pd.concat(frame)
#result = title.append(pred_result)

#Merging UserID with Result
user_id=users.iloc[0:943, 0].values
user_id=pd.DataFrame(user_id)
result=pd.concat([user_id, result], axis=1)

#result.iloc[[0], [0]] = 0
#result.at[0, 0] = 'User-id'

result.to_csv('output1.csv', encoding='utf-8')



