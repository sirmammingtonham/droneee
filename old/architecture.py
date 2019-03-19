import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class PGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 2)
        self.fc3 = nn.Linear(2, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=-1)
        return x


class NNet():
    def __init__(self):
        self.nnet = PGN().cuda()

    def train(self, states, actions, discounted_episode_rewards):
        optimizer = optim.Adam(self.nnet.parameters(), lr=learning_rate)
        self.nnet.train()
        states = torch.FloatTensor(states).contiguous().cuda()
        actions = torch.LongTensor(actions).contiguous().cuda()
        discounted_episode_rewards = torch.FloatTensor(discounted_episode_rewards).contiguous().cuda()

        states, actions = Variable(states), Variable(actions)  ### !!! I WAS BREAKING THE COMPTUATION GRAPH
        pred = torch.exp(self.nnet(states))  ### DON'T WRAP PREDICTIONS IN A VARIABLE
        #       #Variable(torch.max(actions.long(), 1)[1]),

        loss = self.loss(pred, actions, discounted_episode_rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #         a = list(self.nnet.parameters())[0].clone()
        #         loss.backward()
        #         optimizer.step()
        #         b = list(self.nnet.parameters())[0].clone()
        #         print(list(self.nnet.parameters())[0].grad)
        return loss

    def predict(self, state):
        state = torch.FloatTensor(state).contiguous().cuda()
        with torch.no_grad():
            state = Variable(state)
        self.nnet.eval()
        x = self.nnet(state)
        return torch.exp(x).data.cpu().numpy()

    def loss(self, pred, actions, discounted_episode_rewards):
        loss = F.cross_entropy(pred, actions, reduce=False)
        mean_loss = torch.mean(loss * discounted_episode_rewards)
        return mean_loss




import tensorflow as tf

class PGNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='PGNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.name_scope("inputs"):
            input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
            actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
            discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards")

            # Add this placeholder for having this variable in tensorboard
            mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")

            with tf.name_scope("fc1"):
                fc1 = tf.contrib.layers.fully_connected(inputs=input_,
                                                        num_outputs=10,
                                                        activation_fn=tf.nn.relu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("fc2"):
                fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                                        num_outputs=action_size,
                                                        activation_fn=tf.nn.relu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("fc3"):
                fc3 = tf.contrib.layers.fully_connected(inputs=fc2,
                                                        num_outputs=action_size,
                                                        activation_fn=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("softmax"):
                action_distribution = tf.nn.softmax(fc3)

            with tf.name_scope("loss"):
                # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
                # If you have single-class labels, where an object can only belong to one class, you might now consider using
                # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.
                neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=actions)
                loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_)

            with tf.name_scope("train"):
                train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)