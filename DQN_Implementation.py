#!/usr/bin/env python
import tensorflow as tf, numpy as np, gym, sys, copy, argparse # , keras
import random
import time
import matplotlib.pyplot as plt
import os

class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name, sess, nS, nA):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.  
		
		# tf.reset_default_graph()
		global train_op, W, output, features_, act, labels, loss, merged, writer, weights, loss_weights

		features_ = tf.placeholder(dtype = tf.float32, shape = [nS,1])
		features = tf.reshape(features_,[1,nS])
		
		act = tf.placeholder(dtype = tf.int32)
		labels = tf.placeholder(dtype = tf.float32, shape = [1, nA])
		loss_weights = tf.placeholder(dtype = tf.float32, shape = [1, nA])
		
		# W = tf.Variable(tf.random_uniform([nS,nA], 0, 0.01))
		# output = tf.matmul(features, W)

		####### Model #######
		# TODO: CHECK THE MODEL SOMETHING PROBABLY WRONG!

		# Input layer
		input_layer = features # tf.reshape(features_, [-1, 1])

		# Dense Layer
		# dense = tf.layers.dense(inputs = input_layer, units = nS, activation = None, name = 'dense')
		dense1 = tf.layers.dense(inputs = input_layer, units = 32, activation = tf.nn.relu, name = 'dense1', use_bias=True)
		dense2 = tf.layers.dense(inputs = dense1, units = 32, activation = tf.nn.relu, name='dense2', use_bias=True)
		dense3 = tf.layers.dense(inputs = dense2, units = 32, activation = tf.nn.relu, name='dense3', use_bias=True)
		dense4 = tf.layers.dense(inputs=dense3, units=256, activation=tf.nn.relu, name='dense4', use_bias=True)

		# Output Layer
		output = tf.layers.dense(inputs = dense4, units = nA, name = 'output')
		
		#####################

		# predict = output[0, act]
		# predict_ = tf.reshape(predict, [1,1])
		# labels_ = output
		# labels_[0, act] = labels
		loss = tf.losses.mean_squared_error(labels=labels, predictions=output, weights=loss_weights)

		# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001)
		optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)

		train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
		
		# TODO: NOT SURE IF THIS NEEDS TO BE EXPLICITLY SAVED OR NOT
		self.saver = tf.train.Saver()
		
		tf.summary.scalar('loss', loss)
		# tf.summary.scalar('output', output)
		writer = tf.summary.FileWriter("logs", sess.graph)	# After every episode
		merged = tf.summary.merge_all()

		return

	# def model(self, features, labels):
		
		
	# 	return train_op

	# def forward(self, features, labels):

	# 	with tf.Session() as sess:
	# 		sess.run(tf.initialize_all_variables())
	# 	pass

	def save_model_weights(self, sess, epi_no):
		# Helper function to save your model / weights. 

		# save_path = saver.save(sess, "../save/model.ckpt")
		checkpoint_path = os.path.join('./save/', 'model.ckpt')
		self.saver.save(sess, checkpoint_path, global_step=epi_no)
		print("model saved to {}".format(checkpoint_path))

		return

	def load_model(self, sess, model_file):
		# Helper function to load an existing model.m 

		ckpt = tf.train.get_checkpoint_state('./save/')

		if ckpt and ckpt.model_checkpoint_path:
			print("Loading model: ", ckpt.all_model_checkpoint_paths[int(model_file/500) - 1])
			self.saver.restore(sess, ckpt.all_model_checkpoint_paths[int(model_file/500) - 1])
			# for v in tf.global_variables():
			# 	print(v.name)
		else:
			print("Error in loading model: ", ckpt.model_checkpoint_path)

		return

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights. 
		pass

class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
		pass

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		pass

	def append(self, transition):
		# Appends transition to the memory. 	
		pass

class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, sess, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
		
		self.env = environment_name
		self.render = render

		# self.nS = 4	# For CartPole-v0
		# self.nA = 2
		# self.gamma = 0.99

		self.nS = 2	# For MountainCar-v0
		self.nA = 3
		self.gamma = 1.0
		
		self.net = QNetwork(self.env, sess, self.nS, self.nA)
		self.replay_memory = Replay_Memory()

		self.max_iterations = 200
		self.max_episodes = 5000
		self.epsilon = 0.5 

		self.updateWeightIter = 100 # Another random number for now

		self.alpha = 0.0001

		return

	def epsilon_greedy_policy(self, q_values, epi_number):
		# Creating epsilon greedy probabilities to sample from.
		# epi_number: Episode number

		prob = np.random.random_sample()	# Float in the range [0,1)
		eps = self.epsilon
		num_actions = self.nA 

		eps = eps/((epi_number/(self.max_episodes/10)) + 1)
		
		nextAction = np.argmax(q_values)			

		if prob < eps/num_actions:
			while True:
				nextAction_ = self.env.action_space.sample()
				if nextAction_ != nextAction:
					return nextAction_

		return nextAction

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		
		nextAction = np.argmax(q_values)			

		return nextAction 

	def train(self, sess):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.

		# tf.reset_default_graph()

		sess.run(tf.global_variables_initializer())
		global train_op, W, output, features, act, labels, features_, loss, writer, merged, weights, loss_weights

		env = self.env
		wIter = 0
		updateWeightIter = self.updateWeightIter
		gamma = self.gamma
		alpha = self.alpha
		steps_per_episode = []
		qFunc_per_episode = []

		############################### LOAD MODEL ###########################

		self.net.load_model(sess, 2500)

		######################################################################

		for epi_no in range(self.max_episodes):
			print('Episode Number: %d' % epi_no)
			total_qFuncCurrent = 0
			
			# Random start-action pair right
			currentState = env.reset()	# S
			currentAction = env.action_space.sample() # A	
			xCurrent = currentState
			
			nextState, reward, isTerminal, debugInfo = env.step(currentAction)				
			xNext = nextState # A' , generate feature space from nextState

			for iter_no in range(self.max_iterations):
				print('Iteration Number: %d' % iter_no)
				
				if isTerminal:
					target = reward
					target_ = np.zeros((self.nA, 1))
					target_[currentAction,0] = target
					loss_weights_ = np.zeros((self.nA, 1))
					loss_weights_[currentAction,0] = 1.0
					xCurrent = np.reshape(xCurrent, (self.nS,1))
					target_ = np.reshape(target_, (1,self.nA))
					loss_weights_ = np.reshape(loss_weights_, (1,self.nA))
					# xCurrent = np.reshape(xCurrent, (self.nS,1))
					# target = np.reshape(target, (1,1))
					# _, wCurrent, act_qFuncCurrent, loss_ = sess.run([train_op, W, output, loss], feed_dict={features_:xCurrent, act:currentAction, labels:target})
					_, qFuncCurrent, loss_, summary = sess.run([train_op, output, loss, merged], feed_dict={features_:xCurrent, act:currentAction, labels:target_, loss_weights:loss_weights_})
					total_qFuncCurrent = total_qFuncCurrent + qFuncCurrent[0, currentAction]
					# print('Q per episode: %f' % total_qFuncCurrent)
					print('******* EPISODE TERMINATED *******')
					steps_per_episode.append(iter_no)
					qFunc_per_episode.append(total_qFuncCurrent)
					break

				# qFuncOld = np.matmul(xNext, wOld) # Q(S', A', w-) # forward pass of the net with current weights with nextState, nextAction
				xNext = np.reshape(xNext, (self.nS,1))
				qFuncOld = sess.run(output, feed_dict={features_:xNext})
				nextAction = self.epsilon_greedy_policy(qFuncOld, epi_no)
				act_qFuncOld = qFuncOld[0, nextAction]	# max(Q(S', A', w-))

				# qFuncCurrent = np.matmul(xCurrent, wCurrent)  # forward pass of the net with old weights with new nextState, new nextAction
				# act_qFuncCurrent = qFuncCurrent[currentAction] # Q(S, A, w)

				target = reward + gamma * act_qFuncOld # r + gamma*Q(S', A', w-)
				target_ = np.zeros((self.nA, 1))
				target_[currentAction,0] = target
				loss_weights_ = np.zeros((self.nA, 1))
				loss_weights_[currentAction,0] = 1.0
				xCurrent = np.reshape(xCurrent, (self.nS,1))
				target_ = np.reshape(target_, (1,self.nA))
				loss_weights_ = np.reshape(loss_weights_, (1,self.nA))
				# _, wCurrent, act_qFuncCurrent, loss_ = sess.run([train_op, W, output, loss], feed_dict={features_:xCurrent, act:currentAction, labels:target})
				_, qFuncCurrent, loss_, summary, = sess.run([train_op, output, loss, merged], feed_dict={features_:xCurrent, act:currentAction, labels:target_, loss_weights:loss_weights_})
				
				# with tf.variable_scope("foo", reuse = tf.AUTO_REUSE):
				# 	weights = tf.get_variable("output/kernel:0", [2,3])
				# print(tf.trainable_variables())
				# weights = [v for v in tf.trainable_variables() if v.name == "output/kernel:0"]
				total_qFuncCurrent = total_qFuncCurrent + qFuncCurrent[0, currentAction]
				# print('Loss: %f' % loss_)
				xCurrent = xNext
				currentAction = nextAction
				nextState, reward, isTerminal, debugInfo = env.step(nextAction)				
				xNext = nextState # generate feature space from new nextState

				################## Holding target weights constant #################

				# if wIter >= updateWeightIter:
				# 	wOld = wCurrent
				# 	wIter = 0

				####################################################################

				wIter += 1

				if self.render:
					env.render()

			################## Saving models ##################

			if epi_no % 500 == 0:
				self.net.save_model_weights(sess, epi_no)

			###################################################
		
			writer.add_summary(summary, iter_no + (epi_no * self.max_iterations))
		
		writer.close()

		plt.figure(1)
		plt.plot(steps_per_episode)
		
		plt.figure(2)
		plt.plot(qFunc_per_episode)
		
		plt.show()
		return

	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		pass

	def burn_in_memory():
		# Initialize your replay memory with a burn_in number of episodes / transitions. 

		pass

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env
	render = False # args.render

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)
	# Setting this as the default tensorflow session. 
	# keras.backend.tensorflow_backend.set_session(sess)
	# You want to create an instance of the DQN_Agent class here, and then train / test it. 
	env = gym.make(environment_name)

	# initial_state = gym.reset()
	# gamma = 0.9
	train_op = tf.placeholder(dtype = tf.float32)
	output = tf.placeholder(dtype = tf.float32)
	W = tf.placeholder(dtype = tf.float32)
	loss = tf.placeholder(dtype = tf.float32)
	# features_ = tf.placeholder(dtype = tf.float32, shape = [4,1])	# CartPole-v0
	features_ = tf.placeholder(dtype = tf.float32, shape = [2,1]) 	# MountainCar-v0
	# saver = tf.train.Saver(tf.global_variables())

	# W = tf.Variable(tf.random_uniform([4,2], 0, 0.01))
	
	agent = DQN_Agent(env, sess, render)
	agent.train(sess)
	writer.close()

if __name__ == '__main__':
	main(sys.argv)

