#!/usr/bin/env python
import tensorflow as tf, numpy as np, gym, sys, copy, argparse
import random
import time
import matplotlib.pyplot as plt
import os
from sklearn import random_projection
from datetime import datetime

class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name, sess, nS, nA, batch_size=32):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.  
		
		global train_op, W, output, features_, act, labels, loss, merged, writer, weights, loss_weights

		features_ = tf.placeholder(dtype=tf.float32, shape=[None, nS], name='features_')
		features = features_

		act = tf.placeholder(dtype = tf.int32, shape=[None], name='act')
		labels = tf.placeholder(dtype=tf.float32, shape=[None, nA], name='labels')
		loss_weights = tf.placeholder(dtype=tf.float32, shape=[None, nA], name='loss_weights')
		
		# Input layer
		input_layer = features

		# Dense Layers
		# dense = tf.layers.dense(inputs = input_layer, units = nS, activation = None, name = 'dense')
		dense1 = tf.layers.dense(inputs = input_layer, units = 64, activation = None, name = 'dense1', use_bias=True)

		####### Uncomment for DQN #######
		# Note: When loading models for Dueling DQN, change layer names to value_dense2, value_dense3

		# dense2 = tf.layers.dense(inputs = dense1, units = 32, activation = tf.nn.relu, name='dense2', use_bias=True)
		# dense3 = tf.layers.dense(inputs = dense2, units = 16, activation = tf.nn.relu, name='dense3', use_bias=True)

		####### Uncomment for Dueling Networks #######

		# dense4 = tf.layers.dense(inputs = dense1, units = 32, activation = tf.nn.relu, name='adv_dense4', use_bias=True)
		# dense5 = tf.layers.dense(inputs = dense4, units = 32, activation = tf.nn.relu, name='adv_dense5', use_bias=True)
		# dense4 = tf.layers.dense(inputs=dense3, units=256, activation=tf.nn.relu, name='dense4', use_bias=True)

		# Output Layer
		# output_v = tf.layers.dense(inputs=dense3, units=nA, name='output_v')
		# output_a = tf.layers.dense(inputs=dense5, units=nA, name='output_a')
		# avg_a = tf.reduce_mean(input_tensor=output_a)
		# unbiased_a = tf.subtract(output_a, avg_a)
		# output = tf.add(output_v, unbiased_a, name='output')

		#############################################

		output = tf.layers.dense(inputs=dense1, units=nA, name='output')

		loss = tf.losses.mean_squared_error(labels=labels, predictions=output, weights=loss_weights)

		optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

		train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())

		self.saver = tf.train.Saver(max_to_keep=1000000)
		
		tf.summary.scalar('loss', loss)
		writer = tf.summary.FileWriter("logs", sess.graph)	# After every episode
		merged = tf.summary.merge_all()

		return

	def save_model_weights(self, sess, epi_no):
		# Helper function to save your model / weights. 

		checkpoint_path = os.path.join('./save/LinearMC/', 'model.ckpt')
		self.saver.save(sess, checkpoint_path, global_step=epi_no)
		print("model saved to {}".format(checkpoint_path))

		return

	def load_model(self, sess, model_file):
		# Helper function to load an existing model.m 

		ckpt = tf.train.get_checkpoint_state('./important_save/LinearMC/')
		if ckpt and ckpt.model_checkpoint_path:
			print("Loading model: ", ckpt.all_model_checkpoint_paths[int(model_file/500)])
			self.saver.restore(sess, ckpt.all_model_checkpoint_paths[int(model_file/500)])
			# for v in tf.global_variables():
			# 	print(v.name)
		else:
			print("Error in loading model: ", ckpt.model_checkpoint_path)

		return

class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
		
		self.memory = []
		self.memory_size = memory_size
		self.burn_in = burn_in

		return

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.

		samples = random.sample(range(1,len(self.memory)), 32)
		sample_batch = [self.memory[s] for s in samples]

		return sample_batch

	def append(self, transition):
		# Appends transition to the memory. 

		self.memory.append(transition)

		return

	def get_memory(self):
		return self.memory

	def pop_(self):
		self.memory.pop(0)
		return

class DQN_Agent():

	def __init__(self, environment_name, sess, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
		
		self.env = environment_name
		self.render = render

		if self.env == "CartPole-v0":
			self.nS = 4	 # For CartPole-v0
			self.nA = 2
			self.gamma = 0.99

		if self.env == "MountainCar-v0":
			self.nS = 2	 # For MountainCar-v0
			self.nA = 3
			self.gamma = 1.0
		
		self.net = QNetwork(self.env, sess, self.nS, self.nA)

		self.burn_size = 30000
		self.memory_size = 50000
		self.replay_memory = Replay_Memory(self.memory_size, self.burn_size)

		self.max_iterations = 200
		self.max_episodes = 5001
		self.epsilon = 0.1

		self.updateWeightIter = 100  # Not really used any more, here for legacy reasons

		self.alpha = 0.0001

		self.plot = False

		return

	def epsilon_greedy_policy(self, q_values, epi_number):
		# Creating epsilon greedy probabilities to sample from.

		prob = np.random.random_sample()  # Float in the range [0,1)
		eps = self.epsilon
		num_actions = self.nA 

		if epi_number < self.max_episodes/2:
			eps = eps/((epi_number/(self.max_episodes/10)) + 1)
		else:
		 	eps = 0

		# eps = eps/((epi_number/(self.max_episodes/10)) + 1)

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


		# TODO: Set this value to True if using experience replay
		exp_replay = False
		totalUpdates = 0
		numUpdates = 0

		sess.run(tf.global_variables_initializer())
		global train_op, W, output, features, act, labels, features_, loss, writer, merged, weights, loss_weights

		env = self.env
		wIter = 0
		updateWeightIter = self.updateWeightIter
		gamma = self.gamma
		alpha = self.alpha
		steps_per_episode = []
		qFunc_per_episode = []
		avg_reward_per_episode = []
		
		# Burn in memory
		if exp_replay:
			self.burn_in_memory(sess)
		
		############################### LOAD MODEL ###########################

		# self.net.load_model(sess, 0)

		######################################################################

		for epi_no in range(self.max_episodes):
			# print('Episode Number: %d' % epi_no)
			
			if self.plot:
				numUpdates += 1
				if numUpdates == 21:
					plt.figure()
					plt.plot(reward_per_episode)
					save_path = './plots/LinearMC/'+str(epi_no)+'.png'
					print("Plot saved to {}".format(save_path))
					plt.savefig(save_path)
					print('Average reward')
					avg_reward = np.sum(reward_per_episode)/20
					avg_reward_per_episode.append(avg_reward)
					print(avg_reward)
					self.plot = False
					numUpdates = 0

				else:
					reward_per_episode.append(-iter_no)  # For MountainCar
					# reward_per_episode.append(iter_no)  # For CartPole

			total_qFuncCurrent = 0
			
			# Random start-action pair right
			currentState = env.reset()  # S
			currentAction = env.action_space.sample() # A
			xCurrent = currentState
			
			nextState, reward, isTerminal, debugInfo = env.step(currentAction)				
			xNext = nextState  # A' , generate feature space from nextState

			for iter_no in range(self.max_iterations):
				# print('Iteration Number: %d' % iter_no)

				totalUpdates += 1

				if totalUpdates % 10000 == 0:
					numUpdates = 0
					self.plot = True
					reward_per_episode = []

				# if isTerminal:
				if nextState[0] >= 0.5:
					target = reward
					currentAction = np.reshape(currentAction, (-1))
					target_ = np.zeros((1, self.nA))
					target_[0, currentAction] = target
					loss_weights_ = np.zeros((self.nA, 1))
					loss_weights_[currentAction,0] = 1.0
					xCurrent = np.reshape(xCurrent, (-1,self.nS))
					loss_weights_ = np.reshape(loss_weights_, (-1,self.nA))
					xCurrent = np.reshape(xCurrent, (-1,self.nS))

					if exp_replay:
						qFuncCurrent= sess.run([output],feed_dict={features_: xCurrent})

					else:
						_, qFuncCurrent, loss_, summary = sess.run([train_op, output, loss, merged], feed_dict={features_:xCurrent, act:currentAction, labels:target_, loss_weights:loss_weights_})

					qFuncCurrent = np.reshape(qFuncCurrent, (-1, self.nA))
					total_qFuncCurrent = total_qFuncCurrent + qFuncCurrent[0][currentAction[0]]
					print("episode: {}/{}, score: {}".format(epi_no, self.max_episodes, iter_no))
					steps_per_episode.append(iter_no)
					qFunc_per_episode.append(total_qFuncCurrent)
					break
				elif isTerminal:
					print("episode: {}/{}, score: {}".format(epi_no, self.max_episodes, iter_no))
					steps_per_episode.append(iter_no)
					qFunc_per_episode.append(total_qFuncCurrent)
					break

				xNext = np.reshape(xNext, (-1, self.nS))
				qFuncOld = sess.run(output, feed_dict={features_: xNext})
				nextAction = self.epsilon_greedy_policy(qFuncOld, epi_no)
				act_qFuncOld = qFuncOld[0, nextAction]  # max(Q(S', A', w-))

				currentAction = np.reshape(currentAction, (-1))
				target = reward + gamma * act_qFuncOld # r + gamma*Q(S', A', w-)
				target_ = np.zeros((1,self.nA))
				target_[0, currentAction] = target
				loss_weights_ = np.zeros((self.nA, 1))
				loss_weights_[currentAction, 0] = 1.0
				xCurrent = np.reshape(xCurrent,(-1,self.nS))
				target = np.reshape(target, (1,1))
				loss_weights_ = np.reshape(loss_weights_, (-1,self.nA))

				if exp_replay:
					qFuncCurrent = sess.run([output], feed_dict={features_: xCurrent})

				else:
					_, qFuncCurrent, loss_, summary = sess.run([train_op, output, loss, merged],feed_dict={features_: xCurrent, act: currentAction,labels: target_, loss_weights: loss_weights_})

				qFuncCurrent = np.reshape(qFuncCurrent, (-1, self.nA))
				total_qFuncCurrent = total_qFuncCurrent + qFuncCurrent[0][currentAction[0]]
				xCurrent = xNext
				currentAction = nextAction
				nextState, reward, isTerminal, debugInfo = env.step(nextAction)				
				xNext = nextState # generate feature space from new nextState

				# Perform updates in the case of experience replay
				if exp_replay:
					batches = 32
					batch_list = self.replay_memory.sample_batch(batches)
					r_targ = np.zeros((batches, self.nA))
					a_batch = np.zeros((batches, 1))
					x_batch = np.zeros((batches, self.nS))
					loss_weights_ = np.zeros((batches, self.nA))

					for batch in range(batches):
						term_batch = batch_list[batch][4]
						a_batch[batch] = batch_list[batch][1]
						action = int(a_batch[batch])
						x_batch[batch] = batch_list[batch][0]
						loss_weights_[batch][action] = 1
						xp_batch = batch_list[batch][3]
						xp_batch = np.reshape(xp_batch, (-1, self.nS))

						if (term_batch):
							r_targ[batch][action] = batch_list[batch][2]

						else:
							qFuncBatch = sess.run(output, feed_dict={features_: xp_batch})
							a_prime = self.epsilon_greedy_policy(qFuncBatch, epi_no)
							a_prime_feed = np.reshape(a_prime, (1))
							out_prime = sess.run(output, feed_dict={features_: xp_batch, act: a_prime_feed})
							r_targ[batch][action] = batch_list[batch][2] + gamma * qFuncBatch[0, out_prime]
						
					r_targ = np.reshape(r_targ, (-1, self.nA))

					a_batch = np.reshape(a_batch, (-1))
					x_batch = np.reshape(x_batch, (-1, self.nS))
					_, qFuncCurrent, loss_, _ = sess.run([train_op, output, loss, merged], feed_dict={features_: x_batch, act: a_batch, labels: r_targ, loss_weights: loss_weights_})

				# Appending to replay memory
				if len(self.replay_memory.get_memory()) == self.memory_size:
					self.replay_memory.pop_()

				self.replay_memory.append(transition)

				################## Holding target weights constant #################

				# if wIter >= updateWeightIter:
				# 	wOld = wCurrent
				# 	wIter = 0
				
				# wIter += 1

				####################################################################

				if self.render:
					env.render()

			################## Saving models ##################

			if epi_no % 500 == 0:
				self.net.save_model_weights(sess, epi_no)

			###################################################
		
		writer.close()

		plt.figure(1)
		plt.plot(steps_per_episode)
		save_path = './plots/LinearMC/final_steps_per_episode.png'
		print("Plot saved to {}".format(save_path))
		plt.savefig(save_path)
		
		plt.figure(2)
		plt.plot(qFunc_per_episode)

		save_path = './plots/LinearMC/final_steps.png'
		print("Plot saved to {}".format(save_path))
		plt.savefig(save_path)

		return

	def demo(self, sess, model_file=None):
		# Run loaded model weights twenty times with greedy policy
		# Render once
		# Save average reward

		sess.run(tf.global_variables_initializer())
		global train_op, W, output, features, act, labels, features_, loss, writer, merged, weights, loss_weights

		env = self.env
		wIter = 0
		updateWeightIter = self.updateWeightIter
		gamma = self.gamma
		alpha = self.alpha
		rewardSum = 0

		############################### LOAD MODEL ###########################

		self.net.load_model(sess, 8500)

		######################################################################

		for epi_no in range(20):
			total_qFuncCurrent = 0
			steps_per_episode = []

			# Random start-action pair right
			currentState = env.reset()  # S
			currentAction = env.action_space.sample()  # A
			xCurrent = currentState

			nextState, reward, isTerminal, debugInfo = env.step(currentAction)
			xNext = nextState  # A' , generate feature space from nextState

			for iter_no in range(self.max_iterations):
				# print('Iteration Number: %d' % iter_no)

				if isTerminal:
					target = reward
					xCurrent = np.reshape(xCurrent, (-1, self.nS))
					target_ = np.zeros((1, self.nA))
					target_[0, currentAction] = target
					currentAction = np.reshape(currentAction, (-1))
					loss_weights_ = np.zeros((self.nA, 1))
					loss_weights_[currentAction, 0] = 1.0
					loss_weights_ = np.reshape(loss_weights_, (-1, self.nA))
					qFuncCurrent = sess.run([output], feed_dict={features_: xCurrent})
					qFuncCurrent = np.reshape(qFuncCurrent, (-1, self.nA))
					total_qFuncCurrent = total_qFuncCurrent + qFuncCurrent[0][currentAction[0]]
					print("episode: {}/{}, score: {}".format(epi_no, 100, iter_no))
					break

				xNext = np.reshape(xNext, (-1, self.nS))
				qFuncOld = sess.run(output, feed_dict={features_: xNext})
				nextAction = self.greedy_policy(qFuncOld)
				act_qFuncOld = qFuncOld[0, nextAction]  # max(Q(S', A', w-))

				currentAction = np.reshape(currentAction, (-1))
				target = reward + gamma * act_qFuncOld  # r + gamma*Q(S', A', w-)
				xCurrent = np.reshape(xCurrent, (-1, self.nS))
				loss_weights_ = np.zeros((self.nA, 1))
				loss_weights_[currentAction, 0] = 1.0
				loss_weights_ = np.reshape(loss_weights_, (-1, self.nA))

				target_ = np.zeros((1, self.nA))
				target_[0, currentAction] = target
				qFuncCurrent = sess.run([output], feed_dict={features_: xCurrent})
				qFuncCurrent = np.reshape(qFuncCurrent, (-1, self.nA))
				total_qFuncCurrent = total_qFuncCurrent + qFuncCurrent[0][currentAction[0]]
				xCurrent = xNext
				currentAction = nextAction
				nextState, reward, isTerminal, debugInfo = env.step(nextAction)
				xNext = nextState  # generate feature space from new nextState

				# if epi_no == 19:
				# 	# input("Ready_to_render")
				# 	env.render()
				# 	if iter_no == 0:
				# 		time.sleep(2)
				# 	time.sleep(0.05)

			steps_per_episode.append(iter_no)

			if self.nS == 2:  # Mountain Car
				rewardSum += -iter_no
			else:
				rewardSum += iter_no

		avgReward = rewardSum / 20
		print("Average reward is: {}".format(avgReward))

	def test(self, sess, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		
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

		self.net.load_model(sess, 0)

		######################################################################

		for epi_no in range(100):
			# print('Episode Number: %d' % epi_no)
			total_qFuncCurrent = 0
			
			# Random start-action pair right
			currentState = env.reset()	# S
			currentAction = env.action_space.sample() # A	
			xCurrent = currentState
			
			nextState, reward, isTerminal, debugInfo = env.step(currentAction)				
			xNext = nextState # A' , generate feature space from nextState

			for iter_no in range(self.max_iterations):
				# print('Iteration Number: %d' % iter_no)
				
				if isTerminal:
					target = reward
					xCurrent = np.reshape(xCurrent, (-1,self.nS))
					target_ = np.zeros((1, self.nA))
					target_[0, currentAction] = target
					currentAction = np.reshape(currentAction, (-1))
					loss_weights_ = np.zeros((self.nA, 1))
					loss_weights_[currentAction, 0] = 1.0
					loss_weights_ = np.reshape(loss_weights_, (-1, self.nA))
					qFuncCurrent = sess.run([output], feed_dict={features_:xCurrent})
					qFuncCurrent = np.reshape(qFuncCurrent, (-1, self.nA))
					total_qFuncCurrent = total_qFuncCurrent + qFuncCurrent[0][currentAction[0]]
					# print('Q per episode: %f' % total_qFuncCurrent)
					print("episode: {}/{}, score: {}".format(epi_no, 100, iter_no))
					steps_per_episode.append(iter_no)
					qFunc_per_episode.append(total_qFuncCurrent)
					break

				xNext = np.reshape(xNext, (-1,self.nS))
				qFuncOld = sess.run(output, feed_dict={features_:xNext})
				nextAction = self.greedy_policy(qFuncOld)
				act_qFuncOld = qFuncOld[0, nextAction]	# max(Q(S', A', w-))

				currentAction = np.reshape(currentAction, (-1))
				target = reward + gamma * act_qFuncOld # r + gamma*Q(S', A', w-)
				xCurrent = np.reshape(xCurrent, (-1, self.nS))
				loss_weights_ = np.zeros((self.nA, 1))
				loss_weights_[currentAction, 0] = 1.0
				loss_weights_ = np.reshape(loss_weights_, (-1, self.nA))

				target_ = np.zeros((1, self.nA))
				target_[0, currentAction] = target
				qFuncCurrent = sess.run([output], feed_dict={features_:xCurrent})
				qFuncCurrent = np.reshape(qFuncCurrent, (-1, self.nA))
				total_qFuncCurrent = total_qFuncCurrent + qFuncCurrent[0][currentAction[0]]
				xCurrent = xNext
				currentAction = nextAction
				nextState, reward, isTerminal, debugInfo = env.step(nextAction)				
				xNext = nextState # generate feature space from new nextState

				if self.render:
					env.render()

			################## Saving models ##################

			# if epi_no % 500 == 0:
			# 	self.net.save_model_weights(sess, epi_no)

			###################################################
		
		writer.close()

		print(np.sum(steps_per_episode)/100 + 1.0)
		print(np.std(steps_per_episode))

		plt.figure(1)
		plt.plot(steps_per_episode)
		
		plt.figure(2)
		plt.plot(qFunc_per_episode)

		plt.show()

		return

	def burn_in_memory(self, sess):
		# Initialize your replay memory with a burn_in number of episodes / transitions. 
		env = self.env
		
		xCurrent = env.reset()  # S
		currentAction = env.action_space.sample()  # A
		xNext, reward, isTerminal, _ = env.step(currentAction)

		for i in range(self.burn_size):
			xNext = np.reshape(xNext, (-1,self.nS))
			qFunc = sess.run(output, feed_dict={features_:xNext})
			nextAction = self.epsilon_greedy_policy(qFunc, 0)  # Want to maximize exploration, don't decay epsilon

			transition = [xCurrent, currentAction, reward, xNext, isTerminal]
			self.replay_memory.append(transition)

			if isTerminal:
				xCurrent = env.reset()
				currentAction = env.action_space.sample()
				xNext, reward, isTerminal, _ = env.step(currentAction)

			else:
				xCurrent = xNext
				xNext, reward, isTerminal, _ = env.step(nextAction)	
		print('Completed burning in memory')
		return

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
	print('final_duel branch')
	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)
	env = gym.make(environment_name)

	# Some global variables required for feed_dict
	train_op = tf.placeholder(dtype = tf.float32)
	output = tf.placeholder(dtype = tf.float32)
	W = tf.placeholder(dtype = tf.float32)
	loss = tf.placeholder(dtype = tf.float32)
	features_ = tf.placeholder(dtype = tf.float32, shape = [4,1])	# CartPole-v0
	# features_ = tf.placeholder(dtype = tf.float32, shape = [2,1]) 	# MountainCar-v0
	
	agent = DQN_Agent(env, sess, render)
	agent.train(sess)
	# agent.test(sess)
	# agent.demo(sess)
	writer.close()

if __name__ == '__main__':
	main(sys.argv)
