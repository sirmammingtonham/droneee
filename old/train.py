from architecture import PGNetwork       # Deep Learning library
import numpy as np            # Handle matrices
import airsim
import random                 # Handling random number generation
import time                   # Handling time calculation

from collections import deque # Ordered collection with ends

### ENVIRONMENT HYPERPARAMETERS
state_size = [3,3,4] # Our input is a stack of 4 frames hence 100x160x4 (Width, height, channels)
action_size = 4 # 4 possible actions: roll, pitch, yaw, throttle
stack_size = 4 # Defines how many frames are stacked together

## TRAINING HYPERPARAMETERS
learning_rate = 0.0001
num_epochs = 1000  # Total epochs for training
max_episodes = 300

batch_size = 10000 # Each 1 is a timestep (NOT AN EPISODE) # YOU CAN CHANGE TO 5000 if you have GPU
gamma = 0.99 # Discounting rate

def initialize_environment(arm=True):
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    if arm:
        client.armDisarm(True)

    return client

def stack_states(stacked_states, state, is_new_episode):
    # Preprocess frame
    input = np.zeros((3,3), dtype=np.float64)
    input[0] = [state['kinematics_estimated']['position']['x_val'], state['kinematics_estimated']['position']['y_val'], state['kinematics_estimated']['position']['z_val']]
    input[1] = [state['kinematics_estimated']['linear_acceleration']['x_val'], state['kinematics_estimated']['linear_acceleration']['y_val'], state['kinematics_estimated']['linear_acceleration']['z_val']]
    input[2] = [state['kinematics_estimated']['linear_velocity']['x_val'], state['kinematics_estimated']['linear_velocity']['y_val'], state['kinematics_estimated']['linear_velocity']['z_val']]
    if is_new_episode:
        # Clear our stacked_frames
        stacked_states = deque([np.zeros((3, 3), dtype=np.float64) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_states.append(input)
        stacked_states.append(input)
        stacked_states.append(input)
        stacked_states.append(input)

        # Stack the frames
        stacked_state = np.stack(stacked_states, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_states.append(input)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_states, axis=2)

    return stacked_state, stacked_states


def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards



if __name__ == '__main__':
    # Reset the graph
    allRewards = []
    total_rewards = 0
    maximumRewardRecorded = 0
    episode = 0
    episode_states, episode_actions, episode_rewards = [], [], []

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for episode in range(max_episodes):

            episode_rewards_sum = 0

            # Launch the game
            state = env.reset()

            env.render()

            while True:

                # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
                action_probability_distribution = sess.run(action_distribution,
                                                           feed_dict={input_: state.reshape([1, 4])})

                action = np.random.choice(range(action_probability_distribution.shape[1]),
                                          p=action_probability_distribution.ravel())  # select action w.r.t the actions prob

                # Perform a
                new_state, reward, done, info = env.step(action)

                # Store s, a, r
                episode_states.append(state)

                # For actions because we output only one (the index) we need 2 (1 is for the action taken)
                # We need [0., 1.] (if we take right) not just the index
                action_ = np.zeros(action_size)
                action_[action] = 1

                episode_actions.append(action_)

                episode_rewards.append(reward)
                if done:
                    # Calculate sum reward
                    episode_rewards_sum = np.sum(episode_rewards)

                    allRewards.append(episode_rewards_sum)

                    total_rewards = np.sum(allRewards)

                    # Mean reward
                    mean_reward = np.divide(total_rewards, episode + 1)

                    maximumRewardRecorded = np.amax(allRewards)

                    print("==========================================")
                    print("Episode: ", episode)
                    print("Reward: ", episode_rewards_sum)
                    print("Mean Reward", mean_reward)
                    print("Max reward so far: ", maximumRewardRecorded)

                    # Calculate discounted reward
                    discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)

                    # Feedforward, gradient and backpropagation
                    loss_, _ = sess.run([loss, train_opt], feed_dict={input_: np.vstack(np.array(episode_states)),
                                                                      actions: np.vstack(np.array(episode_actions)),
                                                                      discounted_episode_rewards_: discounted_episode_rewards
                                                                      })

                    # Write TF Summaries
                    summary = sess.run(write_op, feed_dict={input_: np.vstack(np.array(episode_states)),
                                                            actions: np.vstack(np.array(episode_actions)),
                                                            discounted_episode_rewards_: discounted_episode_rewards,
                                                            mean_reward_: mean_reward
                                                            })

                    writer.add_summary(summary, episode)
                    writer.flush()

                    # Reset the transition stores
                    episode_states, episode_actions, episode_rewards = [], [], []

                    break

                state = new_state

            # Save Model
            if episode % 100 == 0:
                saver.save(sess, "./models/model.ckpt")
                print("Model saved")
