DQN on Atari Breakout

This is a from scratch implementation of DeepMind's DQN algorithm trained on Atari Breakout.
Trained for 2 million steps on a Google Colab T4 GPU, gets around 140-190 per game which is
well above human average of 30.

The agent learned entirely on its own from raw pixels with no hardcoded rules or game knowledge.
It figured out the tunnel strategy on its own where it breaks through to the top row and lets
the ball bounce around clearing the whole board automatically.


how it works

The agent sees the last 4 grayscale game frames stacked together and passes them through a CNN
to get Q-values for each action. It picks the action with the highest Q-value, which is basically
asking "what action looks most profitable from here". Early in training it acts randomly to explore,
then gradually shifts to trusting what it has learned.

The key tricks that make it work are a replay buffer which stores past experience so the network
can learn from random batches instead of sequential frames, and a target network which is a frozen
copy of the main network that only updates every 10k steps so training stays stable.


files

dqn.py - training code, run this to train from scratch
watch.py - loads a checkpoint and records the agent playing
best_model.pt - pretrained weights after 2 million steps


how to run

install dependencies

pip install gymnasium[atari] ale-py moviepy torch numpy

watch the pretrained agent play

just run the watch.py file, it will load best_model.pt and save a video of 3 episodes to a videos folder

train from scratch

run dqn.py, it will save checkpoints every 100k steps to a checkpoints folder and a final
best_model.pt when done. takes about 45 mins on a GPU, several hours on CPU.


results

step 80k     around 1
step 300k    around 3
step 600k    around 10
step 950k    around 50 (tunnel strategy starts appearing)
step 1.2M    around 85
step 1.5M    around 114
step 1.9M    around 141
step 2M      around 150-190 per full game

random agent scores about 1-2, human average is around 30.


based on

Human-level control through deep reinforcement learning, Mnih et al., Nature 2015


FLOW

1. OBSERVE
   agent looks at the last 4 grayscale frames → (4, 84, 84)

2. ACT
   if random() < epsilon : random action (explore)
   else: pass frames through CNN → get Q-values for all 4 actions
          and pick the highest one (exploit)

3. ENVIRONMENT STEP
   take the action : get reward, next 4 frames, done

4. STORE
   push (obs, action, reward, next_obs, done) into replay buffer

5. SAMPLE (every 4 steps, after 80k)
   pull a random batch of 32 transitions from the buffer

6. COMPUTE TARGET via Bellman
   pass next_obs through target network → get Q-values
   y = r + 0.99 * max Q_target(next_obs) * (1 - done)

7. COMPUTE PREDICTION
   pass obs through online network
   grab Q-value only for the action that was actually taken via gather()

8. COMPUTE LOSS
   huber(prediction - target)
   i.e. how wrong was the online network's Q-value estimate

9. BACKPROP
   optimizer.zero_grad()
   loss.backward()
   clip gradients
   optimizer.step() : online network weights update

10. TARGET NETWORK UPDATE (every 10k steps)
    copy online network weights : target network
    target stays frozen until next 10k steps

11. EPSILON DECAY
    nudge epsilon down slightly
    more exploitation as training progresses

repeat from step 1