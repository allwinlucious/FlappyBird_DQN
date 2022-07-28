# FlappyBird_DQN

## How to Run?
```
git clone https://github.com/mewbot97/FlappyBird_DQN.git
cd FlappyBird_DQN_master/src/
python dqn_play.py
```

## Train new model/ retrain existing model
PS:delete the existing model('b2d.pt') in '/src/model/' folder to train a new model
```
cd FlappyBird_DQN_master/src/
python dqn_train.py
```
##### NN architecture can be changed in '/src/dqn_model_class.py'

##### The state used is 
 horizontal distance to the next pipe;
 difference between the player's y position and the next hole's y position.
 refer [here](https://github.com/Talendar/flappy-bird-gym) for more info
##### The agent receives +1 reward for each timestep alive and -100 when it crashes
