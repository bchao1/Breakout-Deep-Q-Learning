# Deep Q Learning : Atari Breakout
Our actor can achieve an average reward of about 80 over 100 episodes (we didn't have much time to tune the parameters...). We basically followed settings from the Deepmind Q Learning paper.
## Settings 
### Preprocessing 
Each frame is converted to grayscale (single channel) and then resized to 84 * 84. We save each grayscale image * 255 as an np.int array since this saves memory (compared to floating point numbers), which is important in Q-learning since we have to keep a list of history data. 
   
The grayscale values are divided by 255 as the CNN input. We resorted to this solution since we encountered out-of-memory errors when we save the frames as floating point arrays.

### Model architecture 
Please refer to `model.py`. The action space is set to have dimension 3 (left, right and fire/stay).

### Other training settings
- Batchsize (used to train history replay): 32
- Memory length: 400000
- Epsilon linear decay rate (from 1 to 0.1): 1e-6
- Optimizer: RMSprop(lr = 0.00025, alpha = 0.95, eps = 0.01)

## Result
![](./reward.PNG)
