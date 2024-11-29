# Introduction to AI (67842) Final Project
Eliav Opatowsky - 326454550  
Liam Mohr - 209734268  
Eyal Weintroub - 327693024  
Group 66

In this repo we provide all of our code. It is divided into 3 forlders, one for each model we implemented to solve the game:

## Baseline
This folder contains the code of the graphical implementation of Hanabi, and our simple baseline model.  
Most of the code is from [this repository](https://github.com/yawgmoth/pyhanabi), and our implementation is in the file `game.py` in the `FixedPlayer` class.  
Use the README inside the folder for running the code.

## MCTS
This foldoer contains our RIS-MCTS implementation. In order to run the code, you can either run the `mcts_main.py` file, or the `run.bat` file to simulate many runs.
In order to change turn time, do it in the main call in the main file. We started with the game state from this 
[library](https://github.com/git-pushz/hanabi-mcts) and then changed and modified it to our implementation of 
RIS-MCTS

## Learner
This folder contains our deep neural network solution. In order to run the learner, run the `ppo_learner.ipynb` notebook, which contains the environment and the model training.
