Team: Team 69
Author: Matthew Spillman

A player agent consisting of a striker and goalie, each controlled by a neural network
with 1 hidden layer and 1024 neurons. The striker was trained first, then
used to train the goalie, then both were fine-tuned with self-play.

The four stages of training correspond to the following weight files:

Stage 1 : striker.pkl, None
Stage 2 : striker.pkl, goalie.pkl
Stage 3 : striker.pkl, goalie_2.pkl
Stage 4 : striker_2.pkl, goalie_3.pkl
