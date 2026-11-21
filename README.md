# The Purpose of the Model
Imagine that each time player interacts with an NPC, a dialogue box appears, allowing the players to enter anything they want.

In that case, we need a model that can take those inputs, predict the players's intention, and trigger appropriate in-game feature.

Our model takes the text and returns a label representing an in-game feature.

## Description
The model’s main structure is a neural network. We convert the text into numbers and pass those numbers through hidden layers.

Finally, we use Softmax to produce a score associated with each label.

In the hidden layers, we use one of two combinations of activation functions: ReLU and Linear, or SiLU and Linear.

## Requirement

torch                     2.14.1

numpy                     2.5.4

pandas                    3.0.6

scikit-learn              1.9.1

scipy                     1.18.1

streamlit                 1.66.0

## Database
We couldn’t find any free data sources for this problem, so we decided to create around 100 examples by hand and use AI to generate the rest.

We know this isn’t ideal. In the future, we hope to collect more real-world data to further improve the model.

