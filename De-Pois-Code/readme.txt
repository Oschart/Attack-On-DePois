We release partial code for De-pois against GP-attack [21]. 
Complete code for the paper is available upon request.

Prerequisites:
Install Keras and other dependencies (e.g., sklearn).

Training/testing:
Download MNIST dataset from http://yann.lecun.com/exdb/mnist/

Getting Started:
Run python3 main.py

Generating poisoning data:
We use the code in https://github.com/yangcf10/Poisoning-attack for generating poisoning data using GP-attack;


Code structure:
+ generator_CGAN_authen.py: for the synthetic data generation;
+ mimic_model_construction.py: for the mimic model construction.

