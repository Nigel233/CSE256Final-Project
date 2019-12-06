This folder contains the CSE 256 Final Project code by: Yuanzhen Lin (A53320395), Xinxin Yan (A53312247), Tianyu Sun (A53299972).

Folder Index:
model.py - contains the code of CNN model class
main.py - main training procedure
predict.py - main predict procedure
mydatasets.py - dataset class
train.py - detailed training procedure
tfield, best_steps_21000.pt, l2field, l1field - supportive documents for making predictions
train.tsv, test.tsv: train and test data.
Demo - contains code for our webpage interface

How to run this code:
Perquisites: python3, PyTorch and other necessary libraries
Training procedure:
> python main.py
Predicting procedure:
First open a python3 terminal
> import predict from predict
 > predict(“xxxx text passage”)
 Then you will get the predicted two-layer labels.
Demo display:
> cd demo
> export FLASK_APP=view
> flask run
Paste “http://127.0.0.1:5000/mainPage" to any browser and you can use this interface as described in our report 