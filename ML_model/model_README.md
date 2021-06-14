To run the model, make sure you have preprocessed the images in the /final directory within the dataset directory you are using. See resize_readme for details.

Next activate your virtual environment.

Modify train_cnn_pt.py to reference the desired dataset (for instance, "python_dataset", "noise_medium_3", etc).

cd ML_model
python train_cnn_pt.py

To monitor training, run the following in a new shell window (make sure you have tensorboard installed):

cd information-encoding-pytorch/ML_model
tensorboard --logdir=runs

and select the appropriate metrics from your current run to monitor.
