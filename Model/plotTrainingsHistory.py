#!/usr/bin/env python

# Python libraries
import matplotlib
# Specifying the backend to be used before importing pyplot
# to avoid "RuntimeError: Invalid DISPLAY variable"
matplotlib.use('agg')
import keras
import numpy as np
import pydot as pyd
import matplotlib.pyplot as plt
from IPython.display import SVG

def plot_loss_acc(model):
  color_red = 'tab:red'
  color_blue = 'tab:blue'

  epochs=np.array(model.history.epoch)+1 # Add one to the list of epochs which is zero-indexed

  loss=np.array(model.history.history['loss'])
  acc=np.array(model.history.history['acc'])
    
  val_loss=np.array(model.history.history['val_loss'])
  val_acc=np.array(model.history.history['val_acc'])
  
  # Create subplots
  fig, (ax1, ax3) = plt.subplots(1,2, figsize=(20,6))

  # TRAING DIAGRAM
  ax1.title.set_text('TRAININGS DATA')
  
  ax1.set_xlabel('Epochs',fontsize=15) 
  ax1.set_ylabel('Loss', color=color_red,fontsize=15)
  ax1.plot(epochs, loss, color=color_red,lw=2)
  ax1.tick_params(axis='y', labelcolor=color_red)
  ax1.grid(True)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
  ax2.set_ylabel('Accuracy', color=color_blue,fontsize=15)
  ax2.plot(epochs, acc, color=color_blue,lw=2)
  ax2.tick_params(axis='y', labelcolor=color_blue)

  # VALIDATION DIAGRAM
  ax3.title.set_text('VALIDATION DATA')
  
  ax3.set_xlabel('Epochs',fontsize=15)
  ax3.set_ylabel('Validation Loss', color=color_red,fontsize=15)
  ax3.plot(epochs, val_loss, dashes=[2, 2, 10, 2], color=color_red,lw=2)
  ax3.tick_params(axis='y', labelcolor=color_red)
  ax3.grid(True)

  ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
  ax4.set_ylabel('Validation Accuracy', color=color_blue,fontsize=15)
  ax4.plot(epochs, val_acc, dashes=[2, 2, 10, 2], color=color_blue,lw=2)
  ax4.tick_params(axis='y', labelcolor=color_blue)
  fig.tight_layout()

  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()


class TrainingPlot(keras.callbacks.Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            #plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig('output/Epoch-{}.png'.format(epoch))
        plt.close()