import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
import numpy as np
import seaborn as sns
import itertools
import os


class Report():
    
    
    def __init__(self, config, model) -> None:
        """
        Report initialization

        Args:
            config (_type_): yaml configuration file
            model (_type_): CNN model
        """
        
        self.config = config
        self.model = model
        
        return
    
    
    def plot(self):
        """
        Loss and accuracy plot for training and validation dataset
        """
        
        loss_list = [i for i in self.model.history.history.keys() if 'loss' in i and 'val' not in i]
        val_loss_list = [i for i in self.model.history.history.keys() if 'loss' in i and 'val' in i]
        acc_list = [i for i in self.model.history.history.keys() if 'acc' in i and 'val' not in i]
        val_acc_list = [i for i in self.model.history.history.keys() if 'acc' in i and 'val' in i]
  
        if len(loss_list) == 0:
            print('Loss is not avaible in model history')
            return
        
        epochs = range(1, len(self.model.history.history[loss_list[0]]) + 1)
        
        # Plot loss
        plt.figure(1)
        
        for l in loss_list:
            plt.plot(epochs,
                     self.model.history.history[1],
                     'b',
                     label = 'Training loss (' + str(str(format(self.model.history.history[l][-1],'.5f')) + ')')
                     )
            
        for l in val_loss_list:
            plt.plot(epochs,
                     self.model.history.history[1],
                     'b',
                     label = 'Training loss (' + str(str(format(self.model.history.history[l][-1],'.5f')) + ')')
                     )
            
        plt.title('Loss per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()         
        
        # Save the plot.
        loss_path = os.path.join(self.config['plot_dir'], "loss.png")

        if(self.config['save_plots'] == 'true'):
            plt.savefig(loss_path, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

		# Accuracy graph.
        plt.figure(2)
        for l in acc_list:
            plt.plot( epochs,
					  self.model.history.history[l],
					  'b',
					  label = 'Training accuracy (' + str(format(self.model.history.history[l][-1],'.5f')) + ')'
					)

        for l in val_acc_list:
            plt.plot( epochs,
					  self.model.history.history[l],
					  'g',
					  label = 'Validation accuracy (' + str(format(self.model.history.history[l][-1],'.5f')) + ')'
					)

        plt.title('Accuracy per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

		# Save the plot to disk.
        acc_path = os.path.join(self.config['plot_dir'], "accuracy.png")
        if(self.config['save_plots'] == 'true'):
            plt.savefig(acc_path, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

        return
    
    
    def classification_report(self):
        """
        CNN classification report
        """
        
        predicted_classes = np.argmax(self.model.predictions, axis = 1)
        print('Accuracy: '+ str(accuracy_score(self.model.dataset.y_test_ohe, predicted_classes)))
        print('Classification Report')
        print('------------------------------------------------------------------')
        target_names = ['Class {}'.format(i) for i in range(self.config['num_classes'])]
        print(classification_report(self.model.dataset.y_test_ohe,
                              predicted_classes,
                              target_names = target_names
                              )
        )
        
        return
    
    
    def plot_confusion_matrix(self):
        """
        Plot classification confusion matrix on test dataset
        """
        
        predicted_classes = np.argmax(self.model.predictions, axis = 1)
        title = 'Confusion matrix on test dataset'
        confusion_matrix = tf.math.confusion_matrix(self.dataset.y_tset_ohe, predicted_classes)

        f, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
                    confusion_matrix,
                    annot = True,
                    linewidths = .5,
                    fmt = "d",
                    square = True,
                    ax = ax,
                    cmap = 'crest'
                    )
        
        plt.tight_layout()
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

		# Save the plot to disk.
        cm_path = os.path.join( self.config['plot_dir'], "confusion_matrix.png")

        if(self.config['save_plots'] == 'true'):
            plt.savefig(cm_path, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

        return