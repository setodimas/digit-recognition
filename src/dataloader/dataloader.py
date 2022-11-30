from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import math
import pickle
import numpy as np


def dump_to_pickle(path, obj_to_dump):
    with open(path, 'wb') as f:
        pickle.dump(obj_to_dump, f)


class MnistLoader():
    
    def __init__(self, config) -> None:
        """
        Constructor to iniatialize trainig dan testing datasets for MNIST
        config : yaml config
        """
        
        # Configuration parameters
        self.config = config
        
        # MNIST training and testing dataset
        self.X_train_mnist = np.array([])
        self.y_train_mnist = np.array([])
        self.X_test_mnist = np.array([])
        self.y_test_mnist = np.array([])
        
        # Testing dataset
        self.X_train = np.array([])
        self.y_train = np.array([])
        
        # Validation dataset
        self.X_val = np.array([])
        self.y_val = np.array([])
        
        # Testing dataset
        self.X_test = np.array([])
        self.y_test = np.array([])

        # Total number of class labels
        self.num_of_classes = 0
        
        # Class label list
        self.list_of_classes = []
        
        # Load dataset from disk/library
        self.load_dataset()
        
        # Calculate the number of class labels and list them.
        self.calculate_class_label_size()

		# Print the details of the dataset.
        self.print_dataset_details()

		# Preprocess the dataset (normalize, one-hot-shot encoding).
        self.preprocess_dataset()
		
        return

                
    def load_dataset(self):
        """
        Load the dataset
        """
        # Load dataset from Keras
        print("Loading dataset from Keras ...")
        (self.X_train_mnist, self.y_train_mnist), (self.X_test_mnist, self.y_test_mnist) = mnist.load_data()
        
        print("Dataset succesfully loaded!")
        
        # Concatenate training and testing dataset to create raw dataset
        print("Creating raw dataset ...")
        self.X_raw = np.concatenate([self.X_train_mnist, self.X_test_mnist])
        self.y_raw = np.concatenate([self.y_train_mnist, self.y_test_mnist])
        
        dump_to_pickle(self.config['raw_dataset_dir'][0], self.X_raw)
        dump_to_pickle(self.config['raw_dataset_dir'][1], self.y_raw)
        
        print("Raw dataset succesfully created and saved!")
        
    
    def print_dataset_details(self):
        """
        Print details of the dataset
        """
        
        # Number of samples in dataset
        print("Training dataset size (X_train, y_train) is: ", self.X_train.shape, self.y_train.shape)
        print("Testing dataset size (X_test, y_test) is: ", self.X_test.shape, self.y_test.shape)

        # Number of class labels and their list.
        print("Total number of Classes in the dataset: ", self.num_of_classes)
        print("The ", self.num_of_classes," Classes of the dataset are: ", self.list_of_classes)
		
        return
    
        
    def calculate_class_label_size(self):
        """
        Calculates the total number of classes in the dataset
        """
        
        self.list_of_classes = np.unique(self.y_train)
        self.num_of_classes = len(self.list_of_classes)
        print("Calculated number of classes and its list from the dataset.")
		
        return
    
    
    def display_data(self, num_to_display, which_data):
        """
        Display element of selected dataset
        """
        
        # Create plot figure
        plt.figure(figsize=(20,20))
        
        # Display training dataset
        if(which_data=="training"):
            num_cells = math.ceil(math.sqrt(num_to_display))
            for i in range(num_to_display):
                plt.subplot(num_cells, num_cells, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(self.X_train[i], cmap=plt.cm.binary)
                plt.title(f"{num_to_display} Sample images of training dataset")
            train_image_path = os.path.join( self.config['image_dir'], "sample_training_mnist_image.png")
            
            if(self.config['save_plots'] == 'true'):
                plt.savefig( train_image_path , bbox_inches='tight')
                print(num_to_display, which_data, " data is saved at path: ", train_image_path)
            else:
                plt.show()
                print(num_to_display, which_data, " data is displayed.")
                
        # Display training dataset
        if(which_data=="validation"):
            num_cells = math.ceil(math.sqrt(num_to_display))
            for i in range(num_to_display):
                plt.subplot(num_cells, num_cells, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(self.X_val[i], cmap=plt.cm.binary)
                plt.title(f"{num_to_display} Sample images of validation dataset")
            val_image_path = os.path.join( self.config['image_dir'], "sample_validation_mnist_image.png")
            
            if(self.config['save_plots'] == 'true'):
                plt.savefig( val_image_path , bbox_inches='tight')
                print(num_to_display, which_data, " data is saved at path: ", val_image_path)
            else:
                plt.show()
                print(num_to_display, which_data, " data is displayed.")

        # Display training dataset
        if(which_data=="test"):
            num_cells = math.ceil(math.sqrt(num_to_display))
            for i in range(num_to_display):
                plt.subplot(num_cells, num_cells, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(self.X_test[i], cmap=plt.cm.binary)
                plt.title(f"{num_to_display} Sample images of test dataset")
            test_image_path = os.path.join( self.config['image_dir'], "sample_test_mnist_image.png")
            
            if(self.config['save_plots'] == 'true'):
                plt.savefig( test_image_path , bbox_inches='tight')
                print(num_to_display, which_data, " data is saved at path: ", val_image_path)
            else:
                plt.show()
                print(num_to_display, which_data, " data is displayed.")    
    
    
    def preprocess_dataset(self):
        """
        Dataset preprocessing.
        
        Split raw dataset into training, validation and testing data;
        Reshaping dataset; 
        Data type conversion;
        Normalize data values
        """
        # Splitting raw dataset
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_raw, self.y_raw, test_size= 1 - self.config['train_ratio'])
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_val, self.y_val, test_size=((self.config['test_ratio']/(self.config['validation_ratio'] + self.config['test_ratio']))))
        
        # Reshape dataset
        self.X_train_with_channels = self.X_train.reshape(
            self.X_train.shape[0],
            self.config['IMAGE_WIDTH'],
            self.config['IMAGE_HEIGHT'],
            self.config['IMAGE_CHANNELS']
        )
        
        self.X_val_with_channels = self.X_val.reshape(
            self.X_val.shape[0],
            self.config['IMAGE_WIDTH'],
            self.config['IMAGE_HEIGHT'],
            self.config['IMAGE_CHANNELS']
        )
        
        self.X_test_with_channels = self.X_test.reshape(
            self.X_test.shape[0],
            self.config['IMAGE_WIDTH'],
            self.config['IMAGE_HEIGHT'],
            self.config['IMAGE_CHANNELS']
        )
        
        # Convert integer pixel values to float type
        self.X_train_with_channels = self.X_train_with_channels.astype('float32')
        self.X_val_with_channels = self.X_val_with_channels.astype('float32')
        self.X_test_with_channels = self.X_test_with_channels.astype('float32')
        
        # Normalize pixel values from range 0-255 to 0-1
        self.X_train_normalized = self.X_train_with_channels / 255
        self.X_val_normalized = self.X_val_with_channels / 255
        self.X_test_normalized = self.X_test_with_channels / 255
        
        # Convert class labels from categorical to one hot encoding 
        self.y_train_ohe = to_categorical(self.y_train)
        self.y_val_ohe = to_categorical(self.y_val)
        self.y_test_ohe = to_categorical(self.y_test)
        
        # Save prepocessed dataset to pickle
        dump_to_pickle(self.config['processed_dataset_dir'][0], self.X_train_normalized)
        dump_to_pickle(self.config['processed_dataset_dir'][1], self.y_train_ohe)
        dump_to_pickle(self.config['processed_dataset_dir'][2], self.X_val_normalized)
        dump_to_pickle(self.config['processed_dataset_dir'][3], self.y_val_ohe)
        dump_to_pickle(self.config['processed_dataset_dir'][4], self.X_test_normalized)
        dump_to_pickle(self.config['processed_dataset_dir'][5], self.y_test_ohe)
        
        print("Preprocessing dataset completed!")
        
        return