from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import History, ModelCheckpoint
from keras.models import load_model
import numpy as np
import time


class MnistModel():
    
    def __init__(self, config, dataset):
        """
        CNN Mnist constructor

        Args:
            config (_type_): yaml configuration
            dataset (_type_): training, validation and testing datasets
        """
        
        # Configuration parameters
        self.config = config
        
        # Training, validation and testing datasets
        self.dataset = dataset
        
        # CNN model
        self.cnn_model = Sequential()
        
        # Training history object
        self.history = History()
        
        # Saved model path
        self.saved_model_path = self.config['model_path']
        
        # Checkpoint model
        self.checkpoint = ModelCheckpoint(self.saved_model_path,
                                          monitor='val_acc',
                                          verbose= self.config['checkpoint_verbose'],
                                          save_best_only=True,
                                          mode='max')
        
        # Callbacks
        self.callbacks_list = [self.checkpoint]
        
        # Evaluation score
        self.scores = []
        
        # Training time
        self.train_time = 0
        
        # Predicted class labels
        self.predictions = np.array([])
        
        # Construct model
        self.define_model()
        
        # Compile model
        self.compile_model()
        
        # Train model with testing dataset
        self.fit_model()
        
        # Evaluate model
        self.evaluate_model()
        
        # Predict testing dataset
        self.predict()
        
        return
    
    def save_model(self):
        """
        Saves model in h5 format
        """
        
        if(self.cnn_model is None):
            raise Exception("Model is not avaible")
        
        self.cnn_model.save(self.saved_model_path)
        print("Model saved at :", self.saved_model_path, "\n")
        
        return
    
    def load_cnn_model(self):
        """
        Load saved model
        """
        
        if(self.cnn_model is None):
            raise Exception("Model is not avaible")
        
        self.cnn_model.load_weight(self.saved_model_path)
        print("Model loaded from :", self.saved_model_path, "\n")
        
        return
    
    def define_model(self):
        """
        Construct CNN Model
        """
        
        # Layer 1 Conv
        self.cnn_model.add(Conv2D(input_shape = (self.config['image_width'], self.config['image_height'], self.config['image_channels']),
                                  filters = self.config['conv_filter_l1'],
                                  kernel_size = (self.config['kernel_row'], self.config['kernel_col']),
                                  activation = self.config['conv_activation_l1']))
        
        # Layer 2 Pooling
        self.cnn_model.add(MaxPooling2D(pool_size = (self.config['pool_size_row'], self.config['pool_size_col'])))
        
        # Layer 3 Conv
        self.cnn_model.add(Conv2D(filter = self.config['conv_filter_l2'],
                                  kernel_size = (self.config['kernel_row'], self.config['kernel_col']),
                                  activation = self.config['conv_activation_l2']))
        
        # Layer 4 Pooling
        self.cnn_model.add(MaxPooling2D(pool_size = (self.config['pool_size_row'], self.config['pool_size_col'])))
        
        # Layer 5 Flatten
        self.cnn_model.add(Flatten())
        
        # Layer 6 Dense
        self.cnn_model.add(Dense(units = self.config['dense_filter_l1'],
                                 activation = self.config['dense_activation_l1']))
        
        # Layer 7 Dropout
        self.cnn_model.add(Dropout(self.config['dropout_probability']))
        
        # Layer 8 Dense output
        self.cnn_model.add(Dense(units = self.dataset.num_of_classes))
        
        return self.cnn_model
    
    
    def compile_model(self):
        """
        Compile CNN model
        """
        
        self.cnn_model.compile(loss = self.config['compile_loss'],
                               optimizer = self.config['compile_optimizer'],
                               metrics = self.config['compile_metric'])
        
        
    def fit_model(self):
        """
        Train the CNN Model
        """
        
        start_time = time.time()
        
        if(self.config['save_model'] == 'true'):
            print('Training under progress, model will be saved at ', self.config['model_path'], '...\n')
            self.history = self.cnn_model.fit(x = self.dataset.X_train_normalized,
                                              y = self.dataset.y_train_ohe,
                                              epochs = self.config['epochs'],
                                              verbose = self.config['verbose'],
                                              validation_data = (self.dataset.X_val_normalized, self.dataset.y_val_ohe))
        else:
            print('Training under progress ... \n')
            self.history = self.cnn_model.fit(x = self.dataset.X_train_normalized,
                                              y = self.dataset.y_train_ohe,
                                              epochs = self.config['epochs'],
                                              verbose = self.config['verbose'],
                                              validation_data = (self.dataset.X_val_normalized, self.dataset.y_val_ohe))
        
        end_time = time.time()
        
        self.train_time = end_time - start_time
        print('Model raining time : ', self.train_time)
        
        return
    
    
    def evaluate_model(self):
        """
        Evaluate CNN model on validation dataset
        """
        
        self.scores = self.cnn_model.evaluate(x = self.dataset.X_val_normalized,
                                              y = self.dataset.y_val_ohe)
        
        print('Test loss on validation dataset : ', self.scores[0])
        print('Test accuracy on validation dataset : ', self.scores[1])
        
        return
    
    
    def predict(self):
        """
        Model prediction on unseen dataset / test dataset
        """
        
        self.predictions = self.cnn_model.predict(x = self.dataset.X_test_normalized,
                                                  verbose = self.config['verbose'])
        
        return