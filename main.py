from src.dataloader.dataloader import MnistLoader
from src.model.mnist_model import MnistModel
from utils.model_utils import Report
from utils.configs import load_config
# from utils.load_configuration import config
 
yaml_file = 'configs/config.yaml'

def main():
    config = load_config(yaml_file)
    
    # Load dataset from mnist library
    dataset = MnistLoader(config)
    
    # Display data element
    dataset.display_data(100, 'training')
    dataset.display_data(100, 'validation')
    dataset.display_data(100, 'test')
    
    # Construct cnn model
    model = MnistModel(config, dataset)
    
    # Save cnn model
    model.save_model()
    
    # Plot classification report and confusion matrix
    report = Report(config, model)
    report.plot()
    report.model_classification_report()
    report.plot_confusion_matrix()
 
if __name__ == '__main__':       
    main()