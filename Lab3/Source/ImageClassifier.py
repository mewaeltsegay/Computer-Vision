import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.avgpool1(x)
        x = self.tanh(self.conv2(x))
        x = self.avgpool2(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class MNISTClassifier:
    def __init__(self, device=None):
        """
        Initialize the MNIST classification system.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = list(range(10))
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load MNIST dataset
        self.train_dataset = datasets.MNIST('./data', train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST('./data', train=False, transform=self.transform)
        
        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        self.model = LeNet5().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def train(self, epochs=10):
        """
        Train the LeNet-5 model.
        """
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
            # Calculate training metrics
            train_loss = train_loss / len(self.train_loader)
            train_acc = 100. * correct / total
            
            # Validation phase
            val_loss, val_acc = self.evaluate()
            
            # Store metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch: {epoch+1}/{epochs}')
            print(f'Training Loss: {train_loss:.3f} | Training Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')
        
        return history
    
    def evaluate(self):
        """
        Evaluate the model on test data.
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_loss = test_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        
        return test_loss, test_acc
    
    def plot_training_history(self, history):
        """
        Plot training and validation accuracy/loss.
        """
        plt.figure(figsize=(12, 4))
        
        # Accuracy subplot
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'])
        plt.plot(history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Loss subplot
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'])
        plt.plot(history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def get_predictions(self, loader):
        """
        Get predictions for the entire dataset.
        """
        all_predictions = []
        all_targets = []
        all_probs = []
        
        self.model.eval()
        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.device)
                output = self.model(data)
                probs = torch.nn.functional.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_targets), np.array(all_probs)
    
    def plot_confusion_matrix(self):
        """
        Plot confusion matrix for the test set predictions.
        """
        predictions, targets, _ = self.get_predictions(self.test_loader)
        
        # Calculate confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(targets, predictions,
                                 target_names=[str(i) for i in self.class_names]))
    
    def visualize_predictions(self, num_images=10):
        """
        Visualize model predictions on test images.
        """
        dataiter = iter(self.test_loader)
        images, labels = next(dataiter)
        images = images[:num_images]
        labels = labels[:num_images]
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
            probs = torch.nn.functional.softmax(outputs, dim=1)
            predictions = outputs.max(1)[1]
        
        # Create a grid of images
        plt.figure(figsize=(15, 3))
        for i in range(num_images):
            plt.subplot(1, num_images, i+1)
            plt.imshow(images[i].squeeze(), cmap='gray')
            pred_class = predictions[i].item()
            true_class = labels[i].item()
            confidence = probs[i][pred_class].item() * 100
            plt.title(f'Pred: {pred_class}\nTrue: {true_class}\nConf: {confidence:.1f}%')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    # Create and train the classifier
    mnist_classifier = MNISTClassifier()
    
    # Train the model
    history = mnist_classifier.train(epochs=10)
    
    # Plot training history
    mnist_classifier.plot_training_history(history)
    
    # Evaluate the model
    test_loss, test_accuracy = mnist_classifier.evaluate()
    print(f"Test accuracy: {test_accuracy:.2f}%")
    
    # Visualizations
    mnist_classifier.plot_confusion_matrix()
    mnist_classifier.visualize_predictions()

if __name__ == '__main__':
    main()