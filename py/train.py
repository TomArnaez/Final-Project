from torchvision import datasets
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import torch
from tqdm.notebook import tqdm
from torch import optim, LongTensor

class customDataset(Dataset):
    def __init__(self, x_path, y_path, transform=None):
        self.xs = torch.from_numpy(np.transpose(np.load(x_path), [0, 3, 1, 2])  / 255).type('torch.FloatTensor')
        #self.xs = torch.from_numpy(np.load(x_path))
        self.ys = torch.from_numpy(np.load(y_path))
        self.transform = transform
        
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]
    
def train_on_attack(model, criterion, optimizer, train_loader, device, atk, batch_size=128, n_epochs=5, log_interval=10, disp=False):
    train_counter = []
    train_losses = []
    model.train()
    with tqdm(total=len(train_loader) * n_epochs) as pbar:
        for epoch in range(1, n_epochs+1):
            for batch_idx, (images, labels) in enumerate(train_loader):
                X = atk(images, labels).to(device)
                Y = labels.to(device)
                #print(optimizer)
                pre = model(X)
                cost = criterion(pre, Y)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                if batch_idx % log_interval == 0:
                    if disp:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(images), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), cost.item()))
                    train_losses.append(cost.item())
                    train_counter.append(
                    (batch_idx*batch_size) + ((epoch-1)*len(train_loader.dataset)))
                    #torch.save(model.state_dict(), './results/mnist_bath_model.pth')
                    #torch.save(optim.state_dict(), './results/mnist_base_optimizer.pth')
                pbar.update()
                
    return train_counter, train_losses
                                
def test(model, test_loader, device, classes, show_class_accs=False):
    """
    Tests a models accuracy for a given list of attacks
    
    @param model: PyTorch model to be tested
    @param test_loader: PyTorch DataLoader holding test dataset
    @param atks: list of torchattacks
    @param show_class: display per-class accuracy
    """
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    model.eval()
    
    correct = 0
    total = 0
    
    # don't need to calculate gradients for forward and backward phase
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum()

                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1
                    
                pbar.update()
                
    # print accuracy for each class
    if show_class_accs:
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'\tAccuracy for class: {classname} is {accuracy:.1f} %')

    acc = (100 * float(correct) / total)
    # pritn total accuracy
    #print(f'Overall accuracy: {acc:.2f}')
    return acc
                
def test_on_attack(model, test_loader, device, classes, atk, show_class=False):
    """
    Tests a models standard and robust accuracy for a given list of attacks
    
    @param model: PyTorch model to be tested
    @param test_loader: PyTorch DataLoader holding test dataset
    @param atks: list of torchattacks
    @param show_class: display per-class accuracy
    """

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    model.eval()
    
    correct = 0
    total = 0
    with tqdm(total=len(test_loader)) as pbar:
        for images, labels in test_loader:
            images = atk(images, labels).to(device)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum()

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
            pbar.update()
       # print accuracy for each class
    
    if show_class:
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'\tAccuracy for class: {classname} is {accuracy:.1f} %')
            
    robust_acc = (float(correct) / total) * 100
    print(f'Robust accuracy for {atk.__class__.__name__}: {robust_acc:.2f}')
    return robust_acc

def test_on_adv_examples2(model, criterion, test_loader, batch_size, classes, device, atk, disp=False):
    atk.set_return_type('int') # Save as integer.
    atk.save(data_loader=test_loader, save_path="./datasets/temp.pt", verbose=True)
    adv_images, adv_labels = torch.load("./datasets/temp.pt")
    adv_data = TensorDataset(adv_images.float()/255, adv_labels)
    adv_loader = DataLoader(adv_data, batch_size=64, shuffle=False)
    
    return test(model, criterion, adv_loader, batch_size, classes, device, disp=disp)

def train_mnist(model, optim, criterion, train_loader, device, n_epochs=3, batch_size=128, log_interval=10, disp=False):
    train_losses = []
    train_counter = []
    model.to(device)
    model.train()
    with tqdm(total=len(train_loader) * n_epochs) as pbar:
        for epoch in range(1, n_epochs+1):
            running_loss = 0.0
            for batch_idx, (images, labels) in enumerate(train_loader):
                # Flatten MNIST images into a 784 long vector
                #images = images.view(images.shape[0], -1)
                images, labels = images.to(device), labels.to(device)

                # Training pass
                optim.zero_grad()

                output = model(images)

                loss = criterion(output, labels)

                # This is where the model learns by backpropogating
                loss.backward()

                # And optimizes its weights here
                optim.step()

                if batch_idx % log_interval == 0:
                    if disp:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(images), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                    train_losses.append(loss.item())
                    train_counter.append(
                    (batch_idx*batch_size) + ((epoch-1)*len(train_loader.dataset)))
                    #torch.save(model.state_dict(), './results/mnist_bath_model.pth')
                    #torch.save(optim.state_dict(), './results/mnist_base_optimizer.pth')
                pbar.update()
                
    return train_losses, train_counter
       
def test_mnist_c(model, criterion, classes, corruptions, device, batch_size, num_workers=4, disp=False):

    transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    model.eval()
    corruption_acc = {}
    for corruption in corruptions:
        x_path_test = f"./datasets/mnist_c/{corruption}/test_images.npy"
        y_path_test = f"./datasets/mnist_c/{corruption}/test_labels.npy"

        test_dataset = customDataset(x_path, y_path, transform=transform)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        
        x_test = test_dataset[0]
             
        print(f"Testing on corruption {corruption}")
        test_acc, test_loss = test(model, criterion, test_loader, batch_size, classes, device, disp)
        corruption_acc[corruption] = test_acc
        print(f"Test loss: {test_loss:.4f} \nTest acc: {test_acc:.4f}")
    return corruption_acc
    
def test_cifar_on_corruptions(model, criterion, classes, corruptions, device, batch_size, num_workers=4, disp=False):
    transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    base_path = "./datasets/CIFAR-10-C/"
    y_path = base_path + "labels.npy"
    model.eval()
    corruption_acc = {}
    for corruption in corruptions:
        x_path = base_path + corruption + ".npy"

        test_dataset = customDataset(x_path, y_path, transform=transform)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        x_test = test_dataset[0]
             
        print(f"Testing on corruption {corruption}")
        test_acc, test_loss = test(model, criterion, test_loader, batch_size, classes, device, disp)
        corruption_acc[corruption] = test_acc
        print(f"Test loss: {test_loss:.4f} \nTest acc: {test_acc:.4f}")
    return corruption_a

def cifar_calc_fps(model, perturbations, device):
    """
    Method for calculating the flip probability of a model for a given set of perturbations"""
    num_classes = 10
    dummy_targets = LongTensor(np.random.randint(0, num_classes, (10000,)))
    results = {}
    flip_probability = 0
    with tqdm(total=len(perturbations)) as pbar:
        for perturbation in perturbations:
            #pbar.update()
            #pbar.set_postfix_str(f'{perturbation}')
            noise = "noise" in perturbation
            data = torch.from_numpy(np.float32(
            np.load("./datasets/CIFAR-10-P/" + perturbation + ".npy").transpose((0,1,4,2,3))))/255.
            dataset = torch.utils.data.TensorDataset(data, dummy_targets)
            loader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=False, num_workers=2)        
            total_count = 0
            changed_count = 0
            with torch.no_grad():
                for data, target in loader:
                    data = data.to(device)
                    num_vids = data.size(0)
                    data = data.view(-1, 3, 32, 32).to(device)
                    output = model(data)
                    _, pred = torch.max(output, 1)   
                    reshaped = pred.view(num_vids, -1).cpu().numpy()
                    if not noise:
                        changed_count += sum(sum(reshaped[:,1:] != reshaped[:,-1:]))
                    else:
                        for vid in reshaped:
                            changed_count += sum(vid[1:] != vid[0])
                            

                    shape = reshaped.shape
                    total_count += shape[0] * (shape[1]-1)
                                                
                    flip_probability = changed_count / total_count
 
            pbar.update()
            results[perturbation] = flip_probability
    return results

def test_cifar(model, device):
    
    # track test loss
    test_loss = 0.0
    class_correct = [0. for i in range(10)]
    class_total = [0. for i in range(10)]

    model.eval()
    # iterate over test data
    for data, target in test_loader:
        # move tensors to GPU if CUDA is available
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        #print(correct_tensor)
        #correct = np.squeeze(correct_tensor.cpu().numpy() if (device == "cuda") else np.squeeze(correct_tensor.numpy()))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss/len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.6f}\n')
    
    for i in range(len(classes)):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    return test_loss

def train_with_validation(model, optimizer, train_loader, valid_loader, device, classes, save_dir="./models/cifar_model.pt", n_epochs=30):
    """
    @param model: PyTorch model to train on
    @param optimizer: PyTorch optimizer to set values for model
    @param train_data: dataset containg train images and labels
    @param device: CPU or GPU to use for training
    @param classes: list of classes for training_data
    """
    # how many samples per batch to load
    batch_size = 20

    # convert data to a normalized torch.FloatTensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    import torch.nn as nn
    # specify loss function
    criterion = nn.CrossEntropyLoss()
    
    # List to store loss to visualize
    train_losslist = []
    valid_losslist = []
    # track change in validation loss
    
    valid_loss_min = np.Inf 

    with tqdm(total=len(valid_loader) * n_epochs) as pbar:
        for epoch in range(1, n_epochs+1):
            # eep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0

            for data, target in train_loader:
                # Move data to the GPU if available
                data, target = data.to(device), target.to(device)
                # Clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                 # This is where the model learns by backpropogating
                loss.backward()
                 # And optimizes its weights here
                optimizer.step()
                # Update with new training loss
                train_loss += loss.item()*data.size(0)


            model.eval()
            for data, target in valid_loader:
                # move tensors to GPU if CUDA is available
                data, target = data.to(device), target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss 
                valid_loss += loss.item()*data.size(0)
                pbar.update()


            # calculate average losses
            train_loss = train_loss/len(train_loader.dataset)
            valid_loss = valid_loss/len(valid_loader.dataset)
            train_losslist.append(train_loss)
            valid_losslist.append(valid_loss)

            # print training/validation statistics 
            print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')

            #print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                #epoch, train_loss, valid_loss))

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...')
                torch.save(model.state_dict(), save_dir)
                valid_loss_min = valid_loss

    # Load the model with the lowest validation loss
    model.load_state_dict(torch.load(save_dir))
    
    return train_losslist, valid_losslist