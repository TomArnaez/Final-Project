import torch
import torch.nn as nn
from tqdm.notebook import tqdm

#def plot_corruption_errors(models, corruptions, device):
    

def relative_corruption_error(test_model, base_model, corruptions, clean_dataset, device):
    """
    @param test_model: pyTorch model to evaluate on
    @param clean_model: pyTorch model to be evaluated against
    @param corruptions: dict containing corrupted datasets
    @param device: device to perform evaluation with
    @return: dict containing corruption errors for each corruption
    """
    relative_errors = dict()
    
    base_error_clean = error_rate(base_model, clean_dataset, device)
    test_error_clean = error_rate(test_model, clean_dataset, device)
    
    with tqdm(total=len(corruptions)) as pbar:
        for corruption, corrupted_dataset in corruptions.items():
            print(f"Finding relative corruption error for {corruption}")
            
            base_error_corrupted = error_rate(base_model, corrupted_dataset, device)
            base_error_relative = base_error_corrupted - base_error_clean
            
            test_error_corrupted = error_rate(test_model, corrupted_dataset, device)
            test_error_relative = test_error_corrupted - test_error_clean


            try:
                relative_error = (test_error_relative / base_error_relative) * 100
            except ZeroDivisionError:
                relative_error = 0

            relative_errors[corruption] = relative_error
            pbar.update()
        
    return relative_errors

def corruption_errors(model, corruptions, device):
    """
    @param model: pyTorch model to evaluate on
    @param corruptions: dict containing corrupted datasets
    @param device: device to perform evaluation with
    @return: dict containing corruption errors for each corruption
    """
    corruption_errors = dict()
    with tqdm(total=len(corruptions)) as pbar:
        for c, d in corruptions.items():
            e = error_rate(model, d, device)
            #print(f"Error rate for {c}: {e:.4f}")
            corruption_errors[c] = e
            pbar.update()
    return corruption_errors

def mean_corruption_error(corruption_errors):
    errors = [v for _, v in corruption_errors.items() if v != 0]
    average = (sum(errors) / len(errors))
    return average

def error_rate(model, corruption_dataset, device):
    model.to(device)
    all_count = len(corruption_dataset)
    testLoader = torch.utils.data.DataLoader(corruption_dataset, batch_size=512, shuffle=True, num_workers=4)
    sum_E = 0
    all_count = len(corruption_dataset)
    with torch.no_grad():
        for images, labels in testLoader:
            images, labels = images.to(device), labels.to(device)
            E = errors(model, images, labels)
            sum_E += E
    error_rate = (sum_E / all_count) * 100
    return error_rate 

def errors(model, images, labels):
    error_count = 0
    fmodel_outputs = model(images)
    _, fmodel_predictions = torch.max(fmodel_outputs, 1)
    for label, prediction in zip(labels, fmodel_predictions):
        if label != prediction:
            error_count += 1
    return error_count
