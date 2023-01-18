from torchattacks import *
import torch
from tqdm import tqdm

def get_attacks(model):
    """
    @param model: PyTorch model to run attacks on
    """
    atks = [
        FGSM(model, eps=0.3),
        PGD(model, eps=0.3, alpha=0.1, steps=7),
        CW(model, c=5, lr=0.001),
    ]
    
    return atks

def run_attacks(model, testset):
    atks = [
    FGSM(model, eps=8 / 255),
    #BIM(model, eps=8 / 255, alpha=2 / 255, steps=100),
    #RFGSM(model, eps=8 / 255, alpha=2 / 255, steps=100),
    CW(model, c=1e-3, lr=0.001, steps=100, kappa=0),
    PGD(model, eps=8 / 255, alpha=2 / 225, steps=100, random_start=True),
    # PGDL2(model, eps=1, alpha=0.2, steps=100),
    # EOTPGD(model, eps=8 / 255, alpha=2 / 255, steps=100, eot_iter=2),
    # FFGSM(model, eps=8 / 255, alpha=10 / 255),
    # TPGD(model, eps=8 / 255, alpha=2 / 255, steps=100),
    # MIFGSM(model, eps=8 / 255, alpha=2 / 255, steps=100, decay=0.1),
    # VANILA(model),
    # GN(model, std=0.1),
    # APGD(model, eps=8 / 255, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
    # APGD(model, eps=8 / 255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
    # APGDT(model, eps=8 / 255, steps=100, eot_iter=1, n_restarts=1),
    # FAB(model, eps=8 / 255, steps=100, n_classes=10, n_restarts=1, targeted=False),
    # FAB(model, eps=8 / 255, steps=100, n_classes=10, n_restarts=1, targeted=True),
    # Square(model, eps=8 / 255, n_queries=5000, n_restarts=1, loss='ce'),
    # AutoAttack(model, eps=8 / 255, n_classes=10, version='standard'),
    # OnePixel(model, pixels=5, inf_batch=50),
    # DeepFool(model, steps=100),
    # DIFGSM(model, eps=8 / 255, alpha=2 / 255, steps=100, diversity_prob=0.5, resize_rate=0.9)
    ]

    robustness = {}
    for atk in atks:
        correct = 0
        for j, (images, labels) in tqdm(enumerate(testset), total=len(testset)):
            model.eval()
            adv_images = atk(images, labels)
            outputs = model(adv_images.cuda())
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted.cpu() == labels).sum()
            
        robust_acc = correct / len(testset.dataset)
        atk_name = atk.__class__.__name__
        print(f"Robust accuracy for {atk_name}: {robust_acc:.4f}")
        robustness[atk_name] = robust_acc
    return robustness
