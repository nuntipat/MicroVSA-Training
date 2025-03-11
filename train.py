import os
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

import dataset
import ldc_model
import log
from tqdm import trange

def train_ldc(x_train, y_train, x_val, y_val,
              vhv_dimension, fhv_dimension, num_feature, num_value, num_class,
              epochs=50, batch_size=128, initial_learning_rate=0.001, learning_rate_decay=0.98,
              weight_decay=0.0001, dropout_probability=0, input_min=0, input_max=1, loss_weight=None,
              enable_binarize=True, verbose=True, log_file=None):
    if verbose:
        log.show_log(f"Training Parameters:", file=log_file)
        log.show_log(f"    vhv_dimension = {vhv_dimension}", file=log_file)
        log.show_log(f"    fhv_dimension = {fhv_dimension}", file=log_file)
        log.show_log(f"    initial_learning_rate = {initial_learning_rate}", file=log_file)
        log.show_log(f"    learning_rate_decay = {learning_rate_decay}", file=log_file)
        log.show_log(f"    weight_decay = {weight_decay}", file=log_file)
        log.show_log(f"    dropout_probability = {dropout_probability}", file=log_file)
        log.show_log(f"    input_min = {input_min}", file=log_file)
        log.show_log(f"    input_max = {input_max}", file=log_file)
        log.show_log("", file=log_file)

    train_set = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    testloader = DataLoader(test_set, batch_size=batch_size)

    device = ldc_model.device
    model = ldc_model.ValFeaClaSearchNet(num_feature, num_value, fhv_dimension, vhv_dimension, num_class,
                                   dropout_probability, enable_binarize).to(device)
    # model = LDCConv.LDC((1, 10, 49), fhv_dimension, num_value, vhv_dimension, num_class)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weight) # weight=torch.FloatTensor([1, 32.6, 12.52, 113, 11.26])# custom weight can be passing weight=torch.FloatTensor([18,18,1]) weight=torch.FloatTensor([2.63,1])  weight=torch.FloatTensor([2.6,1])
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_acc = 0
    for t in range(epochs):
        if verbose:
            log.show_log(f"Epoch {t + 1}\n-------------------------------", file=log_file)
        start_time = time.time()

        # train
        size = len(trainloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(trainloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)  # TODO: why?

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                if verbose:
                    log.show_log(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", file=log_file)

            if t > 15:
                for mod in model.modules():
                    if hasattr(mod, 'b'):
                        idxs = mod.b.nonzero(as_tuple=True)
                        mod.weight.data[idxs] = mod.weight_pre[idxs].clone()

                freeze_percent = ldc_model.QAT(model)
            else:
                freeze_percent = 0

        scheduler.step()

        size = len(testloader.dataset)
        num_batches = len(testloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in testloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        if verbose:
            log.show_log(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} FreezeRate: {freeze_percent}", file=log_file)

        if correct > best_acc:
            if verbose:
                log.show_log('Saving model state...', file=log_file)
            torch.save(model.state_dict(), f"model_{fhv_dimension}.pth")
            best_acc = correct

        # optimizer.param_groups[0]['lr'] *= learning_rate_decay ** t
        if verbose:
            log.show_log(f'{time.time() - start_time} sec\n', file=log_file)

    if verbose:
        log.show_log('Summary\n-----------------------', file=log_file)
        log.show_log(f'Best accuracy: {(100 * best_acc):>0.1f}%\n', file=log_file)

    # load the best parameter value saved during training

    model.load_state_dict(torch.load(f"model_{fhv_dimension}.pth"))
    model.eval()

    # Extract Model Paramater

    # feature_weights = model.fc1.weight * model.fc1.mask.T
    # bin_feature_weights = torch.Tensor.cpu(torch.sign(feature_weights)).detach().numpy().T
    # featureMemory = np.zeros((num_feature, fhv_dimension))
    # for i in range(num_feature):
    #     for j in range(fhv_dimension):
    #         featureMemory[i, j] = bin_feature_weights[i * vhv_dimension + j % vhv_dimension, j]
    #         if featureMemory[i, j] == 0:
    #             log.show_log(f'featureMemory: ({i}, {j})', file=log_file)
    featureMemory = torch.Tensor.cpu(torch.sign(model.fc1.weight)).detach().numpy()

    valueMemory = np.zeros((num_value, fhv_dimension))
    valueBox = model.fc0
    base = torch.sign(valueBox(torch.linspace(input_min, input_max, num_value).reshape(num_value, 1).to(device)))
    base = torch.Tensor.cpu(base).detach().numpy()
    for i in range(num_value):
        for j in range(fhv_dimension):
            valueMemory[i, j] = base[i, j % vhv_dimension]
            if valueMemory[i, j] == 0:
                log.show_log(f'valueMemory: ({i}, {j})', file=log_file)

    class_weights = torch.Tensor.cpu(torch.sign(model.fc2.weight)).detach().numpy()
    associativeMemory = class_weights

    return featureMemory, valueMemory, associativeMemory


def eval_ldc(F, V, C, x_test, y_test, input_min, input_max, binarize=True, verbose=True, log_file=None,
             result_img_path='result.png', result_image_size=(9.6, 7.2)):
    x_test = ((x_test + input_min) / (input_max - input_min) * 255.0).astype(int)

    num_sample = y_test.shape[0]
    y_predict = np.zeros(num_sample)
    for i in range(num_sample):
        # encode the input
        sample = x_test[i].flatten().astype(int)
        s = np.sum(np.multiply(F, V[sample]), axis=0)
        if binarize:
            s[s < 0] = -1
            s[s >= 0] = 1
        # perform inference
        output = np.sum(np.multiply(s, C), axis=1)
        # log.show_log (y_test[i], output)
        y_predict[i] = np.argmax(output)

    acc = np.count_nonzero(y_test == y_predict) / num_sample
    confusion_matrix_normalize = confusion_matrix(y_test, y_predict, normalize='true')
    if verbose:
        log.show_log('Evaluation Result\n-----------------------', file=log_file)
        log.show_log(confusion_matrix(y_test, y_predict), file=log_file)
        log.show_log(f'Accuracy = {acc}\n', file=log_file)

        plt.figure(figsize=result_image_size)
        sns.heatmap(confusion_matrix_normalize, annot=True, linewidth=.5)
        if result_img_path is not None:
            plt.savefig(result_img_path, dpi=200, bbox_inches='tight')
        # plt.show()
    return acc, confusion_matrix_normalize

def train_mnist(verbose=True, save_model=True, enable_binarize=True, save_model_dir='result/mnist', vhv_dimension=8, fhv_dimension=64):
    if save_model:
        os.makedirs(save_model_dir, exist_ok=True)
        log_file = open(f'{save_model_dir}/log.txt', 'w')
        result_img_path = f'{save_model_dir}/result.png'
    else:
        log_file = sys.stdout
        result_img_path = None

    x_train, y_train, x_val, y_val = dataset.get_mnist()

    F, V, C = train_ldc(x_train, y_train, x_val, y_val,
                        vhv_dimension=vhv_dimension,
                        fhv_dimension=fhv_dimension,
                        num_feature=784,
                        num_value=256,
                        num_class=10,
                        epochs=50,
                        batch_size=128,
                        initial_learning_rate=0.0001,
                        learning_rate_decay=0.995,
                        weight_decay=0,
                        dropout_probability=0,
                        input_min=0,
                        input_max=255,
                        enable_binarize=enable_binarize,
                        log_file=log_file,
                        verbose=verbose
                        )

    acc, cm = eval_ldc(F, V, C, x_val, y_val, input_min=0, input_max=255, binarize=enable_binarize, log_file=log_file, verbose=verbose,
             result_img_path=result_img_path)

    if save_model:
        log_file.close()
        np.save(f'{save_model_dir}/F.npy', F)
        np.save(f'{save_model_dir}/V.npy', V)
        np.save(f'{save_model_dir}/C.npy', C)

    return acc, cm

def train_ptb_ecg(verbose=True, save_model=True, enable_binarize=True, save_model_dir='result/ptb', vhv_dimension=8, fhv_dimension=64, initial_weight_dir=None):
    if save_model:
        os.makedirs(save_model_dir, exist_ok=True)
        log_file = open(f'{save_model_dir}/log.txt', 'w')
        result_img_path = f'{save_model_dir}/result.png'
    else:
        log_file = sys.stdout
        result_img_path = None

    x_train, y_train, x_val, y_val = dataset.get_ptb_ecg()

    F, V, C = train_ldc(x_train, y_train, x_val, y_val,
                        vhv_dimension=vhv_dimension,
                        fhv_dimension=fhv_dimension,
                        num_feature=187,
                        num_value=256,
                        num_class=2,
                        epochs=50,
                        batch_size=128,
                        initial_learning_rate=0.002,
                        learning_rate_decay=0.975,
                        weight_decay=0.0001,
                        dropout_probability=0,
                        input_min=0,
                        input_max=1,
                        loss_weight=torch.FloatTensor([2.6, 1]).to(ldc_model.device),
                        enable_binarize=enable_binarize,
                        log_file=log_file,
                        verbose=verbose
                        )

    acc, cm = eval_ldc(F, V, C, x_val, y_val, input_min=0, input_max=1, binarize=enable_binarize, log_file=log_file, verbose=verbose,
             result_img_path=result_img_path)

    if save_model:
        log_file.close()
        np.save(f'{save_model_dir}/F.npy', F)
        np.save(f'{save_model_dir}/V.npy', V)
        np.save(f'{save_model_dir}/C.npy', C)

    return acc, cm


def train_ucihar(verbose=True, save_model=True, enable_binarize=True, save_model_dir='result/ucihar', vhv_dimension=8, fhv_dimension=64):
    if save_model:
        os.makedirs(save_model_dir, exist_ok=True)
        log_file = open(f'{save_model_dir}/log.txt', 'w')
        result_img_path = f'{save_model_dir}/result.png'
    else:
        log_file = sys.stdout
        result_img_path = None

    x_train, y_train, x_val, y_val = dataset.get_ucihar()

    F, V, C = train_ldc(x_train, y_train, x_val, y_val,
                        vhv_dimension=vhv_dimension,
                        fhv_dimension=fhv_dimension,
                        num_feature=561,
                        num_value=256,
                        num_class=6,
                        epochs=50,
                        batch_size=128,
                        initial_learning_rate=0.002,
                        learning_rate_decay=0.99,
                        weight_decay=0.0001,
                        dropout_probability=0,
                        input_min=0,
                        input_max=255,
                        log_file=log_file,
                        enable_binarize=enable_binarize,
                        verbose=verbose
                        )

    acc, cm = eval_ldc(F, V, C, x_val, y_val, input_min=0, input_max=255, log_file=log_file, binarize=enable_binarize, verbose=verbose,
             result_img_path=result_img_path)

    if save_model:
        log_file.close()
        np.save(f'{save_model_dir}/F.npy', F)
        np.save(f'{save_model_dir}/V.npy', V)
        np.save(f'{save_model_dir}/C.npy', C)

    return acc, cm


def train_qksd(verbose=True, save_model=True, enable_binarize=True, save_model_dir='result/hey_snapdragon', vhv_dimension=8, fhv_dimension=64):
    if save_model:
        os.makedirs(save_model_dir, exist_ok=True)
        log_file = open(f'{save_model_dir}/log.txt', 'w')
        result_img_path = f'{save_model_dir}/result.png'
    else:
        log_file = sys.stdout
        result_img_path = None

    x_train, y_train, x_val, y_val = dataset.get_qksd()

    F, V, C = train_ldc(x_train, y_train, x_val, y_val,
                        vhv_dimension=vhv_dimension,
                        fhv_dimension=fhv_dimension,
                        num_feature=810,
                        num_value=256,
                        num_class=2,
                        epochs=50,
                        batch_size=128,
                        initial_learning_rate=0.001,
                        learning_rate_decay=0.975,
                        weight_decay=0.0001,
                        dropout_probability=0,
                        input_min=0,
                        input_max=1,
                        enable_binarize=enable_binarize,
                        log_file=log_file,
                        verbose=verbose
                        )

    acc, cm = eval_ldc(F, V, C, x_val, y_val, input_min=0, input_max=1, binarize=enable_binarize, log_file=log_file, verbose=verbose,
             result_img_path=result_img_path, result_image_size=(12.8, 9.6))

    if save_model:
        log_file.close()
        np.save(f'{save_model_dir}/F.npy', F)
        np.save(f'{save_model_dir}/V.npy', V)
        np.save(f'{save_model_dir}/C.npy', C)

    return acc, cm


def train_fsdd(verbose=True, save_model=True, enable_binarize=True, save_model_dir='result/fsdd', vhv_dimension=8, fhv_dimension=64):
    if save_model:
        os.makedirs(save_model_dir, exist_ok=True)
        log_file = open(f'{save_model_dir}/log.txt', 'w')
        result_img_path = f'{save_model_dir}/result.png'
    else:
        log_file = sys.stdout
        result_img_path = None

    x_train, y_train, x_val, y_val = dataset.get_fsdd()

    F, V, C = train_ldc(x_train, y_train, x_val, y_val,
                        vhv_dimension=vhv_dimension,
                        fhv_dimension=fhv_dimension,
                        num_feature=800,
                        num_value=256,
                        num_class=10,
                        epochs=50,
                        batch_size=128,
                        initial_learning_rate=0.005,
                        learning_rate_decay=0.99,
                        weight_decay=0.0001,
                        dropout_probability=0,
                        input_min=0,
                        input_max=1,
                        enable_binarize=enable_binarize,
                        log_file=log_file,
                        verbose=verbose
                        )

    acc, cm = eval_ldc(F, V, C, x_val, y_val, input_min=0, input_max=1, binarize=enable_binarize, log_file=log_file, verbose=verbose,
             result_img_path=result_img_path, result_image_size=(12.8, 9.6))

    if save_model:
        log_file.close()
        np.save(f'{save_model_dir}/F.npy', F)
        np.save(f'{save_model_dir}/V.npy', V)
        np.save(f'{save_model_dir}/C.npy', C)

    return acc, cm


def train_wisdm(verbose=True, save_model=True, enable_binarize=True, save_model_dir='result/widsm', vhv_dimension=8, fhv_dimension=64):
    if save_model:
        os.makedirs(save_model_dir, exist_ok=True)
        log_file = open(f'{save_model_dir}/log.txt', 'w')
        result_img_path = f'{save_model_dir}/result.png'
    else:
        log_file = sys.stdout
        result_img_path = None

    x_train, y_train, x_val, y_val, x_test, y_test = dataset.get_wisdm()
    x_train = (x_train / 80.0) + 0.5
    x_val = (x_val / 80.0) + 0.5
    x_test = (x_test / 80.0) + 0.5
    # print(f'X Range : {np.min(x_train)} {np.max(x_train)}')

    F, V, C = train_ldc(x_train, y_train, x_val, y_val,
                        vhv_dimension=vhv_dimension,
                        fhv_dimension=fhv_dimension,
                        num_feature=144,
                        num_value=256,
                        num_class=4,
                        epochs=50,
                        batch_size=128,
                        initial_learning_rate=0.001,
                        learning_rate_decay=0.975,
                        weight_decay=0.0001,
                        dropout_probability=0,
                        # loss_weight=torch.FloatTensor([1.22, 3.42, 1.85, 1]).to(ldc_model.device),
                        input_min=0,
                        input_max=1,
                        log_file=log_file,
                        enable_binarize=enable_binarize,
                        verbose=verbose
                        )

    acc, cm = eval_ldc(F, V, C, x_test, y_test, input_min=0, input_max=1, log_file=log_file, binarize=enable_binarize, verbose=verbose,
             result_img_path=result_img_path)

    if save_model:
        log_file.close()
        np.save(f'{save_model_dir}/F.npy', F)
        np.save(f'{save_model_dir}/V.npy', V)
        np.save(f'{save_model_dir}/C.npy', C)

    return acc, cm


def train_st_handpose(verbose=True, save_model=True, enable_binarize=True, save_model_dir='result/st_handpose', vhv_dimension=8, fhv_dimension=64):
    if save_model:
        os.makedirs(save_model_dir, exist_ok=True)
        log_file = open(f'{save_model_dir}/log.txt', 'w')
        result_img_path = f'{save_model_dir}/result.png'
    else:
        log_file = sys.stdout
        result_img_path = None

    x_train, y_train, x_val, y_val = dataset.get_st_handpose()
    # x_train = (x_train / 128.0) + 0.5
    # x_val = (x_val / 128.0) + 0.5
    # print(f'X Range : {np.min(x_train)} {np.max(x_train)}')

    F, V, C = train_ldc(x_train, y_train, x_val, y_val,
                        vhv_dimension=vhv_dimension,
                        fhv_dimension=fhv_dimension,
                        num_feature=128,
                        num_value=256,
                        num_class=8,
                        epochs=70,
                        batch_size=128,
                        initial_learning_rate=0.003,
                        learning_rate_decay=0.975,
                        weight_decay=0.0001,
                        dropout_probability=0,
                        # loss_weight=torch.FloatTensor([1.22, 3.42, 1.85, 1]),
                        input_min=0,
                        input_max=1,
                        log_file=log_file,
                        enable_binarize=enable_binarize,
                        verbose=verbose
                        )

    acc, cm = eval_ldc(F, V, C, x_val, y_val, input_min=0, input_max=1, log_file=log_file, binarize=enable_binarize, verbose=verbose,
             result_img_path=result_img_path)

    if save_model:
        log_file.close()
        np.save(f'{save_model_dir}/F.npy', F)
        np.save(f'{save_model_dir}/V.npy', V)
        np.save(f'{save_model_dir}/C.npy', C)

    return acc, cm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training the binary LDC model or the MCU-optimized LDC model')
    parser.add_argument('-d', '--dataset-name', choices=['mnist', 'ptb', 'qksd', 'har', 'fsdd', 'wisdm', 'sthand'], required=True, help='name of the dataset')
    parser.add_argument('-n', '--num-rounds', type=int, default=5, help='number of rounds to train the model')
    parser.add_argument('-dv', '--v-dim', type=int, default=8, help='dimension of the V vector (Dv in the paper)')
    parser.add_argument('-df', '--f-dim', type=int, default=8, help='dimension of the F and C vector (Df in the paper)')
    parser.add_argument('--use-sgn', action='store_true', help='train the binary LDC model (default to MCU-optimized LDC model if --use-sgn is not found)')
    args = parser.parse_args()

    model_fn = {
        'mnist': train_mnist,
        'ptb': train_ptb_ecg,
        'qksd': train_qksd, 
        'har': train_ucihar, 
        'fsdd': train_fsdd, 
        'wisdm': train_wisdm, 
        'sthand': train_st_handpose
    }

    all_accuracy = []
    for i in range(args.num_rounds):
        print (f'Training #{i+1}/{args.num_rounds}...')
        acc, cm = model_fn[args.dataset_name](vhv_dimension=args.v_dim, fhv_dimension=args.f_dim, save_model=True, verbose=True, 
                                              enable_binarize=args.use_sgn, 
                                              save_model_dir=f"result/{args.dataset_name}_d{args.f_dim}{'s' if args.use_sgn else ''}_{i+1}")
        all_accuracy.append(acc)
    print (f'Best accuracy: {max(all_accuracy):.4f}')
