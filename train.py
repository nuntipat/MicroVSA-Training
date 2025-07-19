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
import uuid

import train_config
import ldc_model
import log
from tqdm import trange

def train_ldc(x_train, y_train, x_val, y_val,
              vhv_dimension, fhv_dimension, num_feature, num_value, num_class,
              epochs=50, batch_size=128, initial_learning_rate=0.001, learning_rate_decay=0,
              weight_decay=0.0001, dropout_probability=0, input_min=0, input_max=1, loss_weight=None,
              enable_binarize=True, prune_weight_level=0, prune_input_level=0, prune_class_level=0, reduction_method='add', lut_size=6, extra_lut=0,
              verbose=True, log_file=None):
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
        log.show_log(f"    loss_weight = {loss_weight}", file=log_file)
        log.show_log(f"    prune_weight_level = {prune_weight_level}", file=log_file)
        log.show_log(f"    prune_input_level = {prune_input_level}", file=log_file)
        log.show_log(f"    prune_class_level = {prune_class_level}", file=log_file)
        log.show_log(f"    reduction_method = {reduction_method}", file=log_file)
        log.show_log(f"    lut_size = {lut_size}", file=log_file)
        log.show_log(f"    extra_lut = {extra_lut}", file=log_file)
        log.show_log("", file=log_file)

    train_set = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    testloader = DataLoader(test_set, batch_size=batch_size)

    device = ldc_model.device
    model = ldc_model.ValFeaClaSearchNet(num_feature, num_value, fhv_dimension, vhv_dimension, num_class,
                                   dropout_probability, enable_binarize, lut_size, extra_lut, reduction_method).to(device)
    if loss_weight is not None:
        loss_weight = torch.FloatTensor(loss_weight).to(ldc_model.device)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    begin_save_epoch = 12
    session_id = uuid.uuid4().hex

    best_acc = 0
    for t in trange(1, epochs+1):
        if verbose:
            log.show_log(f"Epoch {t}\n-------------------------------", file=log_file)
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

            if t > begin_save_epoch and enable_pruning:
                if prune_input_level != 0 and prune_weight_level != 0:
                    model.fc1.prune_input_and_weight(prune_input_level, prune_weight_level)
                elif prune_input_level != 0:
                    model.fc1.prune_input(prune_input_level)
                elif prune_weight_level != 0:
                    model.fc1.prune_weight(prune_weight_level)
                
                if t > begin_save_epoch and prune_class_level > 0:
                    model.fc2.prune_weight(prune_class_level)

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
            log.show_log(f"Test Error: \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>8f} FreezeRate: {freeze_percent}", file=log_file)

        if (correct > best_acc) and ((not enable_pruning) or (t > begin_save_epoch)):
            if verbose:
                log.show_log('Saving model state...', file=log_file)
            torch.save(model.state_dict(), f".model_{session_id}.pth")
            best_acc = correct

        # optimizer.param_groups[0]['lr'] *= learning_rate_decay ** t
        if verbose:
            log.show_log(f'{time.time() - start_time} sec\n', file=log_file)

    if verbose:
        log.show_log('Summary\n-----------------------', file=log_file)
        log.show_log(f'Best accuracy: {(100 * best_acc):>0.1f}%\n', file=log_file)

    # load the best parameter value saved during training

    model.load_state_dict(torch.load(f".model_{session_id}.pth"))
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
    featureMemory = torch.Tensor.cpu(torch.sign(model.fc1.weight * model.fc1.mask)).detach().numpy()

    valueMemory = np.zeros((num_value, fhv_dimension))
    valueBox = model.fc0
    base = torch.sign(valueBox(torch.linspace(input_min, input_max, num_value).reshape(num_value, 1).to(device)))
    base = torch.Tensor.cpu(base).detach().numpy()
    for i in range(num_value):
        for j in range(fhv_dimension):
            valueMemory[i, j] = base[i, j % vhv_dimension]
            if valueMemory[i, j] == 0:
                log.show_log(f'valueMemory: ({i}, {j})', file=log_file)

    class_weights = torch.Tensor.cpu(torch.sign(model.fc2.weight * model.fc2.mask)).detach().numpy()
    associativeMemory = class_weights

    if reduction_method == 'lut':
        permute_index = model.fc1.permute.cpu().detach().numpy()
        lut_weights = [w.cpu().detach().numpy() for w in model.fc1.lut_weights]
    else:
        permute_index = None
        lut_weights = None

    return featureMemory, valueMemory, associativeMemory, permute_index, lut_weights

def create_lut_from_weight(weight, mask, num_pad_row=0, lut_size=6):
    num_row, fhv_dimension = weight.shape
    if mask is not None:
        mask_pad = np.pad(mask, ((0, num_row - mask.shape[0]), (0, 0)))

    inputs = np.array([list(np.binary_repr(n, lut_size)) for n in range(pow(2, lut_size))], dtype=int)
    inputs[inputs == 1] = -1
    inputs[inputs == 0] = 1

    num_group = num_row//lut_size
    lut = np.empty((num_group, fhv_dimension, pow(2, lut_size)), dtype=int)
    for i in range(fhv_dimension):
        for j in range(0, num_group):
            w = weight[(j*lut_size):(j*lut_size)+lut_size, i]
            if mask is not None:
                w = w * mask_pad[(j*lut_size):(j*lut_size)+lut_size, i]
            if j == num_group - 1:
                w[-num_pad_row:] = 0
            s = np.sum(inputs * w, axis=1)
            lut[j, i] = np.sign(s)
    return lut

def process_lut(input, lut, lut_size=6):
    row, col, _ = lut.shape
    input = input.copy()
    input[input == 1] = 0
    input[input == -1] = 1
    input_pad = np.pad(input, ((0, (row * lut_size)-input.shape[0]), (0, 0)))

    power_of_two = np.power([2], np.arange(0, lut_size)[::-1]).astype(int).reshape(-1, 1)

    out = np.zeros((row, col), dtype=int)
    for j in range(row):
        t = input_pad[(j*lut_size):(j*lut_size)+lut_size, :]
        index = np.sum(t * power_of_two, axis=0).astype(int)
        out[j] = lut[j][np.arange(index.shape[0]), index]

    return out

def eval_ldc(F, V, C, P, L, x_test, y_test, reduction_method, input_min, input_max, lut_size=6, binarize=True, verbose=True, log_file=None,
             result_img_path='result.png', result_image_size=(9.6, 7.2)):
    x_test = ((x_test + input_min) / (input_max - input_min) * 255.0).astype(int)

    mask = np.ones_like(F, dtype=int)
    mask[F == 0] = 0
    
    if reduction_method == 'lut':
        luts = []
        num_lut_output = F.shape[0]
        for i, w in enumerate(L):
            num_lut_input = w.shape[0]
            num_pad_row = num_lut_input - num_lut_output
            luts.append(create_lut_from_weight(w, mask if i == 0 else None, num_pad_row, lut_size))
            num_lut_output = num_lut_input // lut_size

    num_sample = y_test.shape[0]
    y_predict = np.zeros(num_sample)
    # multi = 0
    for i in trange(num_sample):
        # encode the input
        sample = x_test[i].flatten().astype(int)
        if reduction_method == 'lut':
            vf = np.multiply(F, V[sample[P]])
            for lut in luts:
                vf = process_lut(vf, lut, lut_size)
        else:
            vf = np.multiply(F, V[sample])
        s = np.sum(vf, axis=0)
        if binarize:
            s[s < 0] = -1
            s[s >= 0] = 1
        # perform inference
        output = np.sum(np.multiply(s, C), axis=1)
        # log.show_log (y_test[i], output)
        # if np.flatnonzero(output == np.max(output)).shape[0] > 1:
        #     multi += 1
            # print (output)
        y_predict[i] = np.argmax(output)

    acc = np.count_nonzero(y_test == y_predict) / num_sample
    confusion_matrix_normalize = confusion_matrix(y_test, y_predict, normalize='true')
    if verbose:
        log.show_log('Evaluation Result\n-----------------------', file=log_file)
        log.show_log(confusion_matrix(y_test, y_predict), file=log_file)
        log.show_log(f'Accuracy = {acc}\n', file=log_file)
        # log.show_log(f'Multi = {multi}\n', file=log_file)

        plt.figure(figsize=result_image_size)
        sns.heatmap(confusion_matrix_normalize, annot=True, linewidth=.5)
        if result_img_path is not None:
            plt.savefig(result_img_path, dpi=200, bbox_inches='tight')
        # plt.show()
    return acc, confusion_matrix_normalize

def train(dataset_fn, num_feature, num_class, num_value=256, vhv_dimension=8, fhv_dimension=64, enable_binarize=True, input_min=0, input_max=255,
          epochs=50, batch_size=128, learning_rate=0.001, loss_weight=None, has_val=False,
          prune_weight_level=0.0, prune_input_level=0.0, prune_class_level=0.0, 
          reduction_method='add', lut_size=6, extra_lut=0,
          verbose=True, save_model=True, save_model_dir='result/mnist'):
    if save_model:
        os.makedirs(save_model_dir, exist_ok=True)
        log_file = open(f'{save_model_dir}/log.txt', 'w')
        result_img_path = f'{save_model_dir}/result.png'
    else:
        log_file = sys.stdout
        result_img_path = None

    if has_val:
        x_train, y_train, x_val, y_val, x_test, y_test = dataset_fn()
    else:
        x_train, y_train, x_val, y_val = dataset_fn()
        x_test, y_test = x_val, y_val

    F, V, C, P, L = train_ldc(x_train, y_train, x_val, y_val,
                        vhv_dimension=vhv_dimension,
                        fhv_dimension=fhv_dimension,
                        num_feature=num_feature,
                        num_value=num_value,
                        num_class=num_class,
                        epochs=epochs,
                        batch_size=batch_size,
                        initial_learning_rate=learning_rate,
                        # learning_rate_decay=0,
                        # weight_decay=0,
                        # dropout_probability=0,
                        input_min=input_min,
                        input_max=input_max,
                        loss_weight=loss_weight,
                        enable_binarize=enable_binarize,
                        # enable_pruning=, 
                        # prune_method=, 
                        prune_weight_level=prune_weight_level,
                        prune_input_level=prune_input_level,
                        prune_class_level=prune_class_level,
                        reduction_method=reduction_method,
                        lut_size=lut_size,
                        extra_lut=extra_lut,
                        log_file=log_file,
                        verbose=verbose
                        )

    acc, cm = eval_ldc(F, V, C, P, L, x_test, y_test, reduction_method, input_min=input_min, input_max=input_max, lut_size=lut_size, binarize=enable_binarize, log_file=log_file, verbose=verbose,
             result_img_path=result_img_path)

    if save_model:
        log_file.close()
        np.save(f'{save_model_dir}/F.npy', F)
        np.save(f'{save_model_dir}/V.npy', V)
        np.save(f'{save_model_dir}/C.npy', C)
        if reduction_method == 'lut':
            np.save(f'{save_model_dir}/P.npy', P)
            np.savez(f'{save_model_dir}/L.npz', *L)

    return acc, cm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training the binary LDC model or the MCU-optimized LDC model')
    parser.add_argument('-d', '--dataset-name', choices=train_config.model_config.keys(), required=True, help='name of the dataset')
    parser.add_argument('-n', '--num-rounds', type=int, default=5, help='number of rounds to train the model')
    parser.add_argument('-dv', '--v-dim', type=int, default=8, help='dimension of the V vector (Dv in the paper)')
    parser.add_argument('-df', '--f-dim', type=int, default=8, help='dimension of the F and C vector (Df in the paper)')
    parser.add_argument('-pcl', '--prune-class-level', type=float, default=0.0, help='level of pruning (class) from 0.0-1.0')
    parser.add_argument('-pwl', '--prune-weight-level', type=float, default=0.0, help='level of pruning (weight) from 0.0-1.0')
    parser.add_argument('-pnl', '--prune-input-level', type=float, default=0.0, help='level of pruning (input) from 0.0-1.0')
    parser.add_argument('-pm', '--prune-method', choices=['input', 'weight', 'both'], default='weight', help='pruning method (input - remove some input features, weight - remove some cells from the F matrix)')
    parser.add_argument('-rm', '--reduction-method', choices=['adder', 'lut'], default='adder', help='reduction method (adder - compute S with dot-product (Original LDC, MicroVSA), lut - use LUT tree instead of the adder (VSALUT))')
    parser.add_argument('-ls', '--lut-size', type=int, default=6, help='maximum number of input of the LUT')
    parser.add_argument('-el', '--extra-lut', type=float, default=0.0, help='percent of extra (redundant) lut from 0.0-1.0')
    parser.add_argument('--use-sgn', action='store_true', help='train the binary LDC model (default to MCU-optimized LDC model if --use-sgn is not found)')
    parser.add_argument('--no-save', action='store_false', help='do not save the model')
    args = parser.parse_args()

    enable_pruning = (args.prune_weight_level != 0.0) or (args.prune_input_level != 0.0)

    traing_config = {
        'vhv_dimension': args.v_dim, 
        'fhv_dimension': args.f_dim,  
        'enable_binarize': args.use_sgn,
        'prune_weight_level': args.prune_weight_level,
        'prune_input_level': args.prune_input_level,
        'prune_class_level': args.prune_class_level, 
        'reduction_method': args.reduction_method,
        'lut_size': args.lut_size,
        'extra_lut': args.extra_lut,
        'verbose': True,
        'save_model': args.no_save,
    }

    all_accuracy = []
    for i in trange(args.num_rounds):
        # print (f'Training #{i+1}/{args.num_rounds}...')
        acc, cm = train(**{
            **train_config.model_config[args.dataset_name], 
            **traing_config, 
            'save_model_dir': f"result/{args.dataset_name}_d{args.f_dim}{'s' if args.use_sgn else ''}{f'_pn{args.prune_input_level}' if enable_pruning and args.prune_method != 'weight' else ''}{f'_pf{args.prune_weight_level}' if enable_pruning and args.prune_method != 'input' else ''}{f'_pc{args.prune_class_level}' if args.prune_class_level > 0 else ''}{f'_lr{args.lut_size}' if args.reduction_method == 'lut' else ''}{f'_el{args.extra_lut}' if args.extra_lut > 0 else ''}_{i+1}"
        })
        print (acc)
        all_accuracy.append(acc)
