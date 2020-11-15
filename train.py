import argparse
import copy
import cv2
import os
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
import yaml

import multipoint.datasets as datasets
import multipoint.models as models
import multipoint.utils as utils
import multipoint.utils.losses as losses

def main():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('-y', '--yaml-config', required=True, help='YAML config file')
    parser.add_argument('-w', '--weight-file', help='File containing the weights to initialize the weights')
    args = parser.parse_args()

    with open(args.yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # create output directory if it does not exist yet
    if not os.path.isdir(str(config['training']['output_directory'])):
        os.makedirs(str(config['training']['output_directory']))

    # dump the params
    with open(os.path.join(config['training']['output_directory'], 'params.yaml'), 'wt') as fh:
        yaml.safe_dump({'model': config['model'], 'loss': config['loss'], 'training': config['training'], 'dataset': config['dataset']}, fh)

    # check training device
    device = torch.device("cpu")
    if config['training']['allow_gpu']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Training on device: {}'.format(device))

    # dataset
    dataset_class = getattr(datasets, config['dataset']['type'])
    dataset = dataset_class(config['dataset'])
    loader_trainset = torch.utils.data.DataLoader(dataset, batch_size=config['training']['batchsize'],
                                              shuffle=True, num_workers=config['training']['num_worker'])

    if config['training']['validation']['compute_validation_loss']:
        val_config = copy.copy(config['dataset'])
        val_config['filename'] = config['training']['validation']['filename']
        val_config['keypoints_filename'] = config['training']['validation']['keypoints']
        validation_dataset = dataset_class(val_config)
        loader_validationset = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['batchsize'],
                                                           shuffle=False, num_workers=config['training']['num_worker'])

    # network
    net = getattr(models, config['model']['type'])(config['model'])

    start_epoch = 0
    if args.weight_file is not None:
        try:
            start_epoch = int(os.path.split(args.weight_file)[-1].split('.')[0][1:])
        except ValueError:
            pass
        weights = torch.load(args.weight_file, map_location=torch.device('cpu'))
        weights = utils.fix_model_weigth_keys(weights)
        net.load_state_dict(weights)

    if config['training']['allow_gpu'] and (torch.cuda.device_count() > 1):
        print("Using ", torch.cuda.device_count(), " GPUs to train the model")
        net = torch.nn.DataParallel(net)

    net.to(device)

    # loss
    loss_fn = getattr(losses, config['loss']['type'])(config['loss'])

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=float(config['training']['learningrate']))

    # initialize writer if requested
    if config['training']['use_writer']:
        # initialize the tensorboard writer
        writer = SummaryWriter(os.path.join(config['training']['output_directory'], 'learningcurve'))

    num_iter = len(loader_trainset) * (config['training']['n_epochs'] - start_epoch)
    with tqdm(total=num_iter) as pbar:
        for epoch in range(start_epoch, config['training']['n_epochs']):  # loop over the dataset multiple times
            epoch_loss = 0.0
            epoch_loss_components = {}
            epoch_val_loss = 0.0
            epoch_val_loss_components = {}

            for i, data in enumerate(loader_trainset, 0):
                # move data to correct device
                data = utils.data_to_device(data, device)

                # zero the parameter gradients
                optimizer.zero_grad()

                if dataset.returns_pair():
                    # predict
                    pred_optical = net(data['optical'])
                    pred_thermal = net(data['thermal'])

                    # compute the loss
                    loss, loss_components = loss_fn(pred_optical, data['optical'], pred_thermal, data['thermal'])

                else:
                    # predict
                    pred = net(data)

                    # compute the loss
                    loss, loss_components = loss_fn(pred, data)

                # update params
                loss.backward()
                optimizer.step()

                # update epoch losses
                epoch_loss_components = update_loss_components(epoch_loss_components, loss_components)

                epoch_loss += loss.item()
                pbar.set_description('epoch {}/{}, batch {}/{} - epoch_loss: {}, batch_loss: {}'.format(
                    epoch+1, config['training']['n_epochs'],
                    i+1, len(loader_trainset),
                    epoch_loss / (i + 1),
                    loss.item()))
                pbar.update(1)

            if config['training']['validation']['compute_validation_loss']:
                if epoch % config['training']['validation']['every_nth_epoch'] == 0:
                    with torch.no_grad():
                        for k, data in enumerate(loader_validationset, 0):
                            data = utils.data_to_device(data, device)

                            if dataset.returns_pair():
                                pred_optical = net(data['optical'])
                                pred_thermal = net(data['thermal'])
                                loss, loss_components = loss_fn(pred_optical,
                                                                data['optical'],
                                                                pred_thermal,
                                                                data['thermal'])

                            else:
                                pred = net(data)
                                loss, loss_components = loss_fn(pred, data)

                            epoch_val_loss_components = update_loss_components(epoch_val_loss_components, loss_components)
                            epoch_val_loss += loss.item()

                        if config['training']['use_writer']:
                            writer.add_scalar('validation_loss', epoch_val_loss / len(loader_validationset), epoch + 1)
                            for key in epoch_val_loss_components.keys():
                                writer.add_scalar('validation_loss/'+key, epoch_val_loss_components[key] / len(loader_validationset), epoch + 1)


            # store the loss every epoch
            if config['training']['use_writer']:
                writer.add_scalar('loss', epoch_loss / len(loader_trainset), epoch + 1)
                for key in epoch_loss_components.keys():
                    writer.add_scalar('loss/'+key, epoch_loss_components[key] / len(loader_trainset), epoch + 1)

            # save model every save_model_every_n_epoch epochs
            if ((epoch + 1) % config['training']['save_every_n_epoch'] == 0) and config['training']['save_every_n_epoch'] > 0:
                try:
                    state_dict = net.module.state_dict() #for when the model is trained on multi-gpu
                except AttributeError:
                    state_dict = net.state_dict()

                torch.save(state_dict, os.path.join(config['training']['output_directory'], 'e{}.model'.format(epoch + 1)))

    try:
        state_dict = net.module.state_dict() #for when the model is trained on multi-gpu
    except AttributeError:
        state_dict = net.state_dict()
    torch.save(state_dict, os.path.join(config['training']['output_directory'], 'latest.model'.format(epoch + 1)))

def update_loss_components(epoch_loss_components, loss_components):
    for key in loss_components.keys():
        if key in epoch_loss_components.keys():
            epoch_loss_components[key] += loss_components[key]
        else:
            epoch_loss_components[key] = loss_components[key]
    return epoch_loss_components

if __name__ == "__main__":
    main()
