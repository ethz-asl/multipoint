import argparse
import h5py
import os
from tqdm import tqdm
import torch
import yaml

import multipoint.datasets as datasets
import multipoint.models as models
import multipoint.utils as utils

def main():
    parser = argparse.ArgumentParser(description='Script to export the keypoints for images in a dataset using a base detector')
    parser.add_argument('-y', '--yaml-config', default='configs/config_export_keypoints.yaml', help='YAML config file')
    parser.add_argument('-o', '--output_file', required=True, help='Output file name')
    parser.add_argument('-m', '--model-dir', default='model_weights/surf', help='Directory of the model')
    parser.add_argument('-v', '--version', default='none', help='Model version (name of the .model file)')
    parser.add_argument('-snms', '--single-nms', action='store_true', help='Do the nms calculation for each sample separately')
    parser.add_argument('-skip', dest='skip_processed', action='store_true', help='Skip already processed samples')

    args = parser.parse_args()

    with open(args.yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(args.model_dir, 'params.yaml'), 'r') as f:
        # overwrite the model params
        config['model'] = yaml.load(f, Loader=yaml.FullLoader)['model']

    # create output file
    output_file = h5py.File(args.output_file)

    # check device
    device = torch.device("cpu")
    if config['prediction']['allow_gpu']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Predicting on device: {}'.format(device))

    # dataset
    dataset = getattr(datasets, config['dataset']['type'])(config['dataset'])
    loader_dataset = torch.utils.data.DataLoader(dataset, batch_size=config['prediction']['batchsize'],
                                                 shuffle=False, num_workers=config['prediction']['num_worker'])

    # network
    net = getattr(models, config['model']['type'])(config['model'])
    if args.version != 'none':
        weights = torch.load(os.path.join(args.model_dir, args.version + '.model'), map_location=torch.device('cpu'))
        weights = utils.fix_model_weigth_keys(weights)
        net.load_state_dict(weights)

    # multi gpu prediction
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    print("Using", torch.cuda.device_count(), "GPUs for prediction")

    # move net to the right device
    net.to(device)

    # put the network into the evaluation mode
    net.eval()

    with torch.no_grad():
        for batch in tqdm(loader_dataset):
            if args.skip_processed:
                all_processed = True
                for name in batch['name']:
                    all_processed = all_processed and (name in output_file.keys())

                if all_processed:
                    continue

            # move data to device
            batch = utils.data_to_device(batch, device)

            # compute the homographic adaptation probabilities
            if dataset.returns_pair():
                prob_ha = utils.homographic_adaptation_multispectral(batch, net, config['prediction']['homographic_adaptation'])
            else:
                prob_ha = utils.homographic_adaptation(batch, net, config['prediction']['homographic_adaptation'])

            # compute the nms probablity
            if config['prediction']['nms'] > 0:
                if args.single_nms:
                    for i, sample in enumerate(prob_ha.split(1)):
                        prob_ha[i, 0] = utils.box_nms(sample.squeeze(),
                                                      config['prediction']['nms'],
                                                      config['prediction']['detection_threshold'],
                                                      keep_top_k=config['prediction']['topk'],
                                                      on_cpu=config['prediction']['cpu_nms'])
                else:
                    prob_ha = utils.box_nms(prob_ha,
                                            config['prediction']['nms'],
                                            config['prediction']['detection_threshold'],
                                            keep_top_k=config['prediction']['topk'],
                                            on_cpu=config['prediction']['cpu_nms'])

            for name, prob in zip(batch['name'], prob_ha.split(1)):
                if not (args.skip_processed and (name in output_file.keys())):
                    pred = torch.nonzero((prob.squeeze() > config['prediction']['detection_threshold']).float())
                    # save the data
                    output_file.create_group(name)
                    output_file[name].create_dataset('keypoints', data=pred.cpu().numpy())

    output_file.close()

if __name__ == "__main__":
    main()
