import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import pickle
import time
import cv2
import h5py
from PIL import Image
# import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
torch.backends.cudnn.enabled = False
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import clustering
import models
from util import AverageMeter, Logger, UnifLabelSampler
# from scipy.fft import fft, fftfreq
# from scipy.interpolate import interp1d
# from scipy import integrate


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('data', metavar='DIR', default='fenlei.hdf5',
                        help='path to dataset')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16'], default='vgg16',
                        help='CNN architecture (default: alexnet)')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=40,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=32, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=25000,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    return parser.parse_args()

flag = 'data'
class Mdataset(Dataset):
    def __init__(self, H5File, Index, *args, **kwargs):
        super(Mdataset, self).__init__(*args, **kwargs)
        self.H5File = H5File
        self.Index = Index
        # print(H5File)
        # self.train_transforms = transforms.Compose([
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.Resize((1, 11904)),
        #     transforms.ToTensor(),
        #     # transforms.Normalize(mean=[0], std=[1])
        # ])

    def __getitem__(self, idx):
        sample = self.H5File[self.Index[idx]]
        data = np.float32(sample[flag])
        label = np.float32(sample['label'])
        data = np.asarray(data)
        data0 = data[0]  #0z 1 2
        # print(data0)
        # data1 = data[1]
        # data2 = data[2]
        img = data0.reshape((1, -1))
        # img0 = img[:11904]
        img0 = img.copy()
        img0.resize((1, 640), refcheck=False)


        # np.resize(img0, (1, 11936))  
        # # fenlei 11936.qiepian 23808.qiepian10 3968.qiepian30 11904.qiepian5 1984。 qiepian60 23808 .piepian70 27808
        ## 24h 572384; 6h 143072; 12h 286176  ；3h 71520  ;fft 23808;MH2 3h 71520 s12 fft 10-1000s;MH2/MH1 3h 71520 s14 fft 10-1000s
        # mars 1h 36000;0.5h 17984;3h 36000*3,6h 36000*6,fft 36000
        # 1Hz 10784 (10-1000s)
        # PSE100s 640；A17 100s 11744；A17 1000s 23808; A17 20s(ds2) 1152; A17 500s(ds8) 7360; A17 20s 2336; A17 600s(ds4) 17632; A17 600s(ds8) 8800
        # SHZ 100s 5280;
        # MHZ 100s 640; 200s 1312; 300s 1952; 500s 3296;
        img = torch.Tensor(img0)
        n_a = img
        mean_a = torch.mean(img, dim=1)
        std_a = torch.std(img, dim=1)
        if flag == 'data':
            n_a = img.sub_(mean_a[:, None]).div_(std_a[:, None])  #z-score normalization
        if np.any(np.isnan(n_a.numpy())):
            for i in range(len(n_a)):
                n_a[i] = 0.1
            # print(n_a)
        img = n_a
        
        
        # print(img)

        # print(idx, img)
        return img, label, self.Index[idx]

    def __len__(self):
        return len(self.Index)


def main(args):
    D = args.data
    D0 = str(D).split('/')[-2]
    D = str(D).split('/')[-1].split('.hdf5')[0]
    number_cluster = args.nmb_cluster
    # print(D)
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    print(args.sobel)
    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    model = models.__dict__[args.arch](sobel=args.sobel)
    # print(model)
    # print(model.top_layer)
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    if args.verbose:
        print('create optimizer')
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )

    # define loss function
    if args.verbose:
        print('define loss fun')
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            for key in checkpoint['state_dict']:
                if 'top_layer' in key:
                    del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # creating checkpoint repo
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # creating cluster assignments log
    cluster_log = Logger(os.path.join(args.exp, 'clusters'))

    # preprocessing of data
    # normalize = transforms.Normalize(mean=[0],
    #                                  std=[1])
    # tra = [transforms.Resize([1, 11904]),
    #        # transforms.CenterCrop([1, 11926//32*32]),
    #        transforms.ToTensor(),
    #     #    normalize
    #        ]

    # load the data
    end = time.time()
    images = []  # keys' name
    def get_data_items(name, obj):
        if len(name.split('/')) == 1:
            images.append(name)
    DDataset = h5py.File(args.data, 'r')
    # print(DDataset)
    # print(data0)
    DDataset.visititems(get_data_items)
    dataset = Mdataset(DDataset, images)
    print(len(dataset))
    # for i in range(1440):
    #     if np.any(np.isnan(dataset[i][0].numpy())):
    #         # percent = np.isnan(float(dataset[i][0])).sum().item() / float(np.size(float(dataset[i][0]))) * 100
    #         print(i, dataset[i])
    # for i in [154, 155]:
    #     # percent = np.isnan(float(dataset[i][0])).sum().item() / float(np.size(float(dataset[i][0]))) * 100
    #     mean_a = torch.mean(dataset[i][0], dim=1)
    #     std_a = torch.std(dataset[i][0], dim=1)+1e-4
    #     n_a = dataset[i][0].sub_(mean_a[:, None]).div_(std_a[:, None])
    #     print(i, dataset[i])
    #     print(mean_a, std_a, n_a)
    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))
        # print(dataset)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                            #  num_workers=args.workers,
                                             pin_memory=True)

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)

    # training convnet with DeepCluster
    cluster_loss = []
    cnn_loss = []
    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()

        # remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # get the features for the whole dataset
        features = compute_features(dataloader, model, len(dataset))
        print(features.shape)
        # if np.any(np.isnan(features)):
        #     percent = np.isnan(features).sum().item() / float(np.size(features)) * 100
        #     # npdata[np.isnan(npdata)] = 0.
        #     print(percent)

        # cluster the features
        if args.verbose:
            print('Cluster the features')
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

        # assign pseudo-labels
        if args.verbose:
            print('Assign pseudo labels')
            # print('images_lists:{0};;;{1}'.format(deepcluster.images_lists, enumerate(deepcluster.images_lists)))
        train_dataset = clustering.cluster_assign(deepcluster.images_lists,
                                                  dataset)

        # uniformly sample per target
        if args.verbose:
            print('uniformly sample per target')
        sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)),
                                   deepcluster.images_lists)
        if args.verbose:
            print('train_dataloader')
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch,
            # num_workers=args.workers,
            sampler=sampler,
            pin_memory=True,
        )
        # if args.verbose:  可改为保存每轮的聚类结果
        #     print('images_lists:{0}'.format(deepcluster.images_lists))

        # set last fully connected layer
        if args.verbose:
            print('set last fully connected layer')
        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.cuda()

        # train network with clusters as pseudo-labels
        end = time.time()
        if args.verbose:
            print('train network with clusters as pseudo-labels')
        print(train_dataloader)    
        loss = train(train_dataloader, model, criterion, optimizer, epoch)

        # print log
        if args.verbose:
            # print(type(clustering_loss), clustering_loss, type(loss), loss)
            print('###### Epoch [{0}] ###### \n'.format(epoch))
            print('Time: {0:.3f} s\n'.format(time.time() - end))
            print('Clustering loss: {0:.3f} \n'.format(clustering_loss))
            print('ConvNet loss: {0:.3f}'.format(loss))
            cluster_loss.append(clustering_loss)
            c_loss = loss.cpu().numpy()
            cnn_loss.append(c_loss)

                  
            try:
                nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(deepcluster.images_lists),
                    clustering.arrange_clustering(cluster_log.data[-1])
                )
                print('NMI against previous assignment: {0:.3f}'.format(nmi))
            except IndexError:
                pass
            print('####################### \n')
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                   os.path.join(args.exp, 'checkpoint.pth.tar'))

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)
        dir2 = f'/home/liuxin/deepcluster/moonquake/deepcluster_glitch/visual/cluster{number_cluster}/{D0}/{D}/{flag}_seed{args.seed}/cluster_list/'
        if not os.path.exists(dir2):
            os.makedirs(dir2)
        # plt.savefig(dir2+'{0}_{1}.png'.format(epoch, i))
        plt.close()
        filename = "cluster_list.txt"
        file = open(dir2 + str(epoch) + filename,'w')
        for i in range(len(deepcluster.images_lists)):  
            s = deepcluster.images_lists[i]#去除[],这两行按数据不同，可以选择
            # print(s)
            # s = s + ' '   #去除单引号，逗号，每行末尾追加换行符
            file.write(str(s))
            file.write('\n')
        file.close()
    if args.verbose:
        dir1 = f'/home/liuxin/deepcluster/moonquake/deepcluster_glitch/pic/cluster{number_cluster}/{D0}/{D}/{flag}_seed{args.seed}/'
        if not os.path.exists(dir1):
           os.makedirs(dir1)
        plt.plot(cluster_loss, color='k')
        plt.title('cluster_loss')
        plt.xlabel('niter')
        plt.ylabel('kmeans_loss')
        plt.savefig(dir1+'kmeans_loss.png')
        plt.close()
        # print('cluster_loss:', cluster_loss)
        # print('cnn_loss:', cnn_loss)
        plt.plot(cnn_loss, color='k')
        plt.title('cnn_loss')
        plt.xlabel('epoch')
        plt.ylabel('cnn_loss')
        plt.savefig(dir1+'cnn_loss.png')
        plt.close()
       
        # for i in range(args.nmb_cluster):
        #     l = len(deepcluster.images_lists[i])
        #     j = 0
        #     for chdx in deepcluster.images_lists[i]:
        #         data = dataset[chdx][0].numpy().squeeze()
        #         label = dataset[chdx][2]
        #         # print(chdx, label)
        #         data = data - np.mean(data)
        #         data = data / np.max(np.abs(data))
        #         plt.figure(i)
        #         # plt.subplot(l,1,j)
        #         plt.plot(data+2*j, color='k')
        #         j += 1
        #     plt.title(deepcluster.images_lists[i])
        #     dir2 = './visual/qiepian10/001/'
        #     if not os.path.exists(dir2):
        #         os.makedirs(dir2)
        #     # plt.savefig(dir2+'{0}_{1}.png'.format(epoch, i))
        #     plt.close()
        dir3 = f'/home/liuxin/deepcluster/moonquake/deepcluster_glitch/visual/cluster{number_cluster}/{D0}/{D}/{flag}_seed{args.seed}/'
        filename = "cluster_list.txt"
        file = open(dir3 + filename,'w')
        for i in range(len(deepcluster.images_lists)):  
            s = deepcluster.images_lists[i]#去除[],这两行按数据不同，可以选择
            # print(s)
            # s = s + ' '   #去除单引号，逗号，每行末尾追加换行符
            file.write(str(s))
            file.write('\n')
        file.close()
            


def train(loader, model, crit, opt, epoch):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=10**args.wd,
    )

    end = time.time()
    for i, data in enumerate(loader):
        data_time.update(time.time() - end)
        input_tensor= data[0]
        target = data[1]
        # if np.any(np.isnan(data[0].numpy())):
        #     percent = np.isnan(data[0].numpy()).sum().item() / float(np.size(data[0].numpy())) * 100
        #     print(percent, i, data[0])

        # save checkpoint
        n = len(loader) * epoch + i
        if n % args.checkpoints == 0:
            path = os.path.join(
                args.exp,
                'checkpoints',
                'checkpoint_' + str(n / args.checkpoints) + '.pth.tar',
            )
            if args.verbose:
                print('Save checkpoint at: {0}'.format(path))
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict()
            }, path)

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var.squeeze(2))
        # if np.any(np.isnan(output.data.numpy())):
        #     percent = np.isnan(output.data.numpy()).sum().item() / float(np.size(output.data.numpy())) * 100
        #     print(percent, i, output.data)
        loss = crit(output, target_var)
        # print(input_var)

        # record loss
        losses.update(loss.data, input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)    #设置剪裁阈值为5
        opt.step()
        optimizer_tl.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 100) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))

    return losses.avg

def compute_features(dataloader, model, N):
    # global features
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, data in enumerate(dataloader):
        input_tensor = data[0]
        # print(input_tensor[0],len(input_tensor[0][0]))
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_tensor.cuda())
            # print(input_var)
        aux = model(input_var.squeeze(2)).data.cpu().numpy()
        # aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
            if np.any(np.isnan(aux)):
                percent = np.isnan(aux).sum().item() / float(np.size(aux)) * 100
                # npdata[np.isnan(npdata)] = 0.
                print(percent, i, data)
            
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux
            if np.any(np.isnan(aux)):
                percent = np.isnan(aux).sum().item() / float(np.size(aux)) * 100
                # npdata[np.isnan(npdata)] = 0.
                print(percent, i, data)
           
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 100) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features


if __name__ == '__main__':
    start = time.time()
    args = parse_args()
    main(args)
    end = time.time() - start
    print('时间共计：', end)
