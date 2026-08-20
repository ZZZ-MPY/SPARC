import os
import json
import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

                                        
class BufferDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None, loader=default_loader):
           
             
                                                   
                                                        
                                          
           
        self.samples = samples
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

                                                 
class MappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, class_map):
           
                                                                  
             
                                                                          
                                                              
           
        self.dataset = dataset
        self.class_map = class_map
        
    def __getitem__(self, index):
        img, target = self.dataset[index]
                                         
        if target in self.class_map:
            return img, self.class_map[target]
        else:
                                                                            
            raise ValueError(f"Label {target} found in dataset but not in class_map!")
            
    def __len__(self):
        return len(self.dataset)

                                              
class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))


def build_dataset(is_train, args, current_classes=None, class_to_idx_mapping=None):
       
         
                                                                 
                             
                                                                                    
                                                                                   
       
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    
    elif args.data_set in ['IMNET', 'NWPU', 'UCM', 'AID', 'RSI-CB256']:
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000 if args.data_set == 'IMNET' else len(dataset.classes)
    
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    else:
        raise ValueError(f"Unknown dataset: {args.data_set}")

                                            
    if current_classes is not None:
                                     
        current_classes_set = set(current_classes)
        
                                 
        targets = None
        if hasattr(dataset, 'targets'):
                                         
            targets = dataset.targets
        elif hasattr(dataset, 'samples'):
                                                                           
            targets = [s[1] for s in dataset.samples]
            
                           
        if targets is not None:
            indices = [i for i, t in enumerate(targets) if t in current_classes_set]
            
            if len(indices) == 0:
                print(f"[Warning] No samples found for classes {current_classes}. Check dataset/paths.")
            
                                                                         
            dataset = torch.utils.data.Subset(dataset, indices)
        else:
            print("[Warning] Could not find targets in dataset, skipping class filtering!")

                                                        
    if class_to_idx_mapping is not None:
        dataset = MappedDataset(dataset, class_to_idx_mapping)
            
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)