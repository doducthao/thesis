from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
import numpy as np

class MyDataset(Dataset):
    def __init__(self, images, labels):
        super(MyDataset, self).__init__()
        self.images = images
        self.labels = labels
    
    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return img, target
    
    def __len__(self):
        return len(self.images)
    
def get_dataset(indices, raw_loader):
    images, labels = [], []
    for idx in indices:
        image, label = raw_loader[idx]
        images.append(image)
        labels.append(label)
    
    images = torch.stack(images, 0) # shape [100, 1, 28, 28]
    labels = torch.from_numpy(np.array(labels, dtype=np.int64)).squeeze() # torch.Size([100])
    return images, labels

def transform_func(input_size):
    transform = transforms.Compose([
                                    transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])
    return transform 

# make dataloader and save indexes of choosen data
def dataloader(dataset, args, split='train'):
    transform = transform_func(args.input_size)

    if dataset == 'mnist':
        training_set = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)

        indices = np.arange(len(training_set))
        np.random.shuffle(indices)

        mask = np.zeros(shape=indices.shape, dtype=np.bool)
        labels = np.array([training_set[i][1] for i in indices], dtype = np.int64)

        for i in range(10):
            mask[np.where(labels[indices] == i)[0][: args.num_labels // 10]] = True # choosen labeled data
        
        labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
        print('Labeled size:', labeled_indices.shape[0], '|| Unlabeled size:', unlabeled_indices.shape[0])

        # save labeled indices
        path_save = 'labeled_data_indices/' + args.gan_type + '_' + str(args.num_labels) + '_' + str(args.lrC) + '.txt'
        np.savetxt(path_save, labeled_indices.flatten(), fmt="%i", delimiter=',')  

        labeled_set = get_dataset(labeled_indices, training_set)
        unlabeled_set = get_dataset(unlabeled_indices, training_set)
        labeled_set = MyDataset(labeled_set[0], labeled_set[1])
        unlabeled_set = MyDataset(unlabeled_set[0], unlabeled_set[1])

        labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(
            datasets.MNIST('./data/mnist', train=False, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=False
            )
    
    return labeled_loader, unlabeled_loader, test_loader