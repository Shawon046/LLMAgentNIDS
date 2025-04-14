import torch
from torch.utils.data import Dataset

class CreateDataset(Dataset):
    def __init__(self, X, y):

        if isinstance(X, torch.Tensor):  # Check if X is a PyTorch tensor
            X = X.numpy()  # Convert to NumPy array
        if isinstance(y, torch.Tensor):  # Check if y is a PyTorch tensor
            y = y.numpy()  # Convert to NumPy array

        self.X = torch.from_numpy(X).float() #, dtype=torch.float32)  # Convert to torch tensors
        self.y = torch.from_numpy(y).float() #, dtype=torch.float32)  # Assuming regression, use long for classification
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # return sig_window, [label, file, idx]
        return self.X[idx], [self.y[idx], 'None', 'None'] 
    



def print_label_counts(dataloader):



    # Use next to get a sample batch from the DataLoader
    batch = next(iter(dataloader))

    # Assuming the batch is a tuple, get the first element
    batch_data = batch[0]

    # Calculate min, max, and std
    min_value = torch.min(batch_data)
    max_value = torch.max(batch_data)
    std_value = torch.std(batch_data)
    print(f'Minimum value: {min_value.item()}')
    print(f'Maximum value: {max_value.item()}')
    print(f'Standard Deviation: {std_value.item()}')


    label_counts = Counter()

    # Iterate through the DataLoader
    for _, attributes in dataloader:
        if isinstance(attributes, list):
            labels, files, index = attributes
        else:
            labels = attributes
        # Convert labels to CPU and flatten if needed
        labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
        label_counts.update(labels)

    # Print the value counts for each label
    for label, count in label_counts.items():
        print(f"Label {label}: {count}")
