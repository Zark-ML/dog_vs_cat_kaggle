import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DogCatDataset(Dataset):
    def __init__(self, root_dir, size, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            size (tuple): Desired output size of the images (width, height).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.size = size
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        image = image.resize(self.size, Image.LANCZOS)  # Use Image.LANCZOS instead of Image.ANTIALIAS
        
        # Extract label from filename
        label = 1 if 'dog' in self.image_files[idx] else 0
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

if __name__ == "__main__":
    # Example usage:
    root_dir = '/home/edgar/dev/zark_ml/dog_cat/data/train'
    size = (128, 128)  # desired size of the output images
    transform = transforms.Compose([
        transforms.ToTensor(),  # convert the image to a tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize the image
    ])

    # Create the dataset
    dataset = DogCatDataset(root_dir=root_dir, size=size, transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Iterate through the DataLoader
    for batch_data, batch_labels in dataloader:
        print(batch_data.size(), batch_labels.size())
        # Here you would typically pass the batch_data to your model
