from src.dataset_loader import DatasetLoader

loader = DatasetLoader("cifar10")

train_loader, test_loader, classes = loader.load_dataset()

print("Number of classes:", len(classes))

for images, labels in test_loader:
    print(images.shape)
    print(labels.shape)
    break