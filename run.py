from pipeline.pipeline import Pipeline

# Example usage:
train_dir = '/home/edgar/dev/zark_ml/dog_cat/data/train'
val_dir = '/home/edgar/dev/zark_ml/dog_cat/data/val'
size = (64, 64)
batch_size = 32
epochs = 10
learning_rate = 0.001
log_dir = './logs'

pipeline = Pipeline(train_dir, val_dir, size, batch_size, epochs, learning_rate, log_dir)
pipeline.train_and_evaluate()