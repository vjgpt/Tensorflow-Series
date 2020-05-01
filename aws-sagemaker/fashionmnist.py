import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import subprocess



if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation

    # Model Version
    VERSION = '00000123'

    # input image dimensions
    img_rows, img_cols = 28, 28

    train_images = np.load(os.path.join(training_dir, 'training.npz'))['image']
    train_labels = np.load(os.path.join(training_dir, 'training.npz'))['label']
    test_images  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
    test_labels  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']

    # reshape for feeding into the model
    train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
    test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    # scale the values to 0.0 to 1.0
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    model = keras.Sequential([
    keras.layers.Conv2D(input_shape=input_shape, filters=8, kernel_size=3, 
                        strides=2, activation='relu', name='Conv1'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
    ])
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=opt, 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), batch_size=batch_size, epochs=epochs, verbose=1)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print('Validation loss :', test_loss)
    print('Validation accuracy :', test_acc)

    saved_model_path = model.save(os.path.join(args.model_dir, VERSION), save_format='tf')