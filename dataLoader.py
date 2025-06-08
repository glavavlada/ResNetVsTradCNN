import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_and_prepare_cifar10(batch_size=32):
    #Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    #normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    #convert class vectors to binary class matrices (one-hot encoding)
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    #create validation set (20% of training data)
    val_size = int(0.2 * len(x_train))
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")

    #create tf.data.Dataset objects
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    #data augmentation function
    def augment(image, label):
        #random horizontal flip
        image = tf.image.random_flip_left_right(image)

        #mild brightness and contrast adjustments
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)

        #ensure image stays in [0,1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    #apply augmentation only to training data
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    #batch and prefetch all datasets
    train_ds = train_ds.shuffle(10000).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    #class names for visualization
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    #for compatibility with existing code
    train_generator = train_ds
    val_generator = val_ds
    test_generator = test_ds

    return train_generator, val_generator, test_generator, class_names, x_test, y_test

def display_sample_images(images, labels, class_names, num_samples=10):
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        if labels.shape[1] > 1:  # If one-hot encoded
            label_idx = np.argmax(labels[i])
        else:
            label_idx = int(labels[i])
        plt.title(class_names[label_idx])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
