import os
import numpy as np
import tensorflow as tf

#author Vladyslav Glavatskyi

#import custom modules from python modules
from dataLoader import load_and_prepare_cifar10, display_sample_images
from model import create_traditional_cnn, create_resnet, count_parameters
from Trainning import train_model, evaluate_model
from Visualizer import (
    plot_training_history, plot_confusion_matrices, plot_model_comparison,
    visualize_feature_maps, plot_misclassified_examples
)

#set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    #create output directory for results
    os.makedirs('results', exist_ok=True)

    #set training parameters
    epochs = 15  #number of epochs for training
    patience = 5 #patience for early stopping

    #loadCIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    train_generator, val_generator, test_generator, class_names, x_test, y_test = load_and_prepare_cifar10(
        batch_size=256)

    #use subset of the test data
    x_test_small = x_test[:1500]  #use only 1500 test images
    y_test_small = y_test[:1500]  #use only 1500 test labels
    #display sample images
    display_sample_images(x_test[:10], y_test[:10], class_names)

    #enable mixed precision training
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision training enabled")
    except:
        print("Mixed precision training not available")

    #create models
    print("Creating models...")
    traditional_cnn = create_traditional_cnn()
    resnet_model = create_resnet()

    #count parameters
    trad_params = count_parameters(traditional_cnn)
    resnet_params = count_parameters(resnet_model)

    print(f"Traditional CNN parameters: {trad_params:,}")
    print(f"ResNet parameters: {resnet_params:,}")

    #train models
    print("\nTraining Traditional CNN...")
    trad_model, trad_history, trad_time = train_model(
        traditional_cnn, train_generator, val_generator,
        epochs=epochs, model_name="traditional_cnn", patience=patience
    )

    print(f"\nTraditional CNN training completed in {trad_time:.2f} seconds")

    print("\nTraining ResNet...")
    resnet_model, resnet_history, resnet_time = train_model(
        resnet_model, train_generator, val_generator,
        epochs=epochs, model_name="resnet", patience=patience
    )

    print(f"\nResNet training completed in {resnet_time:.2f} seconds")

    #check models
    print("\nEvaluating Traditional CNN...")
    trad_results = evaluate_model(trad_model, test_generator, class_names, x_test_small, y_test_small)

    print("\nEvaluating ResNet...")
    resnet_results = evaluate_model(resnet_model, test_generator, class_names, x_test_small, y_test_small)

    #print report
    print("\nTraditional CNN Classification Report:")
    print(trad_results['classification_report'])

    print("\nResNet Classification Report:")
    print(resnet_results['classification_report'])

    #visualize results
    print("\nGenerating visualizations...")

    #plot training history
    try:
        plot_training_history(trad_history, resnet_history)
    except Exception as e:
        print(f"Error plotting training history: {e}")

    #plot confusion matrices
    try:
        plot_confusion_matrices(
            trad_results['confusion_matrix'],
            resnet_results['confusion_matrix'],
            class_names
        )
    except Exception as e:
        print(f"Error plotting confusion matrices: {e}")

    #plot model comparison
    try:
        plot_model_comparison(
            trad_results, resnet_results,
            trad_time, resnet_time,
            trad_params, resnet_params
        )
    except Exception as e:
        print(f"Error plotting model comparison: {e}")

    #visualize feature maps for sample image
    sample_idx = np.random.randint(0, len(x_test))
    sample_image = x_test[sample_idx]
    true_label = np.argmax(y_test[sample_idx])

    #get valid layer name for feature map visualization
    try:
        #traditional CNN
        trad_conv_layers = [layer.name for layer in traditional_cnn.layers if 'conv' in layer.name]
        if trad_conv_layers:
            trad_layer_name = trad_conv_layers[1]  #use second conv layer
            visualize_feature_maps(
                trad_model, sample_image, trad_layer_name, class_names, true_label
            )

        #ResNet
        resnet_conv_layers = [layer.name for layer in resnet_model.layers if 'conv' in layer.name]
        if resnet_conv_layers:
            resnet_layer_name = resnet_conv_layers[2]  # Use the third conv layer
            visualize_feature_maps(
                resnet_model, sample_image, resnet_layer_name, class_names, true_label
            )
    except Exception as e:
        print(f"Error visualizing feature maps: {e}")

    #plot misclassified examples
    try:
        plot_misclassified_examples(
            x_test, trad_results['y_true'], trad_results['y_pred'],
            class_names, 'traditional_cnn'
        )

        plot_misclassified_examples(
            x_test, resnet_results['y_true'], resnet_results['y_pred'],
            class_names, 'resnet'
        )
    except Exception as e:
        print(f"Error plotting misclassified examples: {e}")

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()