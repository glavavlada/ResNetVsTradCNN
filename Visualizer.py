import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model



def plot_training_history(history_trad, history_resnet):
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    #plot accuracy
    axes[0].plot(history_trad.history['accuracy'], label='Traditional CNN - Training')
    axes[0].plot(history_trad.history['val_accuracy'], label='Traditional CNN - Validation')
    axes[0].plot(history_resnet.history['accuracy'], label='ResNet - Training')
    axes[0].plot(history_resnet.history['val_accuracy'], label='ResNet - Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    #plot loss
    axes[1].plot(history_trad.history['loss'], label='Traditional CNN - Training')
    axes[1].plot(history_trad.history['val_loss'], label='Traditional CNN - Validation')
    axes[1].plot(history_resnet.history['loss'], label='ResNet - Training')
    axes[1].plot(history_resnet.history['val_loss'], label='ResNet - Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('training_history_comparison.png', dpi=300)
    plt.show()


def plot_confusion_matrices(cm_trad, cm_resnet, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    #traditional CNN confusion matrix
    sns.heatmap(cm_trad, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Traditional CNN Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    #resNet confusion matrix
    sns.heatmap(cm_resnet, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('ResNet Confusion Matrix')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300)
    plt.show()


def plot_model_comparison(trad_results, resnet_results, trad_time, resnet_time, trad_params, resnet_params):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    trad_values = [trad_results['accuracy'], trad_results['precision'],
                   trad_results['recall'], trad_results['f1']]
    resnet_values = [resnet_results['accuracy'], resnet_results['precision'],
                     resnet_results['recall'], resnet_results['f1']]

    #plot metrics comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    #performance metrics
    x = np.arange(len(metrics))
    width = 0.35

    axes[0].bar(x - width / 2, trad_values, width, label='Traditional CNN')
    axes[0].bar(x + width / 2, resnet_values, width, label='ResNet')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Performance Metrics')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(True, axis='y')

    #training time and model size
    comparison_metrics = ['Training Time (s)', 'Model Parameters (M)']
    trad_comp_values = [trad_time, trad_params / 1_000_000]
    resnet_comp_values = [resnet_time, resnet_params / 1_000_000]

    x = np.arange(len(comparison_metrics))

    axes[1].bar(x - width / 2, trad_comp_values, width, label='Traditional CNN')
    axes[1].bar(x + width / 2, resnet_comp_values, width, label='ResNet')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Computational Metrics')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(comparison_metrics)
    axes[1].legend()
    axes[1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()


def visualize_feature_maps(model, image, layer_name, class_names, true_label):
    #create a model that outputs feature maps from specified layer
    feature_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    #get feature maps
    image_batch = np.expand_dims(image, axis=0)
    feature_maps = feature_model.predict(image_batch)

    #plot the original image and feature maps
    plt.figure(figsize=(15, 10))

    #plot original image
    plt.subplot(4, 4, 1)
    plt.imshow(image)
    plt.title(f"Original: {class_names[true_label]}")
    plt.axis('off')

    #plot feature maps
    num_feature_maps = min(15, feature_maps.shape[-1])
    for i in range(num_feature_maps):
        plt.subplot(4, 4, i + 2)
        plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
        plt.title(f"Feature Map {i + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'feature_maps_{layer_name}.png', dpi=300)
    plt.show()


def plot_misclassified_examples(x_test, y_true, y_pred, class_names, model_name, num_examples=10):
    misclassified_indices = np.where(y_true != y_pred)[0]

    if len(misclassified_indices) > 0:
        num_to_display = min(num_examples, len(misclassified_indices))
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i in range(num_to_display):
            idx = misclassified_indices[i]
            axes[i].imshow(x_test[idx])
            axes[i].set_title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(f'{model_name}_misclassified.png', dpi=300)
        plt.show()
    else:
        print("No misclassified examples found.")