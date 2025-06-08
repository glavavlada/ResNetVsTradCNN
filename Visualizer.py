import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
import os


def plot_training_history(history_trad, history_resnet):
    """Plot training history for original models"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # Plot accuracy
    axes[0].plot(history_trad.history['accuracy'], label='Traditional CNN - Training', linewidth=2)
    axes[0].plot(history_trad.history['val_accuracy'], label='Traditional CNN - Validation', linewidth=2)
    axes[0].plot(history_resnet.history['accuracy'], label='ResNet - Training', linewidth=2)
    axes[0].plot(history_resnet.history['val_accuracy'], label='ResNet - Validation', linewidth=2)
    axes[0].set_title('Model Accuracy Comparison', fontsize=16)
    axes[0].set_ylabel('Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Plot loss
    axes[1].plot(history_trad.history['loss'], label='Traditional CNN - Training', linewidth=2)
    axes[1].plot(history_trad.history['val_loss'], label='Traditional CNN - Validation', linewidth=2)
    axes[1].plot(history_resnet.history['loss'], label='ResNet - Training', linewidth=2)
    axes[1].plot(history_resnet.history['val_loss'], label='ResNet - Validation', linewidth=2)
    axes[1].set_title('Model Loss Comparison', fontsize=16)
    axes[1].set_ylabel('Loss', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/training_history_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: results/training_history_comparison.png")


def plot_training_history_all_models(histories):
    """Plot training history for all models"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # Plot accuracy
    for i, (name, history) in enumerate(histories.items()):
        if history is not None:
            color = colors[i % len(colors)]
            axes[0].plot(history.history['accuracy'], label=f'{name} - Training',
                         linewidth=2, color=color, linestyle='-')
            axes[0].plot(history.history['val_accuracy'], label=f'{name} - Validation',
                         linewidth=2, color=color, linestyle='--')

    axes[0].set_title('All Models Accuracy Comparison', fontsize=16)
    axes[0].set_ylabel('Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Plot loss
    for i, (name, history) in enumerate(histories.items()):
        if history is not None:
            color = colors[i % len(colors)]
            axes[1].plot(history.history['loss'], label=f'{name} - Training',
                         linewidth=2, color=color, linestyle='-')
            axes[1].plot(history.history['val_loss'], label=f'{name} - Validation',
                         linewidth=2, color=color, linestyle='--')

    axes[1].set_title('All Models Loss Comparison', fontsize=16)
    axes[1].set_ylabel('Loss', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/all_models_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: results/all_models_training_history.png")


def plot_confusion_matrices(cm_model1, cm_model2, class_names, model1_name="Model 1", model2_name="Model 2"):
    """Plot confusion matrices for two models"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # First model confusion matrix
    sns.heatmap(cm_model1, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'{model1_name} Confusion Matrix', fontsize=16)
    axes[0].set_ylabel('True Label', fontsize=14)
    axes[0].set_xlabel('Predicted Label', fontsize=14)

    # Second model confusion matrix
    sns.heatmap(cm_model2, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f'{model2_name} Confusion Matrix', fontsize=16)
    axes[1].set_ylabel('True Label', fontsize=14)
    axes[1].set_xlabel('Predicted Label', fontsize=14)

    plt.tight_layout()
    plt.savefig('results/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: results/confusion_matrices_comparison.png")


def plot_all_confusion_matrices(evaluation_results, class_names):
    """Plot confusion matrices for all models"""
    num_models = len(evaluation_results)
    cols = min(3, num_models)
    rows = (num_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if num_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (model_name, results) in enumerate(evaluation_results.items()):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.3f}', fontsize=14)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)

    # Hide empty subplots
    for idx in range(num_models, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig('results/all_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: results/all_confusion_matrices.png")


def plot_comprehensive_model_comparison(evaluation_results, training_times, models):
    model_names = list(evaluation_results.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    #performance metrics
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        values = []
        for name in model_names:
            if metric == 'F1 Score':
                values.append(evaluation_results[name]['f1'])
            else:
                values.append(evaluation_results[name][metric.lower()])

        bars = ax.bar(range(len(model_names)), values, alpha=0.8,
                      color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(model_names)])
        ax.set_title(f'{metric} Comparison', fontsize=16)
        ax.set_ylabel(metric, fontsize=14)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim([0, 1])

        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: results/comprehensive_model_comparison.png")


def plot_model_efficiency(evaluation_results, training_times, models):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    model_names = list(evaluation_results.keys())
    accuracies = [evaluation_results[name]['accuracy'] for name in model_names]
    params = [sum(p.numpy().size for p in models[name].trainable_weights) / 1000000 for name in
              model_names]  #in millions
    times = [training_times[name] / 60 for name in model_names]  #in minutes

    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(model_names)]

    #accuracy vs Parameters
    scatter1 = axes[0].scatter(params, accuracies, s=200, c=colors, alpha=0.7, edgecolors='black')
    axes[0].set_xlabel('Model Parameters (Millions)', fontsize=14)
    axes[0].set_ylabel('Accuracy', fontsize=14)
    axes[0].set_title('Model Efficiency: Accuracy vs Parameters', fontsize=16)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.8, 1.0])

    #add labels
    for i, name in enumerate(model_names):
        axes[0].annotate(name, (params[i], accuracies[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=12)

    #accuracy vs Training Time
    scatter2 = axes[1].scatter(times, accuracies, s=200, c=colors, alpha=0.7, edgecolors='black')
    axes[1].set_xlabel('Training Time (Minutes)', fontsize=14)
    axes[1].set_ylabel('Accuracy', fontsize=14)
    axes[1].set_title('Model Efficiency: Accuracy vs Training Time', fontsize=16)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.8, 1.0])

    #add labels
    for i, name in enumerate(model_names):
        axes[1].annotate(name, (times[i], accuracies[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=12)

    plt.tight_layout()
    plt.savefig('results/model_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: results/model_efficiency_analysis.png")


def visualize_feature_maps(model, image, layer_name, class_names, true_label, model_name="Model"):
    try:
        #create model that outputs feature maps from specified layer
        feature_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

        #get feature maps
        image_batch = np.expand_dims(image, axis=0)
        feature_maps = feature_model.predict(image_batch, verbose=0)

        #plot original image and feature maps
        plt.figure(figsize=(15, 10))

        #plot original image
        plt.subplot(4, 4, 1)
        plt.imshow(image)
        plt.title(f"Original: {class_names[true_label]}", fontsize=12)
        plt.axis('off')

        #plot feature maps
        num_feature_maps = min(15, feature_maps.shape[-1])
        for i in range(num_feature_maps):
            plt.subplot(4, 4, i + 2)
            plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
            plt.title(f"Feature Map {i + 1}", fontsize=10)
            plt.axis('off')

        plt.suptitle(f'{model_name} - Feature Maps from {layer_name}', fontsize=16)
        plt.tight_layout()
        filename = f'results/feature_maps_{model_name.lower().replace(" ", "_").replace("-", "_")}_{layer_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")

    except Exception as e:
        print(f"Error visualizing feature maps for {model_name}: {e}")


def plot_misclassified_examples(x_test, y_true, y_pred, class_names, model_name, num_examples=10):
    misclassified_indices = np.where(y_true != y_pred)[0]

    if len(misclassified_indices) > 0:
        num_to_display = min(num_examples, len(misclassified_indices))
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i in range(num_to_display):
            idx = misclassified_indices[i]
            axes[i].imshow(x_test[idx])
            axes[i].set_title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}",
                              fontsize=10)
            axes[i].axis('off')

        #hide unused subplots
        for i in range(num_to_display, 10):
            axes[i].set_visible(False)

        plt.suptitle(f'{model_name} - Misclassified Examples ({len(misclassified_indices)} total errors)',
                     fontsize=16)
        plt.tight_layout()
        filename = f'results/{model_name.lower().replace(" ", "_").replace("-", "_")}_misclassified.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")
        print(
            f"Total misclassified: {len(misclassified_indices)} out of {len(y_true)} ({len(misclassified_indices) / len(y_true) * 100:.1f}%)")
    else:
        print(f"No misclassified examples found for {model_name}.")


def create_performance_summary_chart(evaluation_results, training_times, models):
    """Create a comprehensive performance summary chart"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    model_names = list(evaluation_results.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][:len(model_names)]

    #1. Accuracy Comparison
    accuracies = [evaluation_results[name]['accuracy'] for name in model_names]
    bars1 = ax1.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.set_ylim([0.8, 1.0])
    ax1.grid(True, axis='y', alpha=0.3)

    #add accuracy values on bars
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    #2.Training Time Comparison
    times_minutes = [training_times[name] / 60 for name in model_names]
    bars2 = ax2.bar(model_names, times_minutes, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Training Time Comparison', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Training Time (Minutes)', fontsize=14)
    ax2.grid(True, axis='y', alpha=0.3)

    #add time values on bars
    for bar, time_min in zip(bars2, times_minutes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times_minutes) * 0.01,
                 f'{time_min:.1f}m', ha='center', va='bottom', fontsize=12, fontweight='bold')

    #3. Model Parameters Comparison
    params_millions = [sum(p.numpy().size for p in models[name].trainable_weights) / 1000000
                       for name in model_names]
    bars3 = ax3.bar(model_names, params_millions, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_title('Model Size Comparison', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Parameters (Millions)', fontsize=14)
    ax3.grid(True, axis='y', alpha=0.3)

    #add parameter values on bars
    for bar, params in zip(bars3, params_millions):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(params_millions) * 0.01,
                 f'{params:.2f}M', ha='center', va='bottom', fontsize=12, fontweight='bold')

    #4.Efficiency Score (Accuracy per Parameter)
    efficiency_scores = [acc / params for acc, params in zip(accuracies, params_millions)]
    bars4 = ax4.bar(model_names, efficiency_scores, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_title('Model Efficiency (Accuracy/Parameters)', fontsize=16, fontweight='bold')
    ax4.set_ylabel('Efficiency Score', fontsize=14)
    ax4.grid(True, axis='y', alpha=0.3)

    #add efficiency values on bars
    for bar, eff in zip(bars4, efficiency_scores):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(efficiency_scores) * 0.01,
                 f'{eff:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    #rotate x-axis labels for better readability
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('results/performance_summary_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: results/performance_summary_chart.png")


def plot_class_wise_performance(evaluation_results, class_names):
    from sklearn.metrics import classification_report

    fig, axes = plt.subplots(len(evaluation_results), 1, figsize=(15, 5 * len(evaluation_results)))
    if len(evaluation_results) == 1:
        axes = [axes]

    for idx, (model_name, results) in enumerate(evaluation_results.items()):
        #get classification report as dict
        report = classification_report(results['y_true'], results['y_pred'],
                                       target_names=class_names, output_dict=True)

        #extract per-class metrics
        classes = class_names
        precision = [report[cls]['precision'] for cls in classes]
        recall = [report[cls]['recall'] for cls in classes]
        f1_score = [report[cls]['f1-score'] for cls in classes]

        x = np.arange(len(classes))
        width = 0.25

        ax = axes[idx]
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)

        ax.set_title(f'{model_name} - Class-wise Performance', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('results/class_wise_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: results/class_wise_performance.png")


def generate_all_visualizations(evaluation_results, histories, training_times, models, class_names, x_test, y_test):
    print("\n" + "=" * 60)
    print("GENERATING ALL VISUALIZATIONS")
    print("=" * 60)

    try:
        #1.Training history for all models
        if histories:
            print("\n1. Plotting training histories...")
            plot_training_history_all_models(histories)
    except Exception as e:
        print(f"Error plotting training histories: {e}")

    try:
        #2.All confusion matrices
        if evaluation_results:
            print("\n2. Plotting confusion matrices...")
            plot_all_confusion_matrices(evaluation_results, class_names)
    except Exception as e:
        print(f"Error plotting confusion matrices: {e}")

    try:
        #3.Comprehensive model comparison
        if evaluation_results and training_times and models:
            print("\n3. Plotting comprehensive model comparison...")
            plot_comprehensive_model_comparison(evaluation_results, training_times, models)
    except Exception as e:
        print(f"Error plotting model comparison: {e}")

    try:
        #4.Performance summary chart
        if evaluation_results and training_times and models:
            print("\n4. Creating performance summary chart...")
            create_performance_summary_chart(evaluation_results, training_times, models)
    except Exception as e:
        print(f"Error creating performance summary: {e}")

    try:
        #5.Model efficiency analysis
        if evaluation_results and training_times and models:
            print("\n5. Plotting model efficiency analysis...")
            plot_model_efficiency(evaluation_results, training_times, models)
    except Exception as e:
        print(f"Error plotting efficiency analysis: {e}")

    try:
        #6.Class-wise performance
        if evaluation_results:
            print("\n6. Plotting class-wise performance...")
            plot_class_wise_performance(evaluation_results, class_names)
    except Exception as e:
        print(f"Error plotting class-wise performance: {e}")

    try:
        #7.Misclassified examples for each model
        print("\n7. Plotting misclassified examples...")
        for model_name, results in evaluation_results.items():
            plot_misclassified_examples(
                x_test[:2000], results['y_true'], results['y_pred'],
                class_names, model_name, num_examples=10
            )
    except Exception as e:
        print(f"Error plotting misclassified examples: {e}")

    try:
        #8.Feature maps for best model
        if evaluation_results and models:
            print("\n8. Visualizing feature maps for best model...")
            best_model_name = max(evaluation_results.keys(),
                                  key=lambda x: evaluation_results[x]['accuracy'])
            best_model = models[best_model_name]

            #get sample image
            sample_idx = np.random.randint(0, len(x_test))
            sample_image = x_test[sample_idx]
            true_label = np.argmax(y_test[sample_idx])

            #find conv layers
            conv_layers = [layer.name for layer in best_model.layers
                           if 'conv' in layer.name.lower()]

            if conv_layers and len(conv_layers) > 1:
                layer_name = conv_layers[1]  #use second conv layer
                visualize_feature_maps(
                    best_model, sample_image, layer_name,
                    class_names, true_label, best_model_name
                )
    except Exception as e:
        print(f"Error visualizing feature maps: {e}")

    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETED!")
    print("=" * 60)
    print("Check the 'results/' folder for all generated images.")