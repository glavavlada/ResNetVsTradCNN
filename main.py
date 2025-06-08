import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#author Vladyslav Glavatskyi

# import custom modules from python modules
from dataLoader import load_and_prepare_cifar10, display_sample_images
from model import create_traditional_cnn, create_resnet, create_mobilenet, count_parameters
from Trainning import train_model, evaluate_model
from Visualizer import (
    generate_all_visualizations,  #main comprehensive function
    plot_training_history,        #individual comparisons if needed
    plot_confusion_matrices,      #specific model comparisons
    visualize_feature_maps,       #individual feature map analysis
    plot_misclassified_examples   #individual model error analysis
)

#set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    print("Starting main function...")

    #create output directory for results
    os.makedirs('results', exist_ok=True)

    #training parameters
    epochs = 30  #number of epoch
    patience = 8  #patience for convergence

    #load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    train_generator, val_generator, test_generator, class_names, x_test, y_test = load_and_prepare_cifar10(
        batch_size=64)

    #use subset of test data
    x_test_small = x_test[:2000]
    y_test_small = y_test[:2000]

    #display sample images
    display_sample_images(x_test[:10], y_test[:10], class_names)

    print("Using standard precision training")

    #create models
    print("Creating models...")
    traditional_cnn = create_traditional_cnn()
    resnet_model = create_resnet()
    mobilenet_model = create_mobilenet()

    #Dictionary to store all models
    models = {
        'Traditional CNN': traditional_cnn,
        'ResNet': resnet_model,
        'MobileNet-Inspired': mobilenet_model
    }

    #count parameters for all models
    print("\nModel Parameters:")
    for name, model in models.items():
        params = count_parameters(model)
        print(f"{name} parameters: {params:,}")

    #storage for results
    trained_models = {}
    histories = {}
    training_times = {}
    evaluation_results = {}

    #train all models
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")

        try:
            trained_model, history, training_time = train_model(
                model, train_generator, val_generator,
                epochs=epochs, model_name=name.lower().replace(' ', '_').replace('-', '_'),
                patience=patience
            )

            #store results
            trained_models[name] = trained_model
            histories[name] = history
            training_times[name] = training_time

        except Exception as e:
            print(f"Error training {name}: {e}")
            continue

    #evaluate all models
    print("\n" + "="*60)
    print("EVALUATING ALL MODELS")
    print("="*60)

    for name, model in trained_models.items():
        try:
            print(f"\nEvaluating {name}...")
            results = evaluate_model(model, test_generator, class_names, x_test_small, y_test_small)
            evaluation_results[name] = results

            print(f"\n{name} Results:")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"F1-Score: {results['f1']:.4f}")

        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            continue

    #print summary comparison
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    if evaluation_results:
        print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Time(s)':<10} {'Params':<12}")
        print("-" * 95)

        for name in models.keys():
            if name in evaluation_results and name in training_times:
                results = evaluation_results[name]
                params = count_parameters(models[name])
                print(f"{name:<25} {results['accuracy']:<10.4f} {results['precision']:<10.4f} "
                      f"{results['recall']:<10.4f} {results['f1']:<10.4f} {training_times[name]:<10.1f} {params / 1000000:<12.2f}M")

        #find best model
        best_model = max(evaluation_results.keys(), key=lambda x: evaluation_results[x]['accuracy'])
        best_accuracy = evaluation_results[best_model]['accuracy']
        print(f"\nBest performing model: {best_model}")
        print(f"Best accuracy: {best_accuracy:.4f}")
    else:
        print("No models were successfully trained and evaluated.")

    #generate all visualizations using comprehensive function
    print("\nGenerating comprehensive visualizations...")

    try:
        generate_all_visualizations(
            evaluation_results, histories, training_times,
            models, class_names, x_test, y_test
        )
    except Exception as e:
        print(f"Error generating visualizations: {e}")

        #fallback-try individual visualizations
        print("Attempting individual visualizations...")

        try:
            #training history comparison
            if len(histories) >= 2:
                model_names = list(histories.keys())
                plot_training_history(histories[model_names[0]], histories[model_names[1]])
        except Exception as e2:
            print(f"Error with training history: {e2}")

        try:
            #confusion matrices
            if len(evaluation_results) >= 2:
                model_names = list(evaluation_results.keys())
                plot_confusion_matrices(
                    evaluation_results[model_names[0]]['confusion_matrix'],
                    evaluation_results[model_names[1]]['confusion_matrix'],
                    class_names, model_names[0], model_names[1]
                )
        except Exception as e3:
            print(f"Error with confusion matrices: {e3}")

    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()