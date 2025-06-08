import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import math

def cosine_annealing_schedule(epoch, lr, epochs=100):
    """Cosine annealing learning rate schedule"""
    return 0.001 * 0.5 * (1 + math.cos(math.pi * epoch / epochs))

def train_model(model, train_generator, val_generator, epochs=100, model_name="model", patience=15):
    # Enhanced callbacks for better training
    checkpoint = ModelCheckpoint(
        f"results/{model_name}_best.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
        save_weights_only=False
    )

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    )

    # Cosine annealing learning rate schedule
    lr_scheduler = LearningRateScheduler(
        lambda epoch: cosine_annealing_schedule(epoch, 0.001, epochs),
        verbose=1
    )

    # Backup learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1,
        cooldown=3
    )

    callbacks = [checkpoint, early_stopping, lr_scheduler, reduce_lr]

    # Record training time
    start_time = time.time()

    # Train the model
    try:
        # FIXED: Proper calculation for CIFAR-10
        # Training samples: 40,000 (after 20% validation split)
        # Validation samples: 10,000
        batch_size = 32  # Your batch size from dataLoader

        # Calculate steps properly
        train_samples = 40000  # 50000 - 10000 validation
        val_samples = 10000

        steps_per_epoch = train_samples // batch_size  # ~1250 steps
        validation_steps = val_samples // batch_size   # ~312 steps

        print(f"Training {model_name}:")
        print(f"  Epochs: {epochs}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Validation steps: {validation_steps}")
        print(f"  Batch size: {batch_size}")

        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,  # FIXED: Now properly calculated
            verbose=1
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Using best weights so far...")
        try:
            model.load_weights(f"results/{model_name}_best.h5")
            print(f"Loaded best weights from results/{model_name}_best.h5")
        except:
            print("Could not load best weights. Using current weights.")
        # Return empty history if interrupted early
        history = None

    except Exception as e:
        print(f"Error during training: {e}")
        # Fallback with fewer steps
        try:
            print("Attempting fallback training with reduced steps...")
            steps_per_epoch = 100  # Reduced fallback
            validation_steps = 50   # Reduced fallback

            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                verbose=1
            )
        except Exception as e2:
            print(f"Fallback training also failed: {e2}")
            return model, None, time.time() - start_time

    #calculate training time
    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds")

    return model, history, training_time

def evaluate_model(model, test_generator, class_names, x_test, y_test):
    #get predictions
    print("Making predictions...")
    y_pred_prob = model.predict(x_test, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    #calculate metrics
    accuracy = np.mean(y_pred == y_true)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    #generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    #generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names)

    #compile results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_true': y_true,
        'y_pred': y_pred
    }

    return results
