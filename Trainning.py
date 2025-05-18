import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


def train_model(model, train_generator, val_generator, epochs=15, model_name="model", patience=5):
    #create callbacks
    checkpoint = ModelCheckpoint(
        f"{model_name}_best.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    #stable learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    #record training time
    start_time = time.time()

    #train the model
    try:
        #calculate steps for training
        steps_per_epoch = 50  #number of steps per epoch
        validation_steps = None  #use all validation data

        #check if we're using tf.data.Dataset
        is_tf_dataset = isinstance(train_generator, tf.data.Dataset)

        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            steps_per_epoch=steps_per_epoch if is_tf_dataset else None,
            validation_steps=validation_steps,
            verbose=1
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Using best weights so far...")
        try:
            model.load_weights(f"{model_name}_best.h5")
            print(f"Loaded best weights from {model_name}_best.h5")
        except:
            print("Could not load best weights. Using current weights.")
    except Exception as e:
        print(f"Error during training: {e}")
        #fallback with fewer steps
        try:
            steps_per_epoch = 30
            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=[checkpoint, early_stopping, reduce_lr],
                steps_per_epoch=steps_per_epoch if is_tf_dataset else None,
                validation_steps=validation_steps,
                verbose=1
            )
        except Exception as e:
            print(f"Second attempt failed: {e}")
            return model, None, time.time() - start_time

    #calculate training time
    training_time = time.time() - start_time

    return model, history, training_time

def evaluate_model(model, test_generator, class_names, x_test, y_test):
    #get predictions
    y_pred_prob = model.predict(x_test)
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
