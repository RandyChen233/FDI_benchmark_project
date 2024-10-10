# Install necessary packages (if not already installed)
# !pip install tensorflow scikit-learn imbalanced-learn tensorflow-addons

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional, LSTM, Dense, TimeDistributed, Dropout,
    Conv1D, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.utils import class_weight
from Classification_helpers import read_data  # Ensure this module is correctly implemented
from tensorflow.keras import backend as K

# 1. Data Preparation
module_path = 'C:\\Users\\jhaja\\OneDrive\\Desktop\\quovadis\\QuoVadisTAD'
dataset_name = 'ourClassdata'
dataset_trace = 0
preprocess = '0-1'

trainset, valset, testset, y_train, y_val, y_test = read_data(
    module_path,
    dataset_name,
    dataset_trace=dataset_trace,
    preprocess=preprocess
)

n_time_steps = trainset.shape[1]
n_sensors = trainset.shape[2]
n_classes = 4  # No Fault, Fault, Disturbance, Fault and Disturbance

# 2. Normalize the Data
scaler = StandardScaler()

# Fit scaler on training data
X_train_reshaped = trainset.reshape(-1, n_sensors)
scaler.fit(X_train_reshaped)

# Transform the data
X_train_scaled = scaler.transform(trainset.reshape(-1, n_sensors)).reshape(trainset.shape)
X_val_scaled = scaler.transform(valset.reshape(-1, n_sensors)).reshape(valset.shape)
X_test_scaled = scaler.transform(testset.reshape(-1, n_sensors)).reshape(testset.shape)

# 3. Compute Class Weights
y_train_flat = y_train.flatten()

class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_flat),
    y=y_train_flat
)

class_weights = class_weights_array  # e.g., [0.5, 1.2, 1.1, 4.5]
print("Computed Class Weights:", class_weights)

# 4. Encode Labels to One-Hot Vectors
y_train_categorical = to_categorical(y_train, num_classes=n_classes)
y_val_categorical = to_categorical(y_val, num_classes=n_classes)
y_test_categorical = to_categorical(y_test, num_classes=n_classes)


# 5. Define the Custom Weighted Categorical Crossentropy Loss Function
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of categorical crossentropy.

    Variables:
        weights: numpy array of shape (n_classes,)

    Usage:
        weights = np.array([0.5, 2, 10, ...])
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss, optimizer='adam')
    """
    weights = K.constant(weights)

    def loss(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # Compute cross-entropy
        cross_entropy = y_true * K.log(y_pred)
        # Apply the weights
        weighted_cross_entropy = cross_entropy * weights
        # Sum over classes
        loss = -K.sum(weighted_cross_entropy, axis=-1)
        return loss

    return loss


# 6. Build the Enhanced Model
model = Sequential()
# Convolutional Layer to extract local temporal features
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(n_time_steps, n_sensors)))
model.add(BatchNormalization())
# Removed MaxPooling1D to preserve timesteps
model.add(Dropout(0.3))

# Bidirectional LSTM Layers to capture sequential dependencies
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# TimeDistributed Dense Layers
model.add(TimeDistributed(Dense(64, activation='relu')))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Output Layer
model.add(TimeDistributed(Dense(n_classes, activation='softmax')))

model.summary()

# 7. Compile the Model with the Custom Loss Function
loss = weighted_categorical_crossentropy(class_weights)

model.compile(
    optimizer='adam',
    loss=loss,
    metrics=['accuracy']
)

# 8. Define Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# 9. Verify Data Shapes Before Training
print("X_train_scaled shape:", X_train_scaled.shape)
print("y_train_categorical shape:", y_train_categorical.shape)
print("X_val_scaled shape:", X_val_scaled.shape)
print("y_val_categorical shape:", y_val_categorical.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("y_test_categorical shape:", y_test_categorical.shape)

# Ensure that X_train_scaled and y_train_categorical have the same number of samples and timesteps
assert X_train_scaled.shape[0] == y_train_categorical.shape[0], "Mismatch in number of samples."
assert X_train_scaled.shape[1] == y_train_categorical.shape[1], "Mismatch in number of timesteps."

# 10. Train the Model with the Custom Loss Function (No class_weight parameter)
history = model.fit(
    X_train_scaled, y_train_categorical,
    validation_data=(X_val_scaled, y_val_categorical),
    epochs=100,
    batch_size=8,
    callbacks=callbacks
    # class_weight=class_weights_dict  # Ensure this is removed
)

# 11. Model Evaluation
# Load the best model
model.load_weights('best_model.h5')

# Predict on test data
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=-1)
y_true_classes = np.argmax(y_test_categorical, axis=-1)

# Flatten the predictions and true labels
y_pred_flat = y_pred_classes.flatten()
y_true_flat = y_true_classes.flatten()

# Classification Report
print("Classification Report:")
print(classification_report(
    y_true_flat, y_pred_flat,
    target_names=['No Fault', 'Fault', 'Disturbance', 'Fault and Disturbance']
))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true_flat, y_pred_flat))

# Additional Metrics
macro_f1 = f1_score(y_true_flat, y_pred_flat, average='macro')
weighted_f1 = f1_score(y_true_flat, y_pred_flat, average='weighted')
print(f"Macro F1-Score: {macro_f1}")
print(f"Weighted F1-Score: {weighted_f1}")

# ROC-AUC for Each Class
try:
    roc_auc = roc_auc_score(
        y_test_categorical.reshape(-1, n_classes),
        y_pred.reshape(-1, n_classes),
        average=None
    )
    for i, auc in enumerate(roc_auc):
        print(f"ROC-AUC for class {i}: {auc}")
except ValueError as e:
    print("ROC-AUC could not be computed:", e)
