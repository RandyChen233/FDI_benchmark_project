# Install necessary packages (if not already installed)
# !pip install tensorflow scikit-learn

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from Classification_helpers import read_data
# 1. Data Preparation
from tensorflow.keras import backend as K
from sklearn.utils import class_weight

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
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # Clip the prediction tensor to prevent log(0) error
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # Calculate cross entropy
        cross_ent = y_true * K.log(y_pred)
        # Multiply by weights
        weighted_cross_ent = cross_ent * weights
        # Sum over the classes
        loss = -K.sum(weighted_cross_ent, axis=-1)
        return loss

    return loss

# Generate synthetic data for illustration (replace with your actual data)
#np.random.seed(42)
#X = np.random.rand(n_scenarios, n_time_steps, n_sensors)
#y = np.random.randint(0, n_classes, size=(n_scenarios, n_time_steps))

# read dataset
module_path='C:\\Users\\jhaja\\OneDrive\\Desktop\\quovadis\\QuoVadisTAD'
dataset_name='ourClassdata'
dataset_trace=0
preprocess='0-1'
trainset, valset, testset, y_train, y_val, y_test = read_data(module_path,
                                                        dataset_name,
                                                        dataset_trace=dataset_trace,
                                                        preprocess=preprocess)

n_time_steps = trainset.shape[1]
n_sensors = trainset.shape[2]
n_classes = 4  # No Fault, Fault1, Fault2

'''
# 2. Train/Validation/Test Split

# First, split into temporary (train + val) and test sets
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Then, split the temporary set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, shuffle=True
)  # 0.25 x 0.8 = 0.2 for validation

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")

# 3. Normalization

# Initialize the scaler
scaler = StandardScaler()

# Reshape the training data to 2D for fitting the scaler
X_train_reshaped = X_train.reshape(-1, n_sensors)
scaler.fit(X_train_reshaped)

# Transform the training, validation, and test data
X_train_scaled = scaler.transform(X_train.reshape(-1, n_sensors)).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape(-1, n_sensors)).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, n_sensors)).reshape(X_test.shape)
'''
# 4. Encode Labels to One-Hot Vectors

y_train_categorical = to_categorical(y_train, num_classes=n_classes)
y_val_categorical = to_categorical(y_val, num_classes=n_classes)
y_test_categorical = to_categorical(y_test, num_classes=n_classes)

# 5. Model Building

model = Sequential()

# Bidirectional LSTM Layer
model.add(Bidirectional(
    LSTM(64, return_sequences=True),
    input_shape=(n_time_steps, n_sensors)
))

# Dropout for regularization
model.add(Dropout(0.5))

# TimeDistributed Dense Layer
model.add(TimeDistributed(Dense(32, activation='relu')))

# Another Dropout Layer
model.add(Dropout(0.5))

# Output Layer with Softmax Activation for Multi-Class Classification
model.add(TimeDistributed(Dense(n_classes, activation='softmax')))


# Your computed class weights
class_weights = {0: 0.5,
                1: 1.2,
                2: 1.1,
                3: 4.5}

# Convert the dictionary to a list ordered by class indices
weights = [class_weights[i] for i in range(len(class_weights))]

# 6. Model Compilation
# Define the weighted loss
loss = weighted_categorical_crossentropy(weights)

# Flatten the y_train to calculate class weights
y_train_flat = y_train.flatten()

# Compute class weights
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_flat),
    y=y_train_flat
)

# Create a dictionary mapping class indices to weights
class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}
print("Computed Class Weights:", class_weights_dict)

# Compile the model
'''
model.compile(
    optimizer='adam',
    loss=loss,
    metrics=['accuracy']
)
'''
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 7. Model Training

# Define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]

# Train the model
history = model.fit(
    trainset, y_train_categorical,
    validation_data=(valset, y_val_categorical),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)

# 8. Model Evaluation

# Load the best model
model.load_weights('best_model.h5')

# Predict on test data
y_pred = model.predict(testset)
y_pred_classes = np.argmax(y_pred, axis=-1)
y_true_classes = np.argmax(y_test_categorical, axis=-1)

# Flatten the predictions and true labels
y_pred_flat = y_pred_classes.flatten()
y_true_flat = y_true_classes.flatten()

# Classification Report
print("Classification Report:")
print(classification_report(y_true_flat, y_pred_flat, target_names=['No Fault', 'Fault', 'Disturbance','Fault and Disturbance']))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true_flat, y_pred_flat))
