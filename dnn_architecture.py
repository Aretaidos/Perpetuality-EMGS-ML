import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import regularizers

# Training Configuration - OPTIMIZED FOR 7 GESTURES
EPOCHS = args.epochs or 200  # Reduced - fewer classes converge faster
LEARNING_RATE = args.learning_rate or 0.0006  # Slightly higher
ENSURE_DETERMINISM = args.ensure_determinism

# Batch size can be larger with fewer classes
BATCH_SIZE = args.batch_size or 20

if not ENSURE_DETERMINISM:
    train_dataset = train_dataset.shuffle(buffer_size=BATCH_SIZE*4)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

# SIMPLIFIED Architecture for 7 Distinct Gestures
# Fewer layers needed since classes are well-separated
model = Sequential()
model.add(InputLayer(input_shape=(input_length, ), name='x_input'))

# Layer 1: Wide feature extraction
model.add(Dense(56, activation='relu',
    kernel_regularizer=regularizers.l2(0.0008),
    name='dense_1'))
model.add(BatchNormalization(name='batch_norm_1'))
model.add(Dropout(0.35, name='dropout_1'))

# Layer 2: Pattern combination
model.add(Dense(40, activation='relu',
    kernel_regularizer=regularizers.l2(0.0008),
    name='dense_2'))
model.add(BatchNormalization(name='batch_norm_2'))
model.add(Dropout(0.3, name='dropout_2'))

# Layer 3: Gesture-specific features
model.add(Dense(28, activation='relu',
    kernel_regularizer=regularizers.l2(0.001),
    name='dense_3'))
model.add(Dropout(0.25, name='dropout_3'))

# Layer 4: Final refinement (optional - can remove for even faster inference)
model.add(Dense(16, activation='relu',
    name='dense_4'))
model.add(Dropout(0.2, name='dropout_4'))

# Output layer: 7 gestures
model.add(Dense(classes, name='y_pred', activation='softmax'))

# Optimizer
opt = Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.9,
    beta_2=0.999,
    clipnorm=1.0
)

callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS, ensure_determinism=ENSURE_DETERMINISM))

# Early stopping - shorter patience since convergence is faster
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True,
    verbose=1
)
callbacks.append(early_stopping)

# Learning rate reduction
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=12,
    min_lr=0.00001,
    verbose=1
)
callbacks.append(reduce_lr)

# Compile and train
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    verbose=2,
    callbacks=callbacks
)

disable_per_channel_quantization = False