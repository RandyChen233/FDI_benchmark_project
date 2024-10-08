import os
import re
from pathlib import Path
import numpy as np
import yaml
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from quovadis_tad.dataset_utils.dataset_reader import datasets
from quovadis_tad.dataset_utils.data_utils import preprocess_data, preprocess_data_3D,concatenate_windows_feat
from quovadis_tad.model_utils.tf_data_loader import *
from keras.preprocessing import timeseries_dataset_from_array
from quovadis_tad.model_utils.gnn import get_graph_info, LSTMGC, get_graph_info_3D
from quovadis_tad import config_data3D

module_path = str(Path.cwd().parents[1])


def get_label_loader(labels, config, classification = True):
    if classification:
        labels_gen = timeseries_dataset_from_array(
        labels[:],        
        None,
        sequence_length=config['input_sequence_length'],
        shuffle=False,
        batch_size=config['batch_size'],
    )
    else:
        labels_gen = timeseries_dataset_from_array(
            labels[:-1],        
            None,
            sequence_length=config['input_sequence_length'],
            shuffle=False,
            batch_size=config['batch_size'],
        )
    return labels_gen

"""Todo: modify get_dataloader to adapt to classificaiton models (supervised)"""
def get_dataloader(data_array,
                   labels_array,
                   input_sequence_length=12,
                   batch_size=1,
                   reconstruction_dataset=False,
                   classification=True,
                   shuffle=False,
                   ):
    #TODO pass this also from configs. for now it is fixed
    forecast_horizon = 1
    #config_data3D.data3D=False
    if config_data3D.data3D:
        if not classification:
            return create_tf_dataset_from_3d_array(data_array, 
                                                   input_sequence_length, 
                                                   forecast_horizon,
                                                    batch_size=batch_size, 
                                                    reconstruction_dataset=reconstruction_dataset,
                                                    shuffle=shuffle)
        else:
            return create_tf_dataset_from_3d_array_classify(data_array, 
                                                            labels_array,
                                                            input_sequence_length, 
                                                            batch_size=batch_size, 
                                                            shuffle=shuffle)

    else:
        if not classification:
            return create_tf_dataset(data_array,
                                    input_sequence_length,
                                    forecast_horizon,
                                    batch_size=batch_size,
                                    reconstruction_dataset=reconstruction_dataset,
                                    shuffle=shuffle)
        else:
            return create_tf_dataset_classify(data_array,
                                              labels_array,
                                              input_sequence_length,
                                              batch_size,
                                              shuffle=shuffle)


def norm_1_1(arr):
    return 2 * arr - 1

def read_data(module_path, 
                dataset_name, 
                dataset_trace=None,
                preprocess="0-1", 
                config=None):
    # prepare dataset
    trainset, testset, labels = datasets[dataset_name](module_path) #based on load_Ourbench()
    # print(len(trainset), len(testset), len(labels)) #27, 27, 27
    if config_data3D.data3D:
        print(f'Using 3D dataset...')
        dataset_trace = 0
        trainset=np.array(trainset)
        testset=np.array(testset)
        print(f' [WARNING:] Using first trace of this dataset. Specify dataset_trace otherwise')
        # Preprocess datasets
        trainset, valset, testset = preprocess_data_3D(trainset, testset, train_size=0.9, val_size=0.1,
                                                       normalization=preprocess)
        labels=np.array(labels)
        # print(labels.shape) #(27, 8849, 6)
        labels = labels.max(axis=2)

    else:
        if dataset_name == 'wadi_127' or dataset_name == 'wadi_112':
            # WADI dataset is large, we use the first 60.000 samples
            trainset = trainset[:60_000]

        if type(trainset) is list:
            if dataset_trace is not None:
                trainset, testset, labels = trainset[dataset_trace], testset[dataset_trace], labels[dataset_trace]
            else:
                dataset_trace = 0
                trainset, testset, labels = trainset[0], testset[0], labels[0]
                print(f' [WARNING:] Using first trace of this dataset. Specify dataset_trace otherwise')

        if trainset.shape[1] == 1:   # check if univariate timeseries - [TODO] configs and window_size needs to be tuned if we train on univariate data
            trainset = concatenate_windows_feat(trainset, window_size=5)
            testset = concatenate_windows_feat(testset, window_size=5)

        if len(labels.shape) > 1:
            labels = labels.max(1)

        # Preprocess datasets
        trainset, valset, testset = preprocess_data(trainset, testset, 0.9, 0.1, normalization=preprocess)
    

    return trainset, valset, testset, labels, dataset_trace

"""The baseline models:"""
class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches, activation='gelu'),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=num_patches, activation='gelu'),
                layers.Dense(units=hidden_units),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = tf.linalg.matrix_transpose(x)
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = layers.Dense(ff_dim, activation="relu")
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    

    
def gcn_sequense_sensor_error_model_v1(input_shape, config=None, training=True):
    
    lstm_units = config['lstm_units']
    embedd_1 = config['embedding_dim'][0]
    #embedd_2 = config['embedding_dim'][1]
    output_dim_1 = config['output_dim'][0]
    #output_dim_2 = config['output_dim'][1]
    num_seq, num_nodes = input_shape 
    #modeling sensor nodes
    graph_conv_params = {
                "aggregation_type": "max",  #mean
                "combination_type": "concat",
                "activation": 'relu',
            }
    
    gcn_node = LSTMGC(
        1,
        embedd_1,
        config["lstm_units"],
        num_seq,
        output_dim_1,        
        config["graph_info_nodes"],
        graph_conv_params,
        return_normal=True,
    )

         
    inputs = layers.Input(shape=input_shape) #(batch, seq, nodes)    
    
    inputs_nodes = layers.Reshape((num_seq, num_nodes, 1))(inputs)  #(batch, seq, nodes, 1)    
    
    regression_targets = gcn_node(inputs_nodes) #output shape (batch_size, output_dim_1, num_nodes]
    
    regression_targets = layers.Dense(num_nodes)(regression_targets)
    
    if not training: 
        model = keras.Model(inputs=inputs, outputs=[regression_targets, regression_targets])
    else:
        model = keras.Model(inputs=inputs, outputs=regression_targets)
    return model 


"""Supervised classification models:"""
"""The classification models always ouputs a binary ouput: either 0 (no faults) or 1 (faults)"""
# (a) MLP Implementation
# Modified to output fault classification for each time step
def build_mlp(input_shape, num_classes=1):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.TimeDistributed(layers.Dense(num_classes, activation='sigmoid')))
    return model

# (b) FCN Implementation
# Modified to output fault classification for each time step
def build_fcn(input_shape, num_classes=1):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv1D(128, kernel_size=8, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv1D(256, kernel_size=5, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv1D(128, kernel_size=3, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.TimeDistributed(layers.Dense(num_classes, activation='sigmoid')))
    return model

# (c) ResNet Implementation (1D version)
# Modified to output fault classification for each time step
def build_resnet(input_shape, num_classes=1):
    inputs = layers.Input(shape=input_shape)
    
    # First ResNet Block
    x = layers.Conv1D(64, kernel_size=8, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, kernel_size=5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    shortcut = layers.Conv1D(64, kernel_size=1, padding='same')(inputs)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    # Second ResNet Block
    x = layers.Conv1D(128, kernel_size=8, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, kernel_size=5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    shortcut = layers.Conv1D(128, kernel_size=1, padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    # Third ResNet Block
    x = layers.Conv1D(128, kernel_size=8, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, kernel_size=5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    shortcut = layers.Conv1D(128, kernel_size=1, padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    # TimeDistributed Fully Connected Layer and Sigmoid Output
    x = layers.TimeDistributed(layers.Dense(num_classes, activation='sigmoid'))(x)
    
    model = models.Model(inputs, x)
    return model


def build_sequence_embedder(input_shape=(None, None),
                            config=None,
                            training=False):
    # input of shape [batch_size, input_seq_len, sensors) 
    num_seq = input_shape[0]
        
    model = config['model']
    num_blocks = config['num_blocks']
    embedding_dim = config['embedding_dim']
    dropout_rate = config['dropout_rate']
    positional_encoding=config['positional_encoding']
    
    if model == "Single_Transformer_block":
        blocks = keras.Sequential(
            [TransformerBlock(embedding_dim, config['MHA_blocks'], embedding_dim, dropout_rate) for _ in range(num_blocks)],
            name="representation"
        )
        
    elif model == "Single_block_MLPMixer":
        blocks = keras.Sequential(
            [MLPMixerLayer(num_seq, embedding_dim, dropout_rate) for _ in range(num_blocks)],
            name="representation"
        )
    elif model == "1_Layer_MLP":
        blocks = keras.Sequential(
            [layers.Dropout(rate=dropout_rate, name='target_dropout')],
            name="representation"
        )
    
    elif model == "1_Layer_GCN_LSTM": 
        return gcn_sequense_sensor_error_model_v1(input_shape, config=config, training=training)
    
    elif model == "MLP":
        return build_mlp(input_shape,num_classes=1)
    
    elif model == "FCN":
        return build_fcn(input_shape,num_classes=1)
    
    elif model == "ResNet":
        return build_resnet(input_shape,num_classes=1)
  
    else:
        raise ValueError('model type not Implemented. use one of {1_Layer_GCN_LSTM, Single_block_MLPMixer, Single_Transformer_block, 1_Layer_MLP}')    
    


    # Specify Model architecture
    inputs = layers.Input(shape=input_shape)
    print(f'inputs have shape {inputs.shape}')

    x = layers.Dense(units=embedding_dim)(inputs)  # (batch_size, input_seq_len, embedding_dim) project input embedding dim length
    if positional_encoding:
        positions = tf.range(start=0, limit=num_nodes, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_nodes, output_dim=embedding_dim
        )(positions)
        x = x + position_embedding
    
    if model == "1_Layer_MLP":
        embedding = x
    else:
        embedding = blocks(x)
    
    if not config['reconstruction'] and num_seq > 1:
        # Apply global average pooling to generate a [batch_size, 1, embedding_dim] tensor.
        embedding =  layers.GlobalAveragePooling1D(name="embedding")(embedding)
    
    regression_targets = layers.Dense(input_shape[1])(embedding)
    print(f'Regression output has shape {regression_targets.shape}')
    # Create the Keras model.
    if not training:
        return keras.Model(inputs=inputs, outputs=[regression_targets, embedding])
    else:
        return keras.Model(inputs=inputs, outputs=regression_targets)


def get_model(input_shape,
              config_data,
              model_dir=None,
              load_weights=False,
              compile_model=True,
              training=True
              ):
    
    model = build_sequence_embedder(input_shape=input_shape,
                                    config=config_data,
                                    training=training)
        
        
    if config_data['optimizer'] == "adam":
        #optim = keras.optimizers.Adam(learning_rate=config_data['learning_rate'])
        optim = keras.optimizers.legacy.Adam(learning_rate=config_data['learning_rate'])
    elif config_data['optimizer'] == "rmsprop":
        #optim = keras.optimizers.RMSprop(learning_rate=config_data['learning_rate'])
        optim = keras.optimizers.legacy.RMSprop(learning_rate=config_data['learning_rate'])

    else:
        #optim = keras.optimizers.SGD(learning_rate=config_data['learning_rate'], momentum=0.9)
        optim = keras.optimizers.legacy.SGD(learning_rate=config_data['learning_rate'], momentum=0.9)
                  
    if load_weights:
        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        model.load_weights(latest_checkpoint).expect_partial()
        print('Loaded pretrained_checkpoint')

    if compile_model:
        if config_data['model'] in {"MLP", "FCN", "ResNet"}:
            model.compile(optimizer=optim,
                          loss=keras.losses.BinaryCrossentropy(),
            )
        else:
            model.compile(optimizer=optim,
                          loss=keras.losses.MeanSquaredError(),
            )
    return model


    
def train_embedder(module_path,
                   dataset_name,
                   dataset_trace=None,
                   config_path=None,                   
                   load_weights=False,
                   training=True,
                   classification=True):
    config_name = os.path.basename(config_path)
    # Read config
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    
    # handle missing key in old configs, set if doesnt exist
    if 'reconstruction' not in config_data:
        config_data["reconstruction"] = False
        print(f'reconstruction dataset mode not found in config... setting it to False')

    if 'classification' in config_data:
        print(f'Training in classification mode...')

    # read dataset
    trainset, valset, testset, labels, dataset_trace = read_data(module_path,
                                                                dataset_name,
                                                                dataset_trace=dataset_trace,
                                                                preprocess=config_data["preprocess"],
                                                                config=config_data)
    # print(f' labels has shape {labels.shape}')
    # labels_gen = get_label_loader(labels, config_data)
    # gt_labels= []
    # for gt in labels_gen.as_numpy_iterator():
    #     gt_labels.append(gt)
    # gt_labels = np.vstack(gt_labels)
    # print(f' labels has shape {labels.shape}')

    if dataset_trace is None:
        checkpoint_folder = Path(module_path, 'resources', 'model_checkpoints', dataset_name, config_name.split('.')[0])
    else:
        checkpoint_folder = Path(module_path, 'resources', 'model_checkpoints', dataset_name, str(dataset_trace), config_name.split('.')[0])
    
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    
    # get data loaders
    train_dataset = get_dataloader(trainset,
                                   labels_array=labels,
                                   batch_size=config_data['batch_size'],
                                   input_sequence_length=config_data['input_sequence_length'],
                                   reconstruction_dataset=config_data["reconstruction"],
                                   classification=classification,
                                   shuffle=True,
                                   )
    val_dataset = get_dataloader(valset,
                                 labels_array=labels,
                                 batch_size=config_data['batch_size'],
                                 input_sequence_length=config_data['input_sequence_length'],
                                 reconstruction_dataset=config_data["reconstruction"],
                                 classification=classification,
                                 )                  
    
    # model inout_shape (dataloader output [batch_size, input_sequence_length, nodes])              
    input_shape = (config_data['input_sequence_length'], trainset.shape[-1])
    print(f'Input shape is {input_shape}')

    # graph Info for gcn
    match = re.search("GCN", config_data["model"])
    if match:
        if config_data3D.data3D:
            graph_info_nodes, graph_info_seq = get_graph_info_3D(trainset, config_data, verbose=True)
        else:
            graph_info_nodes, graph_info_seq = get_graph_info(trainset, config_data, verbose=True)
        config_data["graph_info_nodes"] =  graph_info_nodes
        config_data["graph_info_seq"] =  graph_info_seq

    model = get_model(input_shape,
                    config_data,
                    model_dir=checkpoint_folder,
                    load_weights=load_weights,
                    compile_model=True,
                    training=training)
    
                  
    # checkpointer
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=Path(checkpoint_folder, "weights_best"),  
        monitor="val_loss",
        mode='min',
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    # Create a learning rate scheduler callback.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.9, patience=15
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True
    )
    # Fit the model.
    history = model.fit(
                        train_dataset,
                        validation_data=val_dataset,
                        epochs=config_data['epochs'],
                        callbacks=[checkpointer, reduce_lr, early_stopping], # 
                    )
    
    return history



######## Model Evaluation ########

def test_embedder(module_path,
                  dataset_name,
                  dataset_trace=None,
                  model_name=None,
                  load_weights=True,
                  training=False,
                  subset='test',
                  output_model=False,
                  classification = True,
                  ):

    nn_baselines_dict = {'1_Layer_MLP': 'mlp_small_embedd_32_seq_5.yaml',
                         'Single_block_MLPMixer': 'MLPmixer_blocks_1_embedd_128_seq_5.yaml',
                         'Single_Transformer_block': 'Transformer_blocks_1_1_embedd_128_seq_5.yaml',
                         '1_Layer_GCN_LSTM': 'gcn_lstm_model_seq_5.yaml'}
    config_name = nn_baselines_dict[model_name]
    config_path = Path(module_path, 'quovadis_tad', 'model_configs', config_name)
    #config_name = os.path.basename(config_path)
    # Read config
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    
   # handle missing key in old configs, set if doesnt exist
    if 'reconstruction' not in config_data:
        config_data["reconstruction"] = False 
    
    # read dataset
    trainset, _, testset, labels, dataset_trace = read_data(module_path,
                                                             dataset_name,
                                                             dataset_trace=dataset_trace,
                                                             preprocess=config_data["preprocess"],
                                                             config=config_data)

    # prep checkpoiint folder
    if dataset_trace is None:
        checkpoint_folder = Path(module_path, 'resources', 'model_checkpoints', dataset_name, config_name.split('.')[0])
    else:
        checkpoint_folder = Path(module_path, 'resources', 'model_checkpoints', dataset_name, str(dataset_trace), config_name.split('.')[0])
    
    
    print(f'NN-Baseline: {config_data["model"]}')
    # graph Info for gcn
    match = re.search("GCN", config_data["model"])
    if match:
        graph_info_nodes, graph_info_seq = get_graph_info(trainset, config_data, verbose=True)
        config_data["graph_info_nodes"] =  graph_info_nodes
        config_data["graph_info_seq"] =  graph_info_seq
    
    
    
    if not config_data["reconstruction"]:
        target_offset = config_data['input_sequence_length']  #input_sequence_length + forecast_horizon - 1
        if not config_data3D.data3D:
            gt_labels = labels[target_offset:]
        else:

            gt_labels_full =labels[:,target_offset:]
            gt_labels = np.reshape(gt_labels_full, (-1, 1))


    else:
        labels_gen = get_label_loader(labels, config_data)
        gt_labels= []
        for gt in labels_gen.as_numpy_iterator():
            gt_labels.append(gt)
        gt_labels = np.vstack(gt_labels)
            
            
    # get data loaders
    if subset == 'train':
        test_dataset = get_dataloader(trainset,
                                     batch_size=config_data['batch_size'],
                                     input_sequence_length=config_data['input_sequence_length'],
                                     reconstruction_dataset=config_data["reconstruction"])                  
    else:
        test_dataset = get_dataloader(testset,
                                 batch_size=config_data['batch_size'],
                                 input_sequence_length=config_data['input_sequence_length'],
                                     reconstruction_dataset=config_data["reconstruction"]) 
    
    # model inout_shape ([batch_size, input_sequence_length, nodes])              
    input_shape = (config_data['input_sequence_length'], testset.shape[-1])
    print(f'Input data size is {input_shape}')
    
    model = get_model(input_shape,
              config_data,
              model_dir=checkpoint_folder,
              load_weights=load_weights,
              compile_model=True,
              training=training,
              )
    
    inputs = []
    predictions = []
    orig_target = []
    embeddings = []
    for input_batch, target in test_dataset:
        
        pred, embedding  = model.predict_on_batch(input_batch)
        predictions.append(pred)
        embeddings.append(embedding)       
        inputs.append(input_batch.numpy())
        if config_data["reconstruction"] :
            orig_target.append(input_batch.numpy())
        elif config_data["classification"]:
            orig_target.append(input_batch.numpy())
        else:
            orig_target.append(gt_labels)
            
    if output_model:
        inputs = np.vstack(inputs)
        predictions = np.vstack(predictions)
        embeddings = np.vstack(embeddings)
        orig_target = np.vstack(orig_target)
        return inputs, predictions.squeeze(), orig_target.squeeze(), gt_labels, embeddings.squeeze(), model
    else:
        predictions = np.vstack(predictions).squeeze()
        #embeddings = np.vstack(embeddings).squeeze()
        orig_target = np.vstack(orig_target).squeeze()
        score = np.abs(predictions - orig_target)
        return score, gt_labels
        