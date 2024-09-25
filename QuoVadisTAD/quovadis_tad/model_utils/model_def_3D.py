import os
import re
from pathlib import Path
import numpy as np
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from quovadis_tad.dataset_utils.dataset_reader import datasets
from quovadis_tad.dataset_utils.data_utils import preprocess_data, concatenate_windows_feat, preprocess_data_3D
from quovadis_tad.model_utils.tf_data_loader import create_tf_dataset
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from quovadis_tad.model_utils.gnn import get_graph_info, LSTMGC


module_path = str(Path.cwd().parents[1])


def get_label_loader(labels, config):
    labels_gen = timeseries_dataset_from_array(
        labels[:-1],        
        None,
        sequence_length=config['input_sequence_length'],
        shuffle=False,
        batch_size=config['batch_size'],
    )
    return labels_gen

def get_dataloader(data_array,
                   input_sequence_length=12,
                   batch_size=1,
                   reconstruction_dataset=False,
                   shuffle=False):
    #TODO pass this also from configs. for now it is fixed
    forecast_horizon = 1    
    return create_tf_dataset(data_array,
                             input_sequence_length,
                             forecast_horizon,
                             batch_size=batch_size,
                             reconstruction_dataset=reconstruction_dataset,
                             shuffle=shuffle)


def norm_1_1(arr):
    return 2 * arr - 1

def read_data_3D(module_path, dataset_name, dataset_trace=None,
             preprocess="0-1", config=None):
    # prepare dataset
    trainset, testset, labels = datasets[dataset_name](module_path)
    trainset=np.array(trainset)
    testset=np.array(testset)
    labels=np.array(labels)

    if len(labels.shape) > 1:
        labels = labels.max(2)
    
    # Preprocess datasets 
    trainset, valset, testset = preprocess_data_3D(trainset, testset, 0.9, 0.1, normalization=preprocess)
    return trainset, valset, testset, labels, dataset_trace


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

    
    if model == "1_Layer_GCN_LSTM":
        return gcn_sequense_sensor_error_model_v1(input_shape, config=config, training=training)
  
    else:
        raise ValueError('model type not Implemented. use one of {1_Layer_GCN_LSTM, Single_block_MLPMixer, Single_Transformer_block, 1_Layer_MLP}')    
    
    # Specify Model architecture
    inputs = layers.Input(shape=input_shape)

    x = layers.Dense(units=embedding_dim)(inputs)  # (batch_size, input_seq_len, embedding_dim) project input embedding dim length
    if positional_encoding:
        positions = tf.range(start=0, limit=num_nodes, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_nodes, output_dim=embedding_dim
        )(positions)
        x = x + position_embedding
    

    embedding = blocks(x)
    
    if not config['reconstruction'] and num_seq > 1:
        # Apply global average pooling to generate a [batch_size, 1, embedding_dim] tensor.
        embedding =  layers.GlobalAveragePooling1D(name="embedding")(embedding)
    
    regression_targets = layers.Dense(input_shape[1])(embedding)
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
        model.compile(optimizer=optim,
                      loss=keras.losses.MeanSquaredError(),
        )
    return model


    

def train_embedder(module_path,
                   dataset_name,
                   dataset_trace=None,
                   config_path=None,                   
                   load_weights=False,
                   training=True
                   ):
    config_name = os.path.basename(config_path)
    # Read config
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    
    
    # handle missing key in old configs, set if doesnt exist
    if 'reconstruction' not in config_data:
        config_data["reconstruction"] = False
        print(f'reconstruction dataset mode not found in config... setting it to False')
    
    
    
    # read dataset
    trainset, valset, testset, _, dataset_trace = read_data_3D(module_path,
                                                         dataset_name,
                                                         dataset_trace=dataset_trace,
                                                         preprocess=config_data["preprocess"],
                                                         config=config_data)

    if dataset_trace is None:
        checkpoint_folder = Path(module_path, 'resources', 'model_checkpoints', dataset_name, config_name.split('.')[0])
    else:
        checkpoint_folder = Path(module_path, 'resources', 'model_checkpoints', dataset_name, str(dataset_trace), config_name.split('.')[0])
    
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    
    # graph Info for gcn
    match = re.search("GCN", config_data["model"])
    if match:
        graph_info_nodes, graph_info_seq = get_graph_info(trainset, config_data, verbose=True)
        config_data["graph_info_nodes"] =  graph_info_nodes
        config_data["graph_info_seq"] =  graph_info_seq
    
    # get data loaders
    train_dataset = get_dataloader(trainset,
                                   batch_size=config_data['batch_size'],
                                   input_sequence_length=config_data['input_sequence_length'],
                                   reconstruction_dataset=config_data["reconstruction"],
                                   shuffle=True)
    val_dataset = get_dataloader(valset,
                                 batch_size=config_data['batch_size'],
                                 input_sequence_length=config_data['input_sequence_length'],
                                 reconstruction_dataset=config_data["reconstruction"])                  
    
    # model inout_shape (dataloader output [batch_size, input_sequence_length, nodes])              
    input_shape = (config_data['input_sequence_length'], trainset.shape[1])
    
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
        monitor="val_loss", patience=10, restore_best_weights=True
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
                  output_model=False
                  ):

    nn_baselines_dict = {'1_Layer_GCN_LSTM': 'gcn_lstm_model_seq_5.yaml'}
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
    trainset, _, testset, labels, dataset_trace = read_data_3D(module_path,
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
        gt_labels = labels[target_offset:]
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
    input_shape = (config_data['input_sequence_length'], testset.shape[1])
    
    model = get_model(input_shape,
              config_data,
              model_dir=checkpoint_folder,
              load_weights=load_weights,
              compile_model=True,
              training=training
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
        else:
            orig_target.append(target.numpy())
            
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
        