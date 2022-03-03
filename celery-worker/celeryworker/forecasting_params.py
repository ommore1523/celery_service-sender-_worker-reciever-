from tkinter.tix import Tree


ui_data = {
        "NBEATS":{
            "algo_params":{
                "input_chunk_length": 24,
                "output_chunk_length": 12,
                "generic_architecture":True,
                "num_stacks":30,
                "num_blocks":1,
                "num_layers":4,
                "layer_widths":256,
                "expansion_coefficient_dim":5,
                "trend_polynomial_degree":2,
                "batch_size":10,
                "optimizer_kwargs":{'lr': 1e-3},
                "n_epochs":20,
                "prediction_period":2 # NONALGO
            },
            "hyper_param_tunning":True
        },
        "TCN":{
            "algo_params":{
                "input_chunk_length":24,
                "output_chunk_length":12,
                "kernel_size":3,
                "num_filters":3,
                "weight_norm": True,
                "random_state":42,
                "dilation_base":2,
                "num_layers":None,
                "dropout":0.3,
                "batch_size":10,
                "n_epochs":20,
                "prediction_period":2 # NONALGO
            },
            "hyper_param_tunning":True
        },

        "FBPROPHET":{
            "algo_params":{
                "holiday_country_code":None,
                "prediction_period":2
            },
            "hyper_param_tunning":True
        },
        
        "RNN": {
            "algo_params": {
                "input_chunk_length":24,
                "model":"RNN",
                "hidden_dim":25,
                "n_rnn_layers":1,
                "dropout":0.0,
                "training_length":12,
                "n_epochs":10,
                "optimizer_kwargs":{"lr": 0.0001},
                "prediction_period":4
            },
            "hyper_param_tunning":True
        },

        "THETA": {
            "algo_params": {
                "prediction_period":4
            },
            "hyper_param_tunning":True
        },
        
        "FFT":{
            "algo_params":{
                "nr_freqs_to_keep":20,
                "trend":"poly",
                "trend_poly_degree": 1,
                "prediction_period":12
            },
            "hyper_param_tunning":True
        }
        
}

causal_defaults_params = {
    "NBEATS": {
            "algo_params":{"input_chunk_length": 24,
                "output_chunk_length": 12,
                "generic_architecture":True,
                "num_stacks":30,
                "num_blocks":1,
                "num_layers":4,
                "layer_widths":256,
                "expansion_coefficient_dim":5,
                "trend_polynomial_degree":2,
                "batch_size":10,
                "optimizer_kwargs":{'lr': 1e-3},
                "n_epochs":20,
                "prediction_period":2 # NONALGO
            },
            "hyper_params": {
                "input_chunk_length":[],
                "output_chunk_length":[],
                "n_epochs":[]
            }
    },
           
    "TCN": {
        "algo_params":  { 
            "input_chunk_length":24,
            "output_chunk_length":12,
            "kernel_size":3,
            "num_filters":3,
            "weight_norm": True,
            "random_state":42,
            "dilation_base":2,
            "num_layers":None,
            "dropout":0.2,
            "batch_size":10,
            "n_epochs":10,
            "prediction_period":2  # NONALGO
        },
        "hyper_params":
            {
                "input_chunk_length":[],
                "output_chunk_length":[],
                "dropout":[0.1],
                "n_epochs":[]
        }
    },
      
    "FBPROPHET":{
        "algo_params":{},
        "hyper_params":{
            "name":"tunning",
            'seasonal_periods': 12,
            'fourier_order': 1 , 
            'prior_scale': 0.5,  
            'mode': "additive"  # ('additive' or 'multiplicative')
        }
    },

    "RNN": {
        "algo_params":
            {
                "input_chunk_length":24,
                "model":"RNN",
                "hidden_dim":25,
                "n_rnn_layers":1,
                "dropout":0.0,
                "training_length":12,
                "n_epochs":20,
                "optimizer_kwargs":{"lr": 0.0001},
                "prediction_period":12
            },
        "hyper_params": {
            "input_chunk_length":[],
            "model":['RNN','LSTM','GRU'],
            "hidden_dim":[],
            "n_rnn_layers":[],
            "dropout":[],
            "training_length":[],
            "n_epochs":[]
        }
    },
    "FFT":{
            "algo_params":{
                "nr_freqs_to_keep":20,
                "trend":"poly",
                "trend_poly_degree": 1,
                "prediction_period":12
            },
            "hyper_params":{
                "nr_freqs_to_keep":[20,30,40,50,100],
                "trend":["poly","exp"],
                "trend_poly_degree":[1,2,3,4,5]
            }
    }
}



