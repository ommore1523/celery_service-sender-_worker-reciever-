

ui_data = {
        "NBEATS":{
            "algo_params":{
                "input_chunk_length": 12,
                "output_chunk_length": 6,
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
            "hyper_param_tunning":False,
            "causal":False,
            "user_cols":"timeindex"
        },
        "TCN":{
            "algo_params":{
                "input_chunk_length":12,
                "output_chunk_length":6,
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
            "hyper_param_tunning":False,
            "causal":False,
            "user_cols":"timeindex"
        },

        "FBPROPHET":{
            "algo_params":{
                "holiday_country_code":None,
                "prediction_period":2
            },
            "hyper_param_tunning":False,
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
            "hyper_param_tunning":False,
            "causal":True,
            "user_cols":"timeindex"
        },

        "THETA": {
            "algo_params": {
                "prediction_period":4
            },
            "hyper_param_tunning":False,
        },
        
        "FFT":{
            "algo_params":{
                "nr_freqs_to_keep":20,
                "trend":"poly",
                "trend_poly_degree": 1,
                "prediction_period":12
            },
            "hyper_param_tunning":False,
        },

        "LINEAR_RG":{
            "algo_params":{
                "fit_intercept": True,
                "positive":False
            },
            "hyper_param_tunning":True
        },
      
        "RIDGE_RG":{
            "algo_params":{
                "alpha": 1.0,
                "fit_intercept":True,
                "tol":0.001,
                "solver":'auto',
                "positive":False
            },
            "hyper_param_tunning":True
        },
      
        "LASSO_RG":{
            "algo_params":{
                "alpha":1.0,
                "fit_intercept":True,
                "copy_X":True,
                "max_iter":1000,
                "tol":0.0001,
                "warm_start":False,
                "positive":False,
                "selection":'cyclic'
            },
            "hyper_param_tunning":True
        },
      
        "DT_RG":{
            "algo_params":{
                "criterion":"squared_error",
                "splitter":"best",
                "max_depth":None,
                "min_samples_split":2,
                "max_features": None
            },
            "hyper_param_tunning":True
        },
        
        "RF_RG":{
            "algo_params":{
                "n_estimators":100,
                "criterion":"squared_error",
                "max_depth":None,
                "min_samples_split":2,
                "min_samples_leaf":1,
                "min_weight_fraction_leaf":0.0,
                "max_features":"auto",
                "max_leaf_nodes":None,
                "min_impurity_decrease":0.0,
                "bootstrap":True,
                "oob_score":False,
                "warm_start":False,
                "ccp_alpha":0.0,
                "max_samples":None
            },
            "hyper_param_tunning":True
        },
        
        "SVR":{
            "algo_params":{
                "kernel":'rbf',
                "degree":3,
                "gamma":'scale',
                "coef0":0.0,
                "tol":0.001,
                "C":1.0,
                "epsilon":0.1
            },
            "hyper_param_tunning":True
        },
      
        "ADAB":{
            "algo_params":{
                "n_estimators":50,
                "learning_rate":1.0,
                "loss":'linear',
            },
            "hyper_param_tunning":True
        },
      
        "XGB":{
            "algo_params":{
                "loss":'squared_error',
                "learning_rate":0.1,
                "n_estimators":100,
                "min_samples_split":2,
                "min_samples_leaf":1,
                "max_depth":3,
                "alpha":0.9,
                "validation_fraction":0.1
            },
            "hyper_param_tunning":True
        },
       
        "ETR":{
            "algo_params":{
                "n_estimators":100,
                "criterion":"squared_error",
                "max_depth":None,
                "min_samples_split":2,
                "min_samples_leaf":1,
                "max_features":"auto",
                "bootstrap":False,
                "ccp_alpha":0.0
            },
            "hyper_param_tunning":True
        },
 
        "HGB":{
            "algo_params":{
                "loss":"squared_error",
                "learning_rate":0.1,
                "max_iter":100,
                "max_leaf_nodes":31,
                "max_depth":None,
                "min_samples_leaf":20,
                "l2_regularization":0.0,
                "max_bins":255,
                "validation_fraction":0.1,
                "n_iter_no_change":10,
                "tol":0.0000001
            },
            "hyper_param_tunning":True
        },
 
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
    },

    "LINEAR_RG": {
            "algo_params":{
                "fit_intercept": True,
                "positive":False
            },
            "hyper_params": {
                "fit_intercept":[False, True],
                "positive":[True,False],
                "normalize":[False, True]
            }
    },
  
    "RIDGE_RG": {
        "algo_params":{
            "alpha": 1.0,
            "fit_intercept":True,
            "tol":0.001,
            "solver":'auto',
            "positive":False
        },
        "hyper_params": {
            "alpha":[1.0, 0.001, 2.0],
            "fit_intercept":[True,False],
            "positive":[True, False],
            "tol":[0.00001, 0.0000001, 0.00001,0.001],
            "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
        }
    },
    
    "LASSO_RG": {
        "algo_params":{
            "alpha":1.0,
            "fit_intercept":True,
            "copy_X":True,
            "max_iter":1000,
            "tol":0.0001,
            "warm_start":False,
            "positive":False,
            "selection":'cyclic'
        },
        "hyper_params": {
            "alpha":[1.0,2.0],
            "fit_intercept":[True,False],
            "copy_X":[True, False],
            "max_iter":[1000, 2000, 10000, 15000],
            "tol" : [0.0001, 0.001, 0.01, 0.1],
            "warm_start":[False,True],
            "positive":[False, True],
            "selection":['cyclic','random']
        }
    },
    
    "DT_RG": {
        "algo_params":{
            "criterion":"squared_error",
            "splitter":"best",
            "max_depth":None,
            "min_samples_split":2,
            "max_features": None
        },
        "hyper_params": {
            "criterion":['squared_error','friedman_mse','absolute_error','poisson'],
            "splitter":['best','random'],
            "max_depth":[20,50,100,200,300, 400, 1000,4000],
            "min_samples_split":[2,3,4,5,6,7,8,9],
            "max_features":['auto','sqrt','log2']
        }
    },
   
    "RF_RG": {
        "algo_params":{
            "n_estimators":100,
            "criterion":"squared_error",
            "max_depth":None,
            "min_samples_split":2,
            "min_samples_leaf":1,
            "min_weight_fraction_leaf":0.0,
            "max_features":"auto",
            "max_leaf_nodes":None,
            "min_impurity_decrease":0.0,
            "bootstrap":True,
            "oob_score":False,
            "warm_start":False,
            "ccp_alpha":0.0,
            "max_samples":None
        },
        "hyper_params": {
            "n_estimators":[100,200],
            "criterion":["squared_error","absolute_error","poisson"],
            "max_depth":[10,20, None],
            "min_samples_split":[2],
            "max_features":["auto","sqrt","log2"],
            "min_impurity_decrease":[0.0, 0.1, 0.2],
            "max_samples":[10, 20,None]
        }
    },
   
    "SVR": {
        "algo_params":{
            "kernel":'rbf',
            "degree":3,
            "gamma":'scale',
            "coef0":0.0,
            "tol":0.001,
            "C":1.0,
            "epsilon":0.1
        },
        "hyper_params": {
            "kernel":['linear','poly','rbf','sigmoid'],
            "degree":[1,2,3],
            "gamma":['scale'],
            "C":[1.0,1.5, 2.0, 2.1],
            "epsilon":[0.1, 0.2]
        }
    },

    "ADAB":{
        "algo_params":{
            "n_estimators":50,
            "learning_rate":1.0,
            "loss":'linear',
        },
        "hyper_params":{
            "n_estimators":[50,70,100,200],
            "learning_rate":[0.0001, 0.001, 0.01, 1.0],
            "loss":['linear','square','exponential']
        }
    },
   
    "XGB":{
        "algo_params":{
            "loss":'squared_error',
            "learning_rate":0.1,
            "n_estimators":100,
            "min_samples_split":2,
            "min_samples_leaf":1,
            "max_depth":3,
            "alpha":0.9,
            "validation_fraction":0.1
        },
        "hyper_params":{
            "learning_rate":[0.1, 0.001, 0.0001],
            "n_estimators":[100,200, 250],
            "max_depth":[3,5,7],
            "alpha":[0.2, 0.4, 0.7, 0.9],

        }
    },
   
    "ETR":{
        "algo_params":{
            "n_estimators":100,
            "criterion":"squared_error",
            "max_depth":None,
            "min_samples_split":2,
            "min_samples_leaf":1,
            "max_features":"auto",
            "bootstrap":False,
            "ccp_alpha":0.0
        },
        "hyper_params":{
            "n_estimators":[100,200,300],
            "criterion":['squared_error','friedman_mse'],
            "max_depth":[3,5,10, None],
            "bootstrap":[True, False]

        }
    },

    "HGB":{
        "algo_params":{
            "loss":"squared_error",
            "learning_rate":0.1,
            "max_iter":100,
            "max_leaf_nodes":31,
            "max_depth":None,
            "min_samples_leaf":20,
            "l2_regularization":0.0,
            "max_bins":255,
            "validation_fraction":0.1,
            "n_iter_no_change":10,
            "tol":1e-07
        },

        "hyper_params":{
            "loss":["squared_error"],
            "learning_rate":[0.1, 0.001, 0.0001],
            "max_iter":[100,200, 300],
            "max_depth":[None, 3, 7, 12],
            "n_iter_no_change":[10, 20],
            "tol":[0.0000001, 0.0001, 0.00001]

        }
    },

}



