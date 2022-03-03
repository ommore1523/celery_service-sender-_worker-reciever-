import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mape
from darts.dataprocessing.transformers import Scaler
from darts.utils.statistics import  check_seasonality
from darts.models import NBEATSModel, TCNModel, RNNModel, Prophet, Theta, FFT
from darts.utils.likelihood_models import LaplaceLikelihood, GaussianLikelihood
import warnings
warnings.filterwarnings("ignore")


# TODO: Remove chunk_size and calculate len inside function using scaled data len
# hyper params function

class Hyperparameterts:
    """
        Hyperparameterts function called when there is hyperparamete_tunning flag set to True from UI.
        It reads hyperparamete from config file. Tune for parameters combination and returns 
        best combination parameters. it return object of model creating a it with new params which 
        are returned by tunning.
    """

    def _NBEATSHyperparams(self,nbeatModel,parameters,causal_defaults_params,params_from_ui,train_chunk_size,val_chunk_size,tsidsBatchDataTrain,tsidsBatchDataVal):
        default_hyper_params = causal_defaults_params["NBEATS"]["hyper_params"]
        input_chunk_sizes = [default_hyper_params["input_chunk_length"].append(train_chunk_size * i) for i in range(1,4)]
        output_chunk_sizes = [default_hyper_params["output_chunk_length"].append(val_chunk_size * i)for i in range(1,4)]
        default_hyper_params["n_epochs"].append(params_from_ui["NBEATS"]["algo_params"]["n_epochs"])
        hyper_params = nbeatModel.gridsearch(parameters=default_hyper_params,series=tsidsBatchDataTrain[0],val_series=tsidsBatchDataVal[0])
        parameters = ForecastDartsMethods().preprocess_input_params_with_hyperparams(parameters,hyper_params[1])

        nbeatModel = NBEATSModel(
                random_state = 42,
                n_epochs = parameters["n_epochs"],
                num_stacks = parameters["num_stacks"],
                num_blocks = parameters["num_blocks"],
                num_layers = parameters["num_layers"],
                batch_size = parameters["batch_size"],
                layer_widths = parameters["layer_widths"],
                optimizer_kwargs = parameters["optimizer_kwargs"],
                input_chunk_length = parameters["input_chunk_length"],
                output_chunk_length = parameters["output_chunk_length"],
                generic_architecture = parameters["generic_architecture"],
                trend_polynomial_degree = parameters["trend_polynomial_degree"],
                expansion_coefficient_dim = parameters["expansion_coefficient_dim"]
            )

        return nbeatModel

    def _TCNHyperparams(self,tcnModel,parameters,causal_defaults_params,params_from_ui,train_chunk_size,val_chunk_size,tsidsBatchDataTrain,tsidsBatchDataVal):
        default_hyper_params = causal_defaults_params["TCN"]["hyper_params"]
        input_chunk_sizes = [default_hyper_params["input_chunk_length"].append(train_chunk_size * i) for i in range(2,4)]
        output_chunk_sizes = [default_hyper_params["output_chunk_length"].append(val_chunk_size * i)for i in range(1,4)]
        default_hyper_params["n_epochs"].append(params_from_ui["TCN"]["algo_params"]["n_epochs"])

        # hyper parameter tuning function 
        hyper_params = tcnModel.gridsearch(parameters=default_hyper_params,series=tsidsBatchDataTrain[0],val_series=tsidsBatchDataVal[0])
        parameters = ForecastDartsMethods().preprocess_input_params_with_hyperparams(parameters,hyper_params[1])

        tcnModel = TCNModel(
            input_chunk_length=parameters["input_chunk_length"],
            output_chunk_length=parameters["output_chunk_length"],
            kernel_size=parameters["kernel_size"],
            num_filters=parameters["num_filters"],
            weight_norm= parameters["weight_norm"],
            random_state=42,
            dilation_base=parameters["dilation_base"],
            num_layers=parameters["num_layers"],
            dropout=parameters["dropout"],
            batch_size=parameters["batch_size"],
            n_epochs=parameters["n_epochs"],
            likelihood=LaplaceLikelihood(),
        )  
        return tcnModel
    
    def _ProphetHyperparams(self,ts_id_data,causal_defaults_params,params_from_ui):
        seasonal_period = ForecastDartsMethods().check_seasonality(ts_id_data)
        hyper_params = causal_defaults_params["FBPROPHET"]["hyper_params"]
        hyper_params["seasonal_periods"] = seasonal_period
        country_code = params_from_ui["FBPROPHET"]["algo_params"]["holiday_country_code"]
        prophetModel = Prophet(add_seasonalities=hyper_params, country_holidays=country_code)
        return prophetModel

    def _rnn_lstm_gru_ForecastHyperparams(self,RnnModel,parameters,causal_defaults_params,params_from_ui,train_chunk_size,val_chunk_size,tsidsBatchDataTrain,tsidsBatchDataVal):
        default_hyper_params = causal_defaults_params["RNN"]["hyper_params"]
        input_chunk_sizes = [default_hyper_params["input_chunk_length"].append(train_chunk_size * i) for i in range(2,4)]
        output_chunk_sizes = [default_hyper_params["training_length"].append(val_chunk_size * i)for i in range(1,4)]
        default_hyper_params["n_epochs"].append(params_from_ui["RNN"]["algo_params"]["n_epochs"])
        default_hyper_params["hidden_dim"].append(params_from_ui["RNN"]["algo_params"]["hidden_dim"])
        default_hyper_params["n_rnn_layers"].append(params_from_ui["RNN"]["algo_params"]["n_rnn_layers"])
        default_hyper_params["dropout"].append(params_from_ui["RNN"]["algo_params"]["dropout"])
        
        hyper_params = RnnModel.gridsearch(parameters=default_hyper_params,series=tsidsBatchDataTrain[0],val_series=tsidsBatchDataVal[0])
        parameters = ForecastDartsMethods().preprocess_input_params_with_hyperparams(parameters,hyper_params[1])
        
        
        RnnModel = RNNModel(
            model=parameters["model"],
            n_epochs=parameters["n_epochs"],
            hidden_dim= parameters["hidden_dim"],
            n_rnn_layers= parameters["n_rnn_layers"],
            random_state=20,
            optimizer_kwargs=parameters["optimizer_kwargs"],
            training_length=parameters["training_length"],
            input_chunk_length=parameters["input_chunk_length"],
            likelihood=GaussianLikelihood(),
        )

        return RnnModel

    def THETAHyperparams(self,tsid_train_data, tsid_val_data, prediction_period:int):

        thetas = 2 - np.linspace(-10, 10, 50)
        best_mape = float("inf")
        best_theta = 0

        for theta in thetas:
            model = Theta(theta)
            model.fit(tsid_train_data)
            pred_theta = model.predict(prediction_period)
            res = mape(tsid_val_data, pred_theta)

            if res < best_mape:
                best_mape = res
                best_theta = theta
                
        best_theta_model = Theta(best_theta)
        best_theta_model.fit(tsid_train_data)

        return best_theta_model

    def FFTHyperparams(self, fftModel, parameters,tsid_train_data, tsid_val_data, causal_defaults_params):
        default_hyper_params = causal_defaults_params["FFT"]["hyper_params"]
        hyper_params = fftModel.gridsearch(parameters=default_hyper_params, series=tsid_train_data, val_series=tsid_val_data )
        parameters = ForecastDartsMethods().preprocess_input_params_with_hyperparams(parameters,hyper_params[1])
        fftModel = FFT(
            nr_freqs_to_keep=parameters["nr_freqs_to_keep"],
            trend=parameters["trend"],
            trend_poly_degree=parameters["trend_poly_degree"]
        )

        return fftModel
        

class returnForecast:

    """
       This function takes the model object and period parameters and return result in json format.
    """
    def __init__(self):
        self.scaler = Scaler()

    def _get_nbeats_forecast(self, model, ts_id_data, prediction_period):
        nbeatPrediction = model.predict(series=ts_id_data, n=prediction_period)
        nbeatPrediction = self.scaler.inverse_transform(nbeatPrediction)
        json_data = eval(nbeatPrediction.to_json())
        del json_data['columns']
        print(json_data)
   
    def _get_tcn_forecast(self, model, ts_id_data, prediction_period):
        tcnPrediction = model.predict(n=prediction_period, series=ts_id_data, num_samples=50)
        tcnPrediction = self.scaler.inverse_transform(tcnPrediction)
        json_data = {"data":tcnPrediction.values().tolist()}
        print(json_data)

    def _get_prophet_forecast(self,model, prediction_period):
        prophetPrediction = model.predict(n=prediction_period)
        prophetPrediction = self.scaler.inverse_transform(prophetPrediction)
        json_data = eval(prophetPrediction.to_json())
        del json_data['columns']
        print(json_data)

    def _get_rnn_lstm_gru_forecast(self,model, ts_id_data, prediction_period):
        rnnPrediction = model.predict(n=prediction_period, series=ts_id_data, num_samples=50)
        rnnPrediction = self.scaler.inverse_transform(rnnPrediction)
        json_data = {"data":rnnPrediction.values().tolist()}
        print(json_data)

    def _get_theta_forecast(self,model, prediction_period):
        thetaForecast = model.predict(n=prediction_period)
        data = {"data":list(thetaForecast.values().flatten())}
        print(data)

    def _get_fft_forecast(self,model, prediction_period):
        fftForecast = model.predict(n=prediction_period)
        data = {"data":list(fftForecast.values().flatten())}
        print(data)


class Datapreprocess:
    pass


class ForecastDartsMethods(returnForecast,Hyperparameterts):
    def __init__(self):
        self.scaler = Scaler()

    def split_data_train_validation(self, data):
        series_data = TimeSeries.from_dataframe(data)
        one_4th_part_size_len = len(series_data) // 4 
        chunks = one_4th_part_size_len // 3
        train_chunk_size  =  chunks * 2
        val_chunk_size =  chunks * 1
        train, val = series_data[:-one_4th_part_size_len], series_data[-one_4th_part_size_len:] 
        return train, val, train_chunk_size, val_chunk_size

    def check_seasonality(self,ts_id_data):
        for m in range(2, 25):
            is_seasonal, period = check_seasonality(ts_id_data, m=m, alpha=0.05)
            if is_seasonal:
                print("There is seasonality of order {}.".format(period))
                return period
            else:
                return int(12)

    def preprocess_data(self, data):
        """
        function is a function that takes in a dataframe and split data to train test data
        Then converts data to darts series type and scale it between -1 and 1. 
        """
        series_train_data, series_val_data, train_chunk_size, val_chunk_size = self.split_data_train_validation(data)
        scaled_train_data,scaled_val_data = self.scaler.fit_transform([series_train_data, series_val_data])
        return series_train_data, series_val_data, scaled_train_data, scaled_val_data, train_chunk_size, val_chunk_size

    def preprocess_input_params_with_default(self,causal_defaults_params, params_from_ui):
        """
         preprocess_input_params_with_default is a function that takes causal_defaults_params and 
         params_from_ui. If any paramenter value from ui is None then it replaces with the default value.
        """
        for param in causal_defaults_params:
            try:
                if params_from_ui[param] == None:
                    params_from_ui[param] = causal_defaults_params[param]
            except KeyError as err:
                pass
            except Exception as e:
                raise e
        return params_from_ui

    def preprocess_input_params_with_hyperparams(self, params_from_ui, hyperparams):
        """
         preprocess_input_params_with_hyperparams is a function that takes params_from_ui and 
         hyperparams from hyper_parameter tunning function . It replaces ui parameter values
         with hyperparameter value.
        """
        for param in hyperparams:
            try:
                params_from_ui[param] = hyperparams[param]
            except Exception as e:
                raise e
        return params_from_ui



    def NBEATSForecast(self,tsidsBatchDataTrain,tsidsBatchDataVal, params_from_ui, causal_defaults_params, train_chunk_size, val_chunk_size ):
        
        """ 
            NBEATSForecast is a function that takes in a dataframe and returns a dataframe with the
            forecasted values. The dataframe is split into chunks of length input_chunk_length
            (train_data_size) and output_chunk_length(test_data_size) by algorithm itself.
            The chunks are then fed into the NBEATS model for multiple times and the forecasted values for
            prediction_period are returned in form of dataframe. columns = ['index','data'].

        """


        
        try:
            parameters = self.preprocess_input_params_with_default(
                            causal_defaults_params["NBEATS"]["algo_params"],
                            params_from_ui["NBEATS"]["algo_params"]
                        )
            
            nbeatModel = NBEATSModel(
                    random_state = 42,
                    n_epochs = parameters["n_epochs"],
                    num_stacks = parameters["num_stacks"],
                    num_blocks = parameters["num_blocks"],
                    num_layers = parameters["num_layers"],
                    batch_size = parameters["batch_size"],
                    layer_widths = parameters["layer_widths"],
                    optimizer_kwargs = parameters["optimizer_kwargs"],
                    input_chunk_length = parameters["input_chunk_length"],
                    output_chunk_length = parameters["output_chunk_length"],
                    generic_architecture = parameters["generic_architecture"],
                    trend_polynomial_degree = parameters["trend_polynomial_degree"],
                    expansion_coefficient_dim = parameters["expansion_coefficient_dim"]
                )

            if params_from_ui["NBEATS"]["hyper_param_tunning"]:
                nbeatModel = self._NBEATSHyperparams(nbeatModel,parameters,causal_defaults_params,params_from_ui,train_chunk_size,val_chunk_size,tsidsBatchDataTrain,tsidsBatchDataVal)


            if len(tsidsBatchDataTrain) == 1:
                tsidsBatchDataTrain = tsidsBatchDataTrain[0]

            nbeatModel.fit(tsidsBatchDataTrain , verbose=True)

            if isinstance(tsidsBatchDataTrain, list):
                for ts_id_data in tsidsBatchDataTrain:
                    self._get_nbeats_forecast(nbeatModel, ts_id_data, parameters["prediction_period"])
            else:
                self._get_nbeats_forecast(nbeatModel, tsidsBatchDataTrain, parameters["prediction_period"])

        except Exception as e:
            print(e)
            return None

    def TCNForecast(self, tsidsBatchDataTrain, tsidsBatchDataVal, params_from_ui, causal_defaults_params, train_chunk_size, val_chunk_size):
        """ 
            TCNForecast is a function that takes in a dataframe and returns a dataframe with the
            forecasted values. The dataframe is split into chunks of length input_chunk_length
            (train_data_size) and output_chunk_length(test_data_size) by algorithm itself.
            The chunks are then fed into the TCN model for multiple times and the forecasted values for
            prediction_period are returned in form of dataframe. columns = ['data'].
        """


        try:
            parameters = self.preprocess_input_params_with_default(
                causal_defaults_params["TCN"]["algo_params"],
                params_from_ui["TCN"]["algo_params"]
            )


            tcnModel = TCNModel(
                input_chunk_length=parameters["input_chunk_length"],
                output_chunk_length=parameters["output_chunk_length"],
                kernel_size=parameters["kernel_size"],
                num_filters=parameters["num_filters"],
                weight_norm= parameters["weight_norm"],
                random_state=42,
                dilation_base=parameters["dilation_base"],
                num_layers=parameters["num_layers"],
                dropout=parameters["dropout"],
                batch_size=parameters["batch_size"],
                n_epochs=parameters["n_epochs"],
                likelihood=LaplaceLikelihood(),
            )

            if params_from_ui["TCN"]["hyper_param_tunning"]:
                tcnModel = self._TCNHyperparams(tcnModel,parameters,causal_defaults_params,params_from_ui,train_chunk_size,val_chunk_size,tsidsBatchDataTrain,tsidsBatchDataVal)

            if len(tsidsBatchDataTrain) == 1:
                tsidsBatchDataTrain = tsidsBatchDataTrain[0]
                
            tcnModel.fit(tsidsBatchDataTrain, verbose=True)

            if isinstance(tsidsBatchDataTrain, list):
                for ts_id_data in tsidsBatchDataTrain:
                    self._get_tcn_forecast(tcnModel, ts_id_data, parameters["prediction_period"])
            else:
                self._get_tcn_forecast(tcnModel, tsidsBatchDataTrain, parameters["prediction_period"])

        except Exception as e:
            raise e

    def ProphetForecast(self, tsidsBatchDataTrain, params_from_ui, causal_defaults_params):
        """
            ProphetForecast is a function that takes in a dataframe and returns a dataframe with the
            forecasted values. It does not allow multiple time series at single time (No batch processing). 
            You need to iterate over for each time series
        """
        try:
            for ts_id_data in tsidsBatchDataTrain:
                prophetModel = Prophet()
                if params_from_ui["FBPROPHET"]["hyper_param_tunning"]:
                    prophetModel = self._ProphetHyperparams(ts_id_data,causal_defaults_params,params_from_ui)

                prophetModel.fit(ts_id_data)
                self._get_prophet_forecast(prophetModel, params_from_ui["FBPROPHET"]["algo_params"]["prediction_period"])
        except Exception as e:
            raise e

    def rnn_lstm_gru_Forecast(self, tsidsBatchDataTrain, tsidsBatchDataVal, params_from_ui, causal_defaults_params, train_chunk_size, val_chunk_size):
        """
            rnn_lstm_gru_Forecast is a function that takes in a dataframe and returns a dataframe with the
            forecasted values. It allows multiple time series at single time and also supports for regression
        """
        
        try:

            parameters = self.preprocess_input_params_with_default(
                causal_defaults_params["RNN"]["algo_params"],
                params_from_ui["RNN"]["algo_params"]
            )


            RnnModel = RNNModel(
                model=parameters["model"],
                n_epochs=parameters["n_epochs"],
                hidden_dim= parameters["hidden_dim"],
                n_rnn_layers= parameters["n_rnn_layers"],
                random_state=20,
                optimizer_kwargs=parameters["optimizer_kwargs"],
                training_length=parameters["training_length"],
                input_chunk_length=parameters["input_chunk_length"],
                likelihood=GaussianLikelihood(),
            )

            if params_from_ui["RNN"]["hyper_param_tunning"]:
                RnnModel = self._rnn_lstm_gru_ForecastHyperparams(RnnModel,parameters,causal_defaults_params,params_from_ui,train_chunk_size,val_chunk_size,tsidsBatchDataTrain,tsidsBatchDataVal)

            if len(tsidsBatchDataTrain) == 1:
                tsidsBatchDataTrain = tsidsBatchDataTrain[0]

            RnnModel.fit(tsidsBatchDataTrain , verbose=True)

            if isinstance(tsidsBatchDataTrain, list):
                for ts_id_data in tsidsBatchDataTrain:
                    self._get_nbeats_forecast( RnnModel, ts_id_data, parameters["prediction_period"])
            else:
                self._get_nbeats_forecast(RnnModel, tsidsBatchDataTrain, parameters["prediction_period"])


        except Exception as e:
            raise e

    def ThetaForecast(self, tsidsBatchDataTrain, tsidsBatchDataVal, params_from_ui):
        """
            ThetaForecast is a function that takes in a dataframe and returns a dataframe with the
            forecasted values. It does not allow multiple time series at single time (No batch processing). 
            You need to iterate over for each time series
        """
        thetaModel = Theta()
        prediction_period = params_from_ui["THETA"]["algo_params"]["prediction_period"]

        for i in range(len(tsidsBatchDataTrain)):
            if params_from_ui["THETA"]["hyper_param_tunning"]:
                thetaModel =self.THETAHyperparams(tsidsBatchDataTrain[i], tsidsBatchDataVal[i], prediction_period)
            thetaModel.fit(tsidsBatchDataTrain[i])
            self._get_theta_forecast(thetaModel, prediction_period)

    def FFTForecast(self, tsidsBatchDataTrain, tsidsBatchDataVal, params_from_ui, causal_defaults_params):
        
        parameters = self.preprocess_input_params_with_default(
                        causal_defaults_params["FFT"]["algo_params"],
                        params_from_ui["FFT"]["algo_params"]
                    )
        
        prediction_period = parameters["prediction_period"]
        
        fftModel = FFT(
            nr_freqs_to_keep=parameters["nr_freqs_to_keep"],
            trend=parameters["trend"],
            trend_poly_degree=parameters["trend_poly_degree"]
        )
        
        for i in range(len(tsidsBatchDataTrain)):
            if params_from_ui["FFT"]["hyper_param_tunning"]:
                fftModel = self.FFTHyperparams(fftModel, parameters, tsidsBatchDataTrain[i], tsidsBatchDataVal[i], causal_defaults_params)
            fftModel.fit(tsidsBatchDataTrain[i])
            self._get_fft_forecast(fftModel, prediction_period)


def rundart():
    df = pd.read_csv('AirPassengers.csv', parse_dates=["Month"])
    df = df.set_index('Month')
    obj = ForecastDartsMethods()
    series_train_data, series_val_data, scaled_train_data, scaled_val_data, train_chunk_size, val_chunk_size = obj.preprocess_data(df)
    ts_id_data_list_train = [scaled_train_data]
    ts_id_data_list_val = [scaled_val_data]
    ts_id_series_train = [series_train_data]
    ts_id_series_val = [series_val_data]
    from forecasting_params import ui_data, causal_defaults_params 

    # ts_id_data_list.append(scaled_data)
    # lst.append(scaled_data)

    obj.NBEATSForecast(ts_id_data_list_train,ts_id_data_list_val, ui_data, causal_defaults_params, train_chunk_size, val_chunk_size )
    # obj.TCNForecast(ts_id_data_list_train,ts_id_data_list_val, ui_data, causal_defaults_params, train_chunk_size, val_chunk_size)
    # obj.ProphetForecast(ts_id_data_list_train, ui_data, causal_defaults_params)
    # obj.rnn_lstm_gru_Forecast(ts_id_data_list_train,ts_id_data_list_val, ui_data, causal_defaults_params, train_chunk_size, val_chunk_size)
    # obj.ThetaForecast(ts_id_series_train,ts_id_series_val, ui_data)
    # obj.FFTForecast(ts_id_data_list_train,ts_id_data_list_val, ui_data, causal_defaults_params)

if __name__ == '__main__':
    rundart()
    # pass




