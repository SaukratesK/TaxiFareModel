import mlflow
from  mlflow.tracking import MlflowClient
from memoized_property import memoized_property

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class Trainer():

    MLFLOW_URI = "https://mlflow.lewagon.co/"

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.set_pipeline()
        self.X = X
        self.y = y
        self.experiment_name = "[FR] [Nantes] [saukratesk] TaxiFareModel v1"


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''



        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])

        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

        self.pipeline = pipe

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        from TaxiFareModel.utils import compute_rmse

        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric("rmse", rmse)
        #print(f'{self.pipeline.get_params()}')
        self.mlflow_log_param("model", "linear")
        print(rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


from TaxiFareModel.data import get_data, clean_data

if __name__ == "__main__":
    # get data
    df = get_data()

    # clean data
    df_cleaned = clean_data(df)

    # set X and y
    y = df.pop("fare_amount")
    X = df

    new_trainer = Trainer(X,y)

    # hold out
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # train
    new_trainer.set_pipeline()
    new_trainer.X = X_train
    new_trainer.y = y_train

    new_trainer.run()

    # evaluate
    rmse = new_trainer.evaluate(X_test, y_test)
    print(f'RMSE du modèle : {rmse}')
