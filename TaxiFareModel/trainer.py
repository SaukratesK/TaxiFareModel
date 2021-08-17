# imports

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.set_pipeline()
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''
        from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer


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
        print(rmse)
        return rmse


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
    self.X = X_train
    self.y = y_train

    new_trainer.run()

    # evaluate
    rmse = new_trainer.evaluate(X_test, y_test)
    print(f'RMSE du modèle : {rmse}')
