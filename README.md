# mlflow-stock-predictions
<img src="https://www.mlflow.org/docs/latest/_static/MLflow-logo-final-black.png" width="200">

Hello and welcome!

I will show you how you can try to predict stock market prices. We'll use mlflow tool. I highly recommend to check official documentation: [MlFlow documentation](https://mlflow.org/docs/latest/index.html).

Mlflow is great MlOps tool which allows you to monitor and experiment with your machine learing models. You can also easly share your created model with others using conda or docker. Finally, you can make deployments and then getting predictions via API.

Keep in mind is not financial advice!

Setup environment
----------------------------------
First clone my repo on your computer:
* `git clone git@github.com:doxenix/mlflow-stock-predictions.git`

We'll work locally, so make sure you already have installed all needed libraries. If you using conda or python 3.6+ just:
* `pip install -r req.txt`

Step 1 - Train and Track the model
-------------------

launch `mlflow ui --backend-store-uri sqlite:///mlruns.db`
Got to `http://127.0.0.1:5000`

You can also run just: `mlflow ui` but without add sqlite database (or any other) you will not be able to store your models and push to productions.

Then create your experiment. I called **USD_PLN_daily**. 

<img src="https://github.com/doxenix/mlflow-stock-predictions/blob/main/readme_screens/create_experiment.jpg">

Now, check `params.py`. You will find all parameters. Don't hesitate to change and experiment with them!

In another terminal, run: `python pipline.py` and wait. Your model shoud start to train.

Great, you trained your model! You also noticed that you have `mlruns` and `models` folders. Start making some experiments and `pipline.py` as many time as you want!

You can also explore your all models in `Mlflow`. You can also check your result predictions on test data. The plot is available in your folders. You can also check it via **Mlflow** page.

<img src="https://github.com/doxenix/mlflow-stock-predictions/blob/main/readme_screens/USD_PLN_daily_prediction_fig.jpg">

Step 2 - Register model to production
-------------------

Pick your best trained model and go to `http://127.0.0.1:5000` again. Click on yor model and register it.

<img src="https://github.com/doxenix/mlflow-stock-predictions/blob/main/readme_screens/register_model.jpg">

Then, go to **Models** tab, click on your model and push to production stage.

<img src="https://github.com/doxenix/mlflow-stock-predictions/blob/main/readme_screens/push_to_production.jpg">

Step 3 - Deploy and make predictions
-------------------

Run `./deploy.sh`

This launches a gunicorn server serving at the localhost `127.0.0.1:5000`. Now you can score locally
on the deployed produciton model as a REST point.
 
From another terminal send a POST request with our fresh data
  * ```python run_prediction.py```

Inside your terminal you will see predicted price.

Step 4 - Serving model as Docker image (testing phase!)
-------------------

`docker build -t mlflow_image -f Dockerfile .`

After that you can try running your experiment:

`mlflow run . --no-conda --param-list config=0`

Bonus
-------------------

`SMA.py` file requires `oanda.cfg` file with your API key. You have to create accouunt on [OANDA](https://www.oanda.com/eu-en/).

Check [OANDA API Guide](https://developer.oanda.com/rest-live-v20/introduction/) how to generate yor API key.

You will also need `tpqoa` library. For more details check: [tpqoa](https://github.com/yhilpisch/tpqoa)
