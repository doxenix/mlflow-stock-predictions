# mlflow-stock-predictions
<img src="https://www.mlflow.org/docs/latest/_static/MLflow-logo-final-black.png" width="200">

Hello and welcome!

I will show you how you can try to predict stock market prices. We'll use mlflow tool. I highly recommended to check official documentation: [MlFlow documentation](https://mlflow.org/docs/latest/index.html).

Mlflow is great MlOps tool which allow you to monitor and experiment with your machine learing models. You can also easly share your created model with others using conda or docker. Finally, you can make deployments and then getting prediciotns via API.

Keep in mind that this is not financial advice!

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

Then create your experiment. I called **USD_PLN_daily**. After that check ID (it will be necessary to add or change it in `pipline.py` file.

Please check `params.py`. You will find all parameters. Don't hesitate to change and experiment with them!

Run: `python pipline.py` and wait. 

Great, you trained your model! You also noticed that you have `mlruns` and `models` folders.


