multiclass_keras
==============================

Test project for different multiclass experiments.

Project Organization
------------
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
	├── config
	│   └── experiments
	|
	├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
	|
	├── docs               <- A default Sphinx project; see sphinx-doc.org for details	
	│
	├── models             <- Trained and serialized models, model predictions, or model summaries
    │
	├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
	├── references         <- Data dictionaries, manuals, and all other explanatory materials.
	|    
	├── reports			   <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── figures        <- Generated graphics and figures to be used in reporting
	│   ├── logs
	│   └── tensorboard_logs
	|
	├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
	├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
	|
	├── src
	│   ├── app
	│   │   ├── app.py
	│   │   ├── static
	│   │   └── templates
	|   |
	│   ├── data
	│   │   ├── data_utils.py
	│   │   ├── reports
	│   │   ├── patch_all_images.py
	│   │   └── split_dataset.py
	│   |
	|   ├── features
	│   │   └── build_features.py
	│   |
	│   ├── models
	│   │   ├── evaluate_model.py
	│   │   ├── model_manager.py
	│   │   ├── model_utils.py
	│   │   ├── predict_model.py
	│   │   ├── reports
	│   │   ├── Result.py
	│   │   └── train_model.py
	│   |
	│   ├── utils_io.py
	│   └── visualization
	│       ├── utils.py
	│       └── visualize.py
	|
	└── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

Contributions:
--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

#Basic Setup: 

Setup native with OSX and Ubuntu
------------

- Precondition Python 3.6 locally installed
(e.g.:  <a target="_blank" href="https://www.anaconda.com/download/#macos">Anaconda</a>)


- clone repository
```
git clone https://github.com/minority4u/multiclass_keras
cd multiclass_keras
```

- Install all Dependencies, and start the app (all in one), works with OSX and Linux
```
make run
```

Setup native with Windows
------------

- Precondition Python 3.6 locally installed
(e.g.:  <a target="_blank" href="https://www.anaconda.com/download/#macos">Anaconda</a>)

- Clone Repository
```
git clone https://github.com/minority4u/keras_flask_deployment
cd keras_flask_deployment
```

```
pip install virtualenv
virtualenv venv
venv\Scripts\activate
pip install -r requirements.txt
python src\app\app.py
```

Setup Docker
------------

- Precondition: Installed Docker Deamon (e.g.:  <a target="_blank" href="https://docs.docker.com/install/">Docker CE</a>)

- Make sure you have docker-compose installed (e.g.:  <a target="_blank" href="https://docs.docker.com/compose/install/">Docker-Compose</a>)

- Clone Repository
```
git clone https://github.com/minority4u/keras_flask_deployment
cd keras_flask_deployment
```
- Create and run Docker-Container
```
docker-compose -up docker-compose.yml
```


Train new Model
------------

- OSX/Linux
```
make train
```

- Windows
```
Python src\models\train_model.py
```
____________
#Experiments and customisation:

Included experiments
------------
###Baseline

###Bigger Model

###Smaller Model

###L2 regularisation

###Dropout

###Data augmentation

###Batch normalisation

###Bigger patchsize

###Smaller patchsize


Setup custom config file
------------



Setup custom model
------------



