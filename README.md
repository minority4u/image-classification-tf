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

- Install all Dependencies, and start the app locally in an virtual environment (all in one), works with OSX and Linux
```
make run
```

- Create a docker image, install all requirements in this container and start the service
```
make docker_run
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
1. Create base-image
	docker-compose -f ./docker-compose_base.yml -p multiclasskeras build
	
2. Create classification-image
	docker-compose -f ./docker-compose.yml build --no-cache
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

Dataset
------------
	# Images:
	ETechnik: 3303
	Tabellen: 2574
	Fliesbilder: 208
	Stahlbau: 300
	Lagepläne: 121
	--------------
	Sum = 6506
	
	# Patches, filtered, percentage
	ETechnik: 25917, 2096, 8%
	Tabellen: 21536, 30, 0.1%
	Fliesbilder: 10437, 2660, 25%
	Stahlbau: 9159, 1458, 16%
	Lagepläne: 5358, 1320, 24%
	---------------
	Sum = 72407, 7564, 10%

	Average # patches per class
	ETechnik: 8.5
	Tabellen: 8.4
	Fliesbilder: 63
	Stahlbau: 35.4
	Lagepläne: 55.2
	--------------
	Sum = 12.3



--> Results in 387 Steps x Batchsize 128 = 49.536 Patche

Included experiments
------------
###Baseline


Patch-level

 			precision	recall	f1-score	support 
 			
 	Etechnik	0.75		0.87	0.81		2390.8000000000206 
 	Fliesbilder	0.66		0.76 	0.71		2390.800000000019 
 	Lageplan 	0.81		0.52 	0.63		2390.7999999999056 
 	Stahlbau 	0.92		0.69 	0.79		2390.7999999999406 
 	Tabellen 	0.75		0.98 	0.85		2390.7999999998724 
 
 	micro avg 	0.77 		0.77 	0.77 		11953.999999999758 
 	macro avg 	0.78 		0.77 	0.76 		11953.999999999758 
	weighted avg 	0.78 		0.77 	0.76 		11953.999999999758 

	Accuracy: 0.7652364075738796

![alt text](https://github.com/minority4u/multiclass_keras/blob/master/reports/conclusion/Logs_baseline/patch_confusion_matrix_1.png "Confusion Matrix")	
![alt text](https://github.com/minority4u/multiclass_keras/blob/master/reports/conclusion/Logs_baseline/patch_confusion_matrix_normalized_1.png "Confusion Matrix normalized")
------
Image-level
	
				precision	recall	f1-score	support

    Etechnik		1.00		0.99	1.00		195.20000000000107
 	Fliesbilder		0.90		1.00	0.95		195.20000000000002
    Lageplan		1.00		0.86	0.93		195.19999999999993
    Stahlbau		1.00		1.00	1.00		195.19999999999996
    Tabellen		0.96		1.00	0.98		195.1999999999998

   	micro avg		0.97		0.97	0.97		976.0000000000008
   	macro avg		0.97		0.97	0.97		976.0000000000008
	weighted avg		0.97		0.97	0.97		976.0000000000008

	Accuracy: 0.9706977406466091

![alt text](https://github.com/minority4u/multiclass_keras/blob/master/reports/conclusion/Logs_baseline/image_confusion_matrix_1.png "Confusion Matrix")	
![alt text](https://github.com/minority4u/multiclass_keras/blob/master/reports/conclusion/Logs_baseline/image_confusion_matrix_normalized.png "Confusion Matrix normalized")
------
Description
TODO:
Description from Raw Data to prediction

###Summary
	__________________________________________________________________________________________________
	___________________________________________________________________________________________________
	Layer (type)                    Output Shape         Param #     Connected to                     
	==================================================================================================
	Input_1 (InputLayer)            (None, 224, 224, 3)  0                                            
	___________________________________________________________________________________________________
	Convolution2D_236 (Conv2D)      (None, 112, 112, 32) 896         Input_1[0][0]                    
	___________________________________________________________________________________________________
	Convolution2D_235 (Conv2D)      (None, 112, 112, 32) 9248        Convolution2D_236[0][0]          
	___________________________________________________________________________________________________
	Convolution2D_237 (Conv2D)      (None, 110, 110, 64) 18496       Convolution2D_235[0][0]          
	___________________________________________________________________________________________________
	MaxPooling2D_69 (MaxPooling2D)  (None, 55, 55, 64)   0           Convolution2D_237[0][0]          
	___________________________________________________________________________________________________
	Convolution2D_238 (Conv2D)      (None, 53, 53, 80)   46160       MaxPooling2D_69[0][0]            
	___________________________________________________________________________________________________
	Convolution2D_239 (Conv2D)      (None, 26, 26, 192)  138432      Convolution2D_238[0][0]          
	___________________________________________________________________________________________________
	Convolution2D_1 (Conv2D)        (None, 26, 26, 32)   6176        Convolution2D_239[0][0]          
	___________________________________________________________________________________________________
	Convolution2D_4 (Conv2D)        (None, 26, 26, 64)   6208        Convolution2D_1[0][0]            
	___________________________________________________________________________________________________
	Convolution2D_2 (Conv2D)        (None, 26, 26, 32)   6176        Convolution2D_239[0][0]          
	___________________________________________________________________________________________________
	Convolution2D_6 (Conv2D)        (None, 26, 26, 64)   12352       Convolution2D_4[0][0]            
	___________________________________________________________________________________________________
	MaxPooling2D_7 (MaxPooling2D)   (None, 13, 13, 192)  0           Convolution2D_239[0][0]          
	___________________________________________________________________________________________________
	Convolution2D_8 (Conv2D)        (None, 13, 26, 64)   6208        Convolution2D_2[0][0]            
	___________________________________________________________________________________________________
	Convolution2D_5 (Conv2D)        (None, 13, 26, 64)   12352       Convolution2D_6[0][0]            
	___________________________________________________________________________________________________
	Convolution2D_10 (Conv2D)       (None, 13, 13, 64)   12352       MaxPooling2D_7[0][0]             
	___________________________________________________________________________________________________
	Convolution2D_9 (Conv2D)        (None, 13, 13, 64)   12352       Convolution2D_8[0][0]            
	___________________________________________________________________________________________________
	Convolution2D_7 (Conv2D)        (None, 13, 13, 64)   12352       Convolution2D_5[0][0]            
	___________________________________________________________________________________________________
	Convolution2D_3 (Conv2D)        (None, 13, 13, 32)   6176        Convolution2D_239[0][0]          
	___________________________________________________________________________________________________
	merge_1 (Concatenate)           (None, 13, 13, 224)  0           Convolution2D_10[0][0]           
									 Convolution2D_9[0][0]            
									 Convolution2D_7[0][0]            
									 Convolution2D_3[0][0]            
	___________________________________________________________________________________________________
	Convolution2D_121 (Conv2D)      (None, 13, 13, 128)  28800       merge_1[0][0]                    
	___________________________________________________________________________________________________
	Convolution2D_122 (Conv2D)      (None, 13, 13, 128)  49280       Convolution2D_121[0][0]          
	___________________________________________________________________________________________________
	Convolution2D_126 (Conv2D)      (None, 13, 13, 128)  28800       merge_1[0][0]                    
	___________________________________________________________________________________________________
	Convolution2D_123 (Conv2D)      (None, 13, 13, 128)  49280       Convolution2D_122[0][0]          
	___________________________________________________________________________________________________
	Convolution2D_127 (Conv2D)      (None, 7, 13, 128)   49280       Convolution2D_126[0][0]          
	___________________________________________________________________________________________________
	MaxPooling2D_15 (MaxPooling2D)  (None, 7, 7, 224)    0           merge_1[0][0]                    
	___________________________________________________________________________________________________
	Convolution2D_124 (Conv2D)      (None, 7, 13, 128)   49280       Convolution2D_123[0][0]          
	___________________________________________________________________________________________________
	Convolution2D_128 (Conv2D)      (None, 7, 7, 128)    49280       Convolution2D_127[0][0]          
	___________________________________________________________________________________________________
	Convolution2D_129 (Conv2D)      (None, 7, 7, 128)    28800       MaxPooling2D_15[0][0]            
	___________________________________________________________________________________________________
	Convolution2D_125 (Conv2D)      (None, 7, 7, 128)    49280       Convolution2D_124[0][0]          
	___________________________________________________________________________________________________
	Convolution2D_130 (Conv2D)      (None, 7, 7, 128)    28800       merge_1[0][0]                    
	___________________________________________________________________________________________________
	merge_10 (Concatenate)          (None, 7, 7, 512)    0           Convolution2D_128[0][0]          
									 Convolution2D_129[0][0]          
									 Convolution2D_125[0][0]          
									 Convolution2D_130[0][0]          
	___________________________________________________________________________________________________
	MaxPooling2D_70 (MaxPooling2D)  (None, 1, 1, 512)    0           merge_10[0][0]                   
	___________________________________________________________________________________________________
	Flatten_5 (Flatten)             (None, 512)          0           MaxPooling2D_70[0][0]            
	___________________________________________________________________________________________________
	Dense_14 (Dense)                (None, 5)            2565        Flatten_5[0][0]                  
	___________________________________________________________________________________________________
	Dense_2 (Dense)                 (None, 5)            30          Dense_14[0][0]                   
	==================================================================================================
	
	Total params: 719,411
	Trainable params: 719,411
	Non-trainable params: 0

###Graph:

![alt text](https://github.com/minority4u/multiclass_keras/blob/master/reports/conclusion/Logs_baseline/Graph_inception_v3.png "Baseline-Model")



##Bigger Model

##Smaller Model

Patch-Level

TODO

![alt text](https://github.com/minority4u/multiclass_keras/blob/master/reports/conclusion/Logs_1inceptionmodule/patch_confusion_matrix_1.png "Confusion Matrix")	
![alt text](https://github.com/minority4u/multiclass_keras/blob/master/reports/conclusion/Logs_1inceptionmodule/patch_confusion_matrix_normalized_1.png "Confusion Matrix normalized")
------

Image-Level

TODO

![alt text](https://github.com/minority4u/multiclass_keras/blob/master/reports/conclusion/Logs_1inceptionmodule/image_confusion_matrix_1.png "Confusion Matrix")	
![alt text](https://github.com/minority4u/multiclass_keras/blob/master/reports/conclusion/Logs_1inceptionmodule/image_confusion_matrix_normalized_1.png "Confusion Matrix normalized")

------
Description
TODO:

### Summery
	__________________________________________________________________________________________________
	Layer (type)                    Output Shape         Param #     Connected to
	==================================================================================================
	Input_1 (InputLayer)            (None, 224, 224, 3)  0
	__________________________________________________________________________________________________
	Convolution2D_236 (Conv2D)      (None, 112, 112, 32) 896         Input_1[0][0]
	__________________________________________________________________________________________________
	Convolution2D_235 (Conv2D)      (None, 112, 112, 32) 9248        Convolution2D_236[0][0]
	__________________________________________________________________________________________________
	Convolution2D_237 (Conv2D)      (None, 110, 110, 64) 18496       Convolution2D_235[0][0]
	__________________________________________________________________________________________________
	MaxPooling2D_69 (MaxPooling2D)  (None, 55, 55, 64)   0           Convolution2D_237[0][0]
	__________________________________________________________________________________________________
	Convolution2D_238 (Conv2D)      (None, 53, 53, 80)   46160       MaxPooling2D_69[0][0]
	__________________________________________________________________________________________________
	Convolution2D_239 (Conv2D)      (None, 26, 26, 192)  138432      Convolution2D_238[0][0]
	__________________________________________________________________________________________________
	Convolution2D_1 (Conv2D)        (None, 26, 26, 32)   6176        Convolution2D_239[0][0]
	__________________________________________________________________________________________________
	Convolution2D_4 (Conv2D)        (None, 26, 26, 64)   6208        Convolution2D_1[0][0]
	__________________________________________________________________________________________________
	Convolution2D_2 (Conv2D)        (None, 26, 26, 32)   6176        Convolution2D_239[0][0]
	__________________________________________________________________________________________________
	Convolution2D_6 (Conv2D)        (None, 26, 26, 64)   12352       Convolution2D_4[0][0]
	__________________________________________________________________________________________________
	MaxPooling2D_7 (MaxPooling2D)   (None, 13, 13, 192)  0           Convolution2D_239[0][0]
	__________________________________________________________________________________________________
	Convolution2D_8 (Conv2D)        (None, 13, 26, 64)   6208        Convolution2D_2[0][0]
	__________________________________________________________________________________________________
	Convolution2D_5 (Conv2D)        (None, 13, 26, 64)   12352       Convolution2D_6[0][0]
	__________________________________________________________________________________________________
	Convolution2D_10 (Conv2D)       (None, 13, 13, 64)   12352       MaxPooling2D_7[0][0]
	__________________________________________________________________________________________________
	Convolution2D_9 (Conv2D)        (None, 13, 13, 64)   12352       Convolution2D_8[0][0]
	__________________________________________________________________________________________________
	Convolution2D_7 (Conv2D)        (None, 13, 13, 64)   12352       Convolution2D_5[0][0]
	__________________________________________________________________________________________________
	Convolution2D_3 (Conv2D)        (None, 13, 13, 32)   6176        Convolution2D_239[0][0]
	__________________________________________________________________________________________________
	merge_1 (Concatenate)           (None, 13, 13, 224)  0           Convolution2D_10[0][0]
																	 Convolution2D_9[0][0]
																	 Convolution2D_7[0][0]
																	 Convolution2D_3[0][0]
	__________________________________________________________________________________________________
	MaxPooling2D_70 (MaxPooling2D)  (None, 1, 1, 224)    0           merge_1[0][0]
	__________________________________________________________________________________________________
	Flatten_5 (Flatten)             (None, 224)          0           MaxPooling2D_70[0][0]
	__________________________________________________________________________________________________
	Dense_14 (Dense)                (None, 5)            1125        Flatten_5[0][0]
	__________________________________________________________________________________________________
	Dense_2 (Dense)                 (None, 5)            30          Dense_14[0][0]
	==================================================================================================
	Total params: 307,091
	Trainable params: 307,091
	Non-trainable params: 0
	__________________________________________________________________________________________________
	2018-12-05 10:25:45,607 INFO None
	2018-12-05 10:25:48,390 INFO len train_generator: 387
	2018-12-05 10:25:48,391 INFO Batch-size: 128

###Graph:

![alt text](https://github.com/minority4u/multiclass_keras/blob/master/reports/conclusion/Logs_1inceptionmodule/model_graph.png "Small-Model")

##L2 regularisation

##Dropout

##Data augmentation

##Batch normalisation

##Bigger patchsize

##Smaller patchsize


Setup custom config file
------------



Setup custom model
------------



