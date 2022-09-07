## Repository content

This repository contains some tasks. First algorithm should calculate the sum from 1 to N. You can find this code in **sum_of_num.ipynb**.

The second task is to write an algorithm which counts a number of islands on map MxN matrix, where 1-islands, 0-ocean. This code is in **finding_islands.ipynb**.

The next task is prediction some target based on 53 features. Target metric is *RMSE*. The jupyter notebook with data analysis (data distribution, correlation, outliers detection, scaling), building different regression models (DecisionTreeRegressor, Lasso, RandomForestRegressor, ExtraTreesRegressor, LGBMRegressor, XGBRegressor) and comparing model's metrics such as *r2 score* and *RMSE*. Almost all models gave a good result, but for prediction I chose LGBMRegressor, because the different between *Train_RMSE* and *Test_RMSE* is minimal which is mean that model was predicted well. So, the code you can find in **model_training.py**, **model_prediction.py** and **target_prediction.ipynb**. And the train and test sets in **internship_train.csv** and **internship_hidden_test.csv**.

The last task is about soil erosion prediction. I don't finish this task, because I have some questions for data and for a short piece of information about that problem. I wish I could have more pictures for training a model. I found only one and didn't understand from where I should take them more for training.
I haven't got the result but my plan was the next: I deleted some columns and why I did that I wrote in comments.
Then I tried to create binary mask for image and after that trained a U-Net model. U-Net is a segmentation model which uses a strong data augmentation to use the available annotated samples more efficiently. It's architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.
As for the model summary I used *activation = relu*, to preserve dimensions I used *padding = same* and then concatenate layers that are supposed to have the same dimensions. The *setting batch_size = 5* and *steps_per_epoch = 100*, *optimizer = adam*. 
There are a lot of built-in metrics, but default metrics are not always good idea. In this case we need to create a function for measuring quality of model. This metric will be callling *Intersection over Union*. It's arguments would present ground truth mask of a soil erosion and predicted mask. When this mattching is perfect, metric value is 1 and the lower predicting precison is, the lower is this value (down to zero). 
The code you can find in **soil_erosion_detection.ipynb**. The data for that task I couldn't download to this repo, because of size.
