# Store Sales - TS Forecast

<h4>Emmm... wouldn't it be if we could use machine learning to predict grocery sales.<br> So Yeah! you guess it! this notebook is going to be about the Store Sales TS Forecast Compeition 😄😄<br><br>I will be doing a EDA of review texts, some Visualization and Pre-Processing. and finally modelling <br></h4>

**<h2>To get started:</h2>**

Creating and cd to new folder for Dataset
```
cd /D ../input & mkdir store-sales-time-series-forecasting & cd store-sales-time-series-forecasting
```

Downloading Dataset from Kaggle
```
kaggle competitions download -c store-sales-time-series-forecasting
```

Unzipping and deleting .zip file
<h5>For Linux:</h5>

```
unzip store-sales-time-series-forecasting.zip && trash store-sales-time-series-forecasting.zip
```
<h5>For Windows:</h5>

```
PowerShell Expand-Archive -Path "store-sales-time-series-forecasting.zip" -DestinationPath ./ & del /f store-sales-time-series-forecasting.zip
```

Lunching the notebook
```
jupyter notebook "..\..\Store Sales\store-sales-ts-forecast-a-beginner-s-notebook.ipynb"
```
