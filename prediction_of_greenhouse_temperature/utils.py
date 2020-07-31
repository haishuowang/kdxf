import pandas as pd
from loc_lib.tools_analyse import scatter_matrix_plot, kdeplot
from loc_lib.tools_train import regressor_dict, train_reg_model


root_path = 'J:\data\prediction_of_greenhouse_temperature\data'
train_df = pd.read_csv(fr'{root_path}\train\train.csv')
test_df = pd.read_csv(fr'{root_path}\test\test.csv')
label = 'temperature'
use_feat = ['温度(室外)', '湿度(室外)', '气压(室外)', '湿度(室内)', '气压(室内)']
# scatter_matrix_plot(train_df[use_feat])
# scatter_matrix_plot(train_df[['温度(室外)', '湿度(室外)', '气压(室外)', '湿度(室内)', '气压(室内)', 'temperature']])
kdeplot(train_df[use_feat].ffill(), test_df[use_feat].ffill())

model = regressor_dict['LinearRegression']

train_reg_model(model, {}, X, y, splits=5, repeats=5)