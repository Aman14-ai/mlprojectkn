from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            logging.info("starting the process of getting data transformer object")
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("starting the data transformation process")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            preprocessor_obj=self.get_data_transformer_object()
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            train_df_input_features=train_df.drop(target_column_name,axis=1)
            train_df_target_feature=train_df[target_column_name]    

            test_df_input_features=test_df.drop(target_column_name,axis=1)
            test_df_target_feature=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            preprocessed_train_input_features=preprocessor_obj.fit_transform(train_df_input_features)
            preprocessed_test_input_features=preprocessor_obj.transform(test_df_input_features)

            train_arr=np.c_[preprocessed_train_input_features,np.array(train_df_target_feature)]
            test_arr=np.c_[preprocessed_test_input_features,np.array(test_df_target_feature)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            pass
        except Exception as e:
            raise CustomException(e,sys)
