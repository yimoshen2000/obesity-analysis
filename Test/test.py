"""
This is the testing portion of the module draft for the
final project where it tests the 4 functions in the
functions.py file.
"""
import os.path
import pandas as pd
import FinalProject.functions as f


def test_import_and_clean():
    """
    This function tests the returned output generated
    by the import_and_clean function in functions.py
    """
    dataframe = pd.read_csv("obesity.csv").dropna()
    dataframe_generated = f.import_and_clean("obesity.csv")
    # check number of columns for the dataframe generated here
    assert len(dataframe_generated.columns) ==  17
    # check null values
    assert dataframe_generated.isNull().sum() == 0
    # check columns of the cleaned datadframe in functions.py
    # compared to data frame generated here
    assert len(dataframe_generated.columns) == len(dataframe.columns)
    print("test_import_and_clean passed")


def test_data_visualization():
    """
    This function is a unit test that tests the returned
    output by the data_visualization function in functions.py
    """
    # test for figure 1
    path_1 = 'Documents/ECON406/FinalProject/FinalProject/Figure1.png'
    if os.path.isfile(path_1) and os.access(path_1, os.R_OK):
        print("File exists and is readable. test_data_visualization passed.")
    else:
        print("Either the file is missing or not readable")
    # test for figure 2
    path_2 = 'Documents/ECON406/FinalProject/FinalProject/Figure2.png'
    if os.path.isfile(path_2) and os.access(path_2, os.R_OK):
        print("File exists and is readable. test_data_visualization passed.")
    else:
        print("Either the file is missing or not readable")
    # test for figure 3
    path_3 = 'Documents/ECON406/FinalProject/FinalProject/Figure3.png'
    if os.path.isfile(path_3) and os.access(path_3, os.R_OK):
        print("File exists and is readable. test_data_visualization passed.")
    else:
        print("Either the file is missing or not readable")
    # test for figure 4
    path_4 = 'Documents/ECON406/FinalProject/FinalProject/Figure4.png'
    if os.path.isfile(path_4) and os.access(path_4, os.R_OK):
        print("File exists and is readable. test_data_visualization passed.")
    else:
        print("Either the file is missing or not readable")


def test_descriptive_statistics():
    """
    This function is a unit test that tests the returned
    output by the descriptive_statistics function in functions.py
    """
    des_stat = f.descriptive_statistics("obesity.csv")
    cols = ["Age","Height","Weight","NCP","FAF","CALC","NObeyesdad"]
    assert len(des_stat.columns) == len(cols)
    assert isinstance(des_stat, list) is True
    print("test_ddescriptive_statistics passed.")


def test_data_model():
    """
    This function is a unit test that tests the returned
    output by the data_model function in functions.py
    """
    cleaned_data = f.import_and_clean("obesity.csv")
    ols_model = f.data_model(cleaned_data)
    ols_fit = ols_model.fit()
    assert ols_fit.params["const"] != 0
    assert len(ols_fit.params) == 15
    assert ols_fit.cov_type == "nonrobust"
    assert len(ols_fit.predict()) == len(cleaned_data["NObeyesdad"])
    print("test_data_model passed.")


def main_test():
    """
    This function runs all the tests together at once to
    determine if all the tests are passed.
    """
    test_import_and_clean()
    test_data_visualization()
    test_descriptive_statistics()
    test_data_model()
    print("All unit tests are passed!")
