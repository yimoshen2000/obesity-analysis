"""
This is the functions portion of the module for the
final project where it aims to better understand the impact
of different variables on obesity.
"""
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score


def import_and_clean(dataset: str):
    """
    This function imports and cleans the dataset
    "obesity.csv", which contains information about
    obesity rates and its related variables. It replaces
    the categorical variable "NObeyesdad" with numerical
    values 0-6, from insufficient weight to obesity type III.
    Parameters
    ==========
    dataset: string
        name of the csv dataset file
    Returns
    =======
    "cleaned" dataset with missing values removed and "NObeyesdad"
    column original values replaced with numerical values for easier
    analysis.
    """
    obesity = pd.read_csv(dataset).dropna()
    # replace values in the "gender" column where male=0 and female=1
    obesity["Gender"].replace(["Male","Female"],[0,1],inplace=True)
    # replace values in the "family_history_with_overweight" column
    # where no=0 and yes=1.
    obesity["family_history_with_overweight"].replace(["no","yes"],[0,1],inplace=True)
    # replace values in the "SCC" column where no corresponds to 0
    # and yes corresponds to 1.
    obesity["FAVC"].replace(["no","yes"],[0,1],inplace=True)
    # replace values in the "SCC" column where no corresponds to 0
    # and yes corresponds to 1.
    obesity["SCC"].replace(["no","yes"],[0,1],inplace=True)
    # replace values in the "SMOKE" column where no corresponds to 0
    # and yes corresponds to 1.
    obesity["SMOKE"].replace(["no","yes"],[0,1],inplace=True)
    # replace values in the "NObeyesdad" column where categorical variables are
    # replaced with numericals from 0-6 for easier analysis.
    obesity["NObeyesdad"].replace(["Insufficient_Weight", "Normal_Weight",
                    "Overweight_Level_I", "Overweight_Level_II",
                    "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"],
                    [0, 1, 2, 3, 4, 5, 6], inplace=True)
    # replace values in the "CAEC" column where categorical variables are
    # replaced with numericals from 0-3 for easier analysis.
    obesity["CAEC"].replace(["no", "Sometimes", "Always", "Frequently"],
                            [0,1,2,3], inplace=True)
    # replace values in the "NObeyesdad" column where categorical variables are
    # replaced with numericals from 0-3 for easier analysis.
    obesity["CALC"].replace(["no", "Sometimes", "Always", "Frequently"],
                            [0,1,2,3], inplace=True)
    return obesity


def data_visualization(dataset: str):
    """
    This function generates graphs and other visuals to show
    the relationships between each variable and obesity
    rates. This will give us a better undertstanding of
    what factors can make someone more susceptible to obesity.
    Parameters
    ==========
    dataset: string
        name of the csv dataset file
    """
    obesity = import_and_clean(dataset)
    # lmplot that shows relationship between alcohol consumption
    # and obesity level.
    sns.lmplot(x = "CALC", y = "NObeyesdad", data = obesity)
    # lmplot that shows relationship between frequency of physical
    # activities and obesity level.
    sns.lmplot(x = "FAF", y = "NObeyesdad", data = obesity)
    # barplot that shows the difference in obesity level between
    # males and females.
    sns.barplot(x = "Gender", y = "NObeyesdad", data = obesity)
    # barplot that shows the difference in obesity level between
    # people who have overweight family history and people who don't
    sns.barplot(x = "family_history_with_overweight", y = "NObeyesdad",
                data = obesity)


def descriptive_statistics(dataset: str):
    """
    This function provides descriptive statistics like median,
    mean and standard deviations etc for us to better understand
    the dataset. Specifically, comparison of obesity level between
    male vs female, family history vs no family history, smoker vs
    nonsmoker.
    Parameters
    ==========
    dataset: string
        name of the csv dataset file
    """
    obesity = import_and_clean(dataset)
    # Overview of dataset
    relevant_vars = obesity[["Age","Height","Weight","NCP","FAF","CALC","NObeyesdad"]]
    return relevant_vars.describe()


def data_model(dataset: str):
    """
    This function generates a model for obesity data and estimates
    such model by producing an OLS regression table.
    Parameters
    ==========
    dataset: string
        name of the csv dataset file
    """
    obesity = import_and_clean(dataset)
    lhs = obesity["NObeyesdad"]
    ind_vars = ["Gender","Age","Weight","Height","family_history_with_overweight",
                "FAVC","FCVC","NCP","CAEC","SMOKE","CH2O","FAF","TUE","CALC"]
    rhs = obesity.loc[:, ind_vars]
    rhs = sm.add_constant(rhs)
    mod = sm.OLS(lhs, rhs)
    res = mod.fit()
    print(res.summary())
    yhat = res.predict(rhs)
    prediction = list(map(round, yhat))
    accuracy = accuracy_score(lhs, prediction)
    print('Test accuracy = ', accuracy*100, '%')
    