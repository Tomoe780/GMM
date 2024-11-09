import pandas as pd


# 读取数据集
file_path = r"./dataset/original-adult.csv"
data = pd.read_csv(file_path, delimiter=',')
print(data.head())
# 检查是否存在 NaN 值
print(data.isna().sum())

# bank
# data = data[['age', 'balance', 'duration']]
# data = data.dropna()

# adult
data.columns = data.columns.str.strip()
data = data[['age', 'fnlwgt', 'education-num', 'hours-per-week']]
data = data.dropna()

# athlete
# data = data[['Age', 'Height', 'Weight', 'Sex']]
# data.replace([], pd.NA, inplace=True)
# data = data.dropna()
# # 去除所有重复行，只保留第一出现的那一行
# data = data.drop_duplicates()
# data['Sex'] = data['Sex'].map({'F': 0, 'M': 1}).astype(int)
# # 输出预处理后的数据集
# new_file_path = r"./dataset/bank.csv"
# data.to_csv(new_file_path, header=1, index=0)

# diabetes
# data = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Outcome']]
# data.replace([], pd.NA, inplace=True)
# data = data.dropna()

data = data.drop_duplicates()
new_file_path = r"./dataset/adult.csv"
data.to_csv(new_file_path, header=1, index=0)
