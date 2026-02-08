# STARTING THE PROJECT
# FIRST: load the dataset and then rename the columns properly

import pandas as pd
import numpy as np

# Reading the messy fitness & nutrition dataset
data = pd.read_csv("fitness_nutrition_dataset_messy.csv", encoding="latin1")

# Setting user_id as the index since it uniquely identifies each user
data.set_index("user_id", inplace=True)


# Renaming columns for consistency and clarity
# height, weight, daily calories, protein, carbs, fat, steps, sleep

data.rename(columns={'height_cm': 'HEIGHT'}, inplace=True)
data.rename(columns={"weight_kg": "WEIGHT"}, inplace=True)
data.rename(columns={"protein_g": "PROTEIN"}, inplace=True)
data.rename(columns={"carbs_g": "CARBS"}, inplace=True)
data.rename(columns={"fat_g": "FAT"}, inplace=True)


# Checking a sample of the data
# Searching for missing values


# HANDLING NON-NUMERIC VALUES (like 'unknown') by converting strings to numeric
data["CARBS"] = pd.to_numeric(data["CARBS"], errors="coerce")

# Printing total missing values per column (optional check)
# print(data.isnull().sum())


# HANDLING MISSING VALUES
# Age, height, weight and sleep have missing values

# Filling missing age values with the median age
data["age"] = data["age"].fillna(data["age"].median())

# Filling missing height values with the average height
data["HEIGHT"] = data["HEIGHT"].fillna(data["HEIGHT"].mean())

# Forward filling carbs since they are sequential nutrition values
data["CARBS"] = data["CARBS"].fillna(data["CARBS"].ffill())


# Handling incorrect weight values
# Any weight less than 10 kg is unrealistic, so we convert it to NaN
data.loc[data["WEIGHT"] < 10, "WEIGHT"] = np.nan


# Filling missing weight values based on average weight for the same height
# Using groupby + transform to align values correctly
data["WEIGHT"] = data["WEIGHT"].fillna(
    data.groupby("HEIGHT")["WEIGHT"].transform("mean")
)

# Filling missing sleep hours using median sleep based on age group
data["sleep_hours"] = data["sleep_hours"].fillna(
    data.groupby("age")["sleep_hours"].transform("median")
)


# CHECKING FOR INFINITE VALUES
print(np.isinf(data["HEIGHT"]).sum())
print(np.isinf(data["WEIGHT"]).sum())
print(np.isinf(data["daily_calories"]).sum())
print(np.isinf(data["PROTEIN"]).sum())
print(np.isinf(data["FAT"]).sum())
print(np.isinf(data["steps_per_day"]).sum())
print(np.isinf(data["CARBS"]).sum())


# Daily calories has 5 infinite values and protein has 3 infinite values
# We calculate the median after removing inf values
# Then replace inf and -inf using that median

meanc = data["daily_calories"].replace([np.inf, -np.inf], np.nan).median()
data["daily_calories"] = np.nan_to_num(
    data["daily_calories"], posinf=meanc, neginf=meanc
)

meanp = data["PROTEIN"].replace([np.inf, -np.inf], np.nan).median()
data["PROTEIN"] = np.nan_to_num(
    data["PROTEIN"], posinf=meanp, neginf=meanp
)


# TYPE CASTING
# Converting age to integer after cleaning missing and invalid values
data["age"] = data["age"].astype(int)

print(data.info())


# FINDING WEIRD VALUES AND OUTLIERS

# Checking unrealistic age values
print((data["age"] > 115).sum(), (data["age"] < 0).sum())
weird_age = data.loc[(data["age"] > 110) | (data["age"] < 1)]

# Validating gender categories
print(data["gender"].isin(["Male", "Female", "Other"]).sum())

# Checking height limits
print((data["HEIGHT"] > 230).sum(), (data["HEIGHT"] < 100).sum())

# Checking daily calorie limits
print((data["daily_calories"] > 12000).sum(), (data["daily_calories"] < 500).sum())

# Checking protein limits
print((data["PROTEIN"] > 600).sum(), (data["PROTEIN"] < 10).sum())

# Checking carbs limits
print((data["CARBS"] > 1000).sum(), (data["CARBS"] < 50).sum())

# Checking steps per day limits
print((data["steps_per_day"] > 30000).sum(), (data["steps_per_day"] < 50).sum())


# Checking sleep duration limits
print((data["sleep_hours"] > 14).sum(), (data["sleep_hours"] < 0).sum())

# Sleep values greater than 14 hours are unrealistic
# We replace them with NaN and then fill using the mean
data.loc[data["sleep_hours"] > 14, "sleep_hours"] = np.nan
data["sleep_hours"] = data["sleep_hours"].fillna(data["sleep_hours"].mean())


# Checking fat and weight limits
print((data["FAT"] > 500).sum(), (data["FAT"] < 0).sum())
print((data["WEIGHT"] > 180).sum(), (data["WEIGHT"] < 10).sum())

weird_w = data.loc[(data["WEIGHT"] < 10)]


# Age has some values that are not realistically fixable
# We drop those rows completely
# Weight values below 10 kg were already handled earlier

data.drop(index=[9, 26, 52, 105, 150], inplace=True)

# Removing duplicate rows if any exist
data.drop_duplicates(inplace=True)


# Final renaming of remaining columns for consistency
data.rename(columns={"user_id":"USER ID"},inplace=True)
data.rename(columns={"age": "AGE"}, inplace=True)
data.rename(columns={"gender": "GENDER"}, inplace=True)
data.rename(columns={"daily_calories": "DAILY CALORIES"}, inplace=True)
data.rename(columns={"steps_per_day": "STEPS PER DAY"}, inplace=True)
data.rename(columns={"sleep_hours": "SLEEP HOURS"}, inplace=True)


# FEATURE ENGINEERING
# Creating a new BMI column

data.insert(6, "BMI", np.nan)

# Converting height from cm to meters squared
metres = (data["HEIGHT"] / 100) ** 2

# Calculating BMI using weight and height
data["BMI"] = data["WEIGHT"] / metres


data=data.round(1)

print(data.info())

# Saving the cleaned dataset (optional)
data.to_csv("cleaned_fitness data_file.csv")
