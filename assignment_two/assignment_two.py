import read_data as rd
import explore_data as ed
import pre_process as pp
import generate_features as gf
import classify as c

import sys
f = open("assignment_two_output.txt", 'w')
sys.stdout = f


#read data
df = rd.read_data('csv', '/Users/emmapeterson/machine-learning-for-public-policy/assignment_two/credit-data.csv')


#explore data
print('Overview')
print()
ed.overview(df)
print()

print('Nulls')
print()
ed.view_nulls(df)
print()

print('Describe Data')
print()
ed.describe_data(df)
print()


#preprocess data
df = pp.fill_missing(df) #fill NaN with mean
df.columns = [pp.camel_to_snake(col) for col in df.columns] #convert column names


#view some distributions
print('Distributions')
print()
ed.group_by(df, 'number_of_dependents')
print()
ed.get_plot(df, 'number_of_dependents')
print()

ed.group_by(df, 'serious_dlqin2yrs')
print()
ed.get_plot(df, 'serious_dlqin2yrs', plot_type='pie')

ed.get_plot(df, 'monthly_income', plot_type='box')
ed.get_plot(df, 'age')
ed.get_plot(df, 'zipcode', plot_type='pie')
print()
print()


#generate features
df = gf.create_dummies(df, 'serious_dlqin2yrs')
df = gf.bin_data(df, 'monthly_income', 10)
print()
print()


#logistic regression
target = 'serious_dlqin2yrs_1'

for column in df.columns:
    if column != 'serious_dlqin2yrs_1' and column != 'serious_dlqin2yrs_0':
        feature = column 
        model = c.regression(df, [feature], target)
        c.evaluate_regression(model, df, [feature], target)

f.close()