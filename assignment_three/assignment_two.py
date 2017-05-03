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
df.columns = [pp.camel_to_snake(col) for col in df.columns] #convert column names

#fill missing values
df = pp.fill_missing(df, 'serious_dlqin2yrs', filler='mode')
df = pp.fill_missing(df, 'revolving_utilization_of_unsecured_lines', filler='mean')
df = pp.fill_missing(df, 'age', filler='mean')
df = pp.fill_missing(df, 'zipcode', filler='mode')
df = pp.fill_missing(df, 'number_of_time30-59_days_past_due_not_worse', filler='median')
df = pp.fill_missing(df, 'debt_ratio', filler='median')
df = pp.fill_missing(df, 'monthly_income', filler='median')
df = pp.fill_missing(df, 'number_of_open_credit_lines_and_loans', filler='mode')
df = pp.fill_missing(df, 'number_of_times90_days_late', filler='median')
df = pp.fill_missing(df, 'number_real_estate_loans_or_lines', filler='mean')
df = pp.fill_missing(df, 'number_of_time60-89_days_past_due_not_worse', filler='median')
df = pp.fill_missing(df, 'number_of_dependents', filler='mode')

#view some distributions

ed.get_plot(df, plot_type='correlation')

print('Distributions')
print()

ed.group_by(df, 'number_of_dependents')
print()
ed.get_plot(df, 'number_of_dependents')
print()

ed.group_by(df, 'serious_dlqin2yrs')
print()
ed.get_plot(df, 'serious_dlqin2yrs', plot_type='bar')

ed.get_plot(df, 'monthly_income', plot_type='violin')
ed.get_plot(df, 'age', plot_type='hist', bins=12)
ed.get_plot(df, 'number_of_open_credit_lines_and_loans', plot_type='box')

print()
print()


#generate features
df = gf.bin_data(df, 'monthly_income', 10)
df = gf.bin_data(df, 'age', 3)
df = gf.create_dummies(df, 'age_categories')
print()
print()


#logistic regression
target = 'serious_dlqin2yrs'

for column in df.columns:
    if column != 'serious_dlqin2yrs':
        feature = column 
        model = c.regression(df, [feature], target)
        c.evaluate_regression(model, df, [feature], target)

f.close()