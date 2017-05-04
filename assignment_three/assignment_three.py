import read_data as rd
import explore_data as ed
import pre_process as pp
import generate_features as gf
import classify as c
import sys

f = open("assignment_three_output.txt", 'w')
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

fillers = {
            'serious_dlqin2yrs': 'mode', 
            'revolving_utilization_of_unsecured_lines': 'mean',
            'age': 'mean',
            'zipcode': 'mode',
            'number_of_time30-59_days_past_due_not_worse': 'median',
            'debt_ratio': 'median',
            'monthly_income': 'median',
            'number_of_open_credit_lines_and_loans': 'mode',
            'number_of_times90_days_late': 'median',
            'number_real_estate_loans_or_lines': 'mean',
            'number_of_time60-89_days_past_due_not_worse': 'median',
            'number_of_dependents': 'median'
            }
        
for column, fill in fillers.items():
    print(column, fill)
    df = pp.fill_missing(df, column, filler=fill)

#view some distributions

ed.get_plot(df, plot_type='correlation', title='Credit Data')

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
ed.get_plot(df, 'age', plot_type='hist', num_bins=12)
ed.get_plot(df, 'number_of_open_credit_lines_and_loans', plot_type='box')

print()
print()


#generate features
#df = gf.bin_data(df, 'monthly_income', 10)
#df = gf.bin_data(df, 'age', 3)
#df = gf.create_dummies(df, 'age_categories')
print()
print()


#fit models
features = list(df.columns)
y = 'serious_dlqin2yrs'
features.remove(y)

r = c.model_loop(df, features, y, 'small')


#evaluate results
c.evaluate_results(r)



