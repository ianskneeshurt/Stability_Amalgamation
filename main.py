# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 13:23:44 2021

@author: IPerry
"""

import tkinter as tk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sys import exit
import scipy.stats as stats

'''
6wk accel = 1yr real time

12wk accel = 2yr real time

20wk accel = 3yr real time

26wk accel = 4yr real time
'''


class stabilityapp():
    def __init__(self, master):

        self.star_csv = 'starlims_data.csv'

        # Add checkbuttons for each condition
        self.conditions = ['25C_60RH', '30C_65RH', '30C_75RH', '40C_75RH']
        self.StringVars = []
        self.lbl_conditions = tk.Label(master, text='Conditions to be Included').grid(row=0, sticky=tk.W)
        for i, item in enumerate(self.conditions):
            self.StringVars.append(tk.StringVar())
            self.chk_conditions = tk.Checkbutton(master, text=item, variable=self.StringVars[i],
                                                 onvalue=item, offvalue='None', command=self.condition_select)
            self.chk_conditions.deselect()
            self.chk_conditions.grid(column=1, row=0 + i)

        # shelf life and accelerated shelf life label and entry
        self.lbl_shelflife = tk.Label(master, text='Formula Shelf Life (weeks)')
        self.lbl_shelflife.grid(row=5, sticky=tk.W)
        self.ent_shelflife = tk.Entry(master, width=20)
        self.ent_shelflife.grid(column=1, row=5)
        self.lbl_accshelflife = tk.Label(master, text='Formula Accelerated Shelf Life (weeks)')
        self.lbl_accshelflife.grid(row=6)
        self.ent_accshelflife = tk.Entry(master, width=20)
        self.ent_accshelflife.grid(column=1, row=6)

        # csv entry label, entry, button
        self.label = tk.Label(master, text="CSV Entry (include .csv)").grid(sticky=tk.W, row=8)
        self.ent_csv = tk.Entry(master, width=50)
        self.ent_csv.insert(0, 'All Labware Data.csv')
        self.ent_csv.grid(column=1, row=8, columnspan=2)
        self.btn_csv = tk.Button(root, text='Import Labware csv', command=self.labware_import)
        self.btn_csv.grid(column=4, row=8)
        self.btn_csv = tk.Button(root, text='Import Starlims ONLY', command=self.starlims_import)
        self.btn_csv.grid(column=5, row=8)

        # analyze button
        self.btn_finish = tk.Button(master, text='Analyze', command=self.analyze).grid(column=1, row=100)

    def starlims_import(self):
        '''import starlims data and filter by product entered'''
        self.lbl_starproducts = tk.Label(root, text='StarLims Formula to include in graphs')
        self.lbl_starproducts.grid(column=0, row=9, sticky=tk.W)
        self.ent_starproduct = tk.Entry(root, width=50)
        self.ent_starproduct.grid(column=1, row=9, columnspan=2)
        self.btn_starproduct = tk.Button(root, text='Select this Formula', command=self.formula_filter)
        self.btn_starproduct.grid(column=4, row=9)

    def star_formula_filter(self):
        formula = [int(self.ent_starproduct.get())]

        star_df = pd.read_csv(self.star_csv)
        # convert numeric columns to float
        star_columns = ['PRODUCT', 'TIMEPOINT', 'FORMATTED_ENTRY', 'SAMPLE_NUMBER']
        for col in star_columns:
            star_df[col] = pd.to_numeric(star_df[col], errors='coerce')
        # filter by formula number
        star_df = star_df[star_df['PRODUCT'].isin(formula)]
        # concatenate dataframes
        df = star_df
        # Capitalize every word in the test name column
        df['REPORTED_NAME'] = df['REPORTED_NAME'].str.title()
        df = df.replace(to_replace=['Glucosamine Hcl', 'Glucosamine Hcl - Opa'],
                        value=['Glucosamine HCl', 'Glucosamine HCl'])
        # Change names of dissolution tests
        df.loc[(df.REPORTED_NAME == 'Dissolution') & (df.NAME == 'Result 1'), ['REPORTED_NAME',
                                                                               'NAME']] = 'Glucosamine HCl Dissolution', 'Mean'
        df.loc[(df.REPORTED_NAME == 'Dissolution') & (df.NAME == 'Result 2'), ['REPORTED_NAME',
                                                                               'NAME']] = 'Chondroitin Sulfate Dissolution', 'Mean'
        df.loc[(df.REPORTED_NAME == 'Glucosamine Hcl (60M 75R W)'), 'REPORTED_NAME'] = 'Glucosamine HCl Dissolution'
        df.loc[(
                    df.REPORTED_NAME == 'Chondroitin Sulfate (60M 75R W)'), 'REPORTED_NAME'] = 'Chondroitin Sulfate Dissolution'
        # Change units of dissolution tests
        df = df.replace(to_replace='PCT', value='%')
        # Filter by results with "Mean" in the name
        df = df[df['NAME'].str.contains('Mean')]
        # remove zero value results
        df.drop(df[df['FORMATTED_ENTRY'] == 0].index, inplace=True)
        # include initial timepoints in all conditions 
        every_condition = pd.DataFrame()
        ambient = df[df['CONDITION'] == 'AMBIENT']
        for item in self.selected_conditions:
            for row in ambient:
                ambient.loc[:, 'CONDITION'] = item
            every_condition = every_condition.append(ambient)
        df = df.append(every_condition)
        # Set timepoint to float
        df['TIMEPOINT'] = pd.to_numeric(df['TIMEPOINT'])
        '''#Remove duplicate starlims and labware rows
        find_duplicate_df = df[['TIMEPOINT','CONDITION','FORMATTED_ENTRY', 'REPORTED_NAME']]
        find_duplicate_df['duplicate'] = find_duplicate_df.duplicated()
        duplicate_index = find_duplicate_df[find_duplicate_df['duplicate'] == True].index.tolist()
        df.drop(duplicate_index, inplace = True, axis = 0)'''

        return df

    def labware_import(self):
        '''Imports dataframe from filename entry, adds checkbuttons based on formulas available in df '''
        try:
            self.csv = self.ent_csv.get()
            self.df = pd.read_csv(self.csv)
        except FileNotFoundError:
            tk.messagebox.showerror('Error', 'Filename not found')
            return

        self.lbl_products = tk.Label(root, text='LabWare Formulas to include')
        self.lbl_products.grid(column=0, row=9, sticky=tk.W)
        # create series of unique products
        self.unique_products = self.df['PRODUCT'].unique()
        # checkbuttons for formula selection
        self.IntVars = []
        for i, f in enumerate(self.unique_products):
            self.IntVars.append(tk.IntVar())
            self.check = tk.Checkbutton(root, text=str(f), variable=self.IntVars[i],
                                        onvalue=f, offvalue=0, command=self.product_list)
            self.check.deselect()
            self.check.grid(column=1 + i, row=9, sticky=tk.W)
        self.none_intvar = tk.IntVar()
        self.none_check = tk.Checkbutton(root, text='None', variable=self.none_intvar,
                                         onvalue=1, offvalue=0, command=self.product_list)
        self.none_check.grid(column=1, row=10)
        # starlims formula filter
        self.lbl_starproducts = tk.Label(root, text='StarLims Formula to include in graphs')
        self.lbl_starproducts.grid(column=0, row=11, sticky=tk.W)

        self.ent_starproduct = tk.Entry(root, width=50)
        self.ent_starproduct.grid(column=1, row=11, columnspan=2)

        self.btn_starproduct = tk.Button(root, text='Select this Formula', command=self.formula_filter)
        self.btn_starproduct.grid(column=4, row=11)

    def formula_filter(self):
        '''LABWARE FILTER filters dataframe and adds analyte checkboxes'''
        if len(self.products) > 0:

            lab_df = self.df
            lab_df = lab_df[lab_df['PRODUCT'].isin(self.products)]
            # remove strings from results
            results = lab_df['FORMATTED_ENTRY']
            results = pd.to_numeric(results, errors='coerce')
            lab_df['FORMATTED_ENTRY'] = results.values
            # lab_df = lab_df.dropna()
            lab_df['TIMEPOINT'] = lab_df['TIMEPOINT'].replace(to_replace='TP0', value=0)
            # remove letters from timepoint column
            for item in lab_df['TIMEPOINT']:
                if item != 0 and item[0] == 'W':
                    lab_df = lab_df.replace(to_replace=item, value=item[1:])
                elif item != 0 and item[0] == 'M':
                    months = float(item[1:])
                    weeks = months / 0.230137
                    lab_df = lab_df.replace(to_replace=item, value=weeks)
                elif item != 0 and item[0] == 'D':
                    days = float(item[1:])
                    weeks = days / 7
                    lab_df = lab_df.replace(to_replace=item, value=weeks)
                elif item:
                    print('error: labware timepoint:', item, 'contains unexpected format')

            # include initial timepoints in all conditions 
            every_condition = pd.DataFrame()
            ambient = lab_df[lab_df['CONDITION'] == 'AMBIENT']
            for item in self.selected_conditions:
                for row in ambient:
                    ambient.loc[:, 'CONDITION'] = item
                every_condition = every_condition.append(ambient)
            lab_df = lab_df.append(every_condition)
            # Set timepoint to float
            lab_df['TIMEPOINT'] = pd.to_numeric(lab_df['TIMEPOINT'])

        elif len(self.products) == 0:
            lab_df = pd.DataFrame

        star_formula = self.ent_starproduct.get()
        # combine starlims and labware if both have formulas selected
        if star_formula != '' and self.none_intvar.get() == 0:
            star_df = pd.read_csv(self.star_csv)
            # convert numeric columns to float
            star_columns = ['PRODUCT', 'TIMEPOINT', 'FORMATTED_ENTRY', 'SAMPLE_NUMBER']
            for col in star_columns:
                star_df[col] = pd.to_numeric(star_df[col], errors='coerce')

            star_df['PROJECT'] = star_df['PROJECT'].astype('string')
            # filter by formula number
            star_df = star_df[star_df['PRODUCT'] == int(star_formula)]
            # concatenate dataframes
            df = pd.concat([star_df, lab_df])
        # use only starlims if none checkbox is checked for labware
        elif star_formula != '' and self.none_intvar.get() == 1:
            star_df = pd.read_csv(self.star_csv)
            # convert numeric columns to float
            star_columns = ['PRODUCT', 'TIMEPOINT', 'FORMATTED_ENTRY', 'SAMPLE_NUMBER']
            for col in star_columns:
                star_df[col] = pd.to_numeric(star_df[col], errors='coerce')

            star_df['PROJECT'] = star_df['PROJECT'].astype('string')
            # filter by formula number
            star_df = star_df[star_df['PRODUCT'] == int(star_formula)]
            df = star_df
        # use just labware if starlims entry is empty
        else:
            df = lab_df
        # Remove duplicate starlims and labware rows
        '''find_duplicate_df = df[['TIMEPOINT','CONDITION','FORMATTED_ENTRY', 'REPORTED_NAME']]
        find_duplicate_df['duplicate'] = find_duplicate_df.duplicated()
        duplicate_index = find_duplicate_df[find_duplicate_df['duplicate'] == True].index.tolist()
        df.drop(duplicate_index, inplace = True, axis = 0)'''

        # Capitalize every word in the test name column
        # df['REPORTED_NAME'] = df['REPORTED_NAME'].str.title()
        df = df.replace(to_replace=['Glucosamine Hcl', 'Glucosamine Hcl - Opa'],
                        value=['Glucosamine HCl', 'Glucosamine HCl'])
        # Change names of dissolution tests
        df.loc[(df.REPORTED_NAME == 'Dissolution') & (df.NAME == 'Result 1'), ['REPORTED_NAME',
                                                                               'NAME']] = 'Glucosamine HCl Dissolution', 'Mean'
        df.loc[(df.REPORTED_NAME == 'Dissolution') & (df.NAME == 'Result 2'), ['REPORTED_NAME',
                                                                               'NAME']] = 'Chondroitin Sulfate Dissolution', 'Mean'
        df.loc[(df.REPORTED_NAME == 'Glucosamine Hcl (60M 75R W)'), 'REPORTED_NAME'] = 'Glucosamine HCl Dissolution'
        df.loc[(
                    df.REPORTED_NAME == 'Chondroitin Sulfate (60M 75R W)'), 'REPORTED_NAME'] = 'Chondroitin Sulfate Dissolution'
        # Change units of dissolution tests
        df = df.replace(to_replace='PCT', value='%')
        # Filter by results with "Mean" in the name
        df = df[df['NAME'].str.contains('Mean')]

        # labels for analyte selection
        self.lbl_choose_analyte = tk.Label(root, text='Select Analytes to be included')
        self.lbl_choose_analyte.grid(row=12, sticky=tk.W)

        self.lbl_analytes = tk.Label(root, text='Analyte')
        self.lbl_analytes.grid(column=1, row=12, sticky=tk.W)

        self.lbl_label_claim = tk.Label(root, text='Label Claim')
        self.lbl_label_claim.grid(column=2, row=12, sticky=tk.W)

        self.lbl_target = tk.Label(root, text='Target')
        self.lbl_target.grid(column=3, row=12, sticky=tk.W)

        self.lbl_units = tk.Label(root, text='Units')
        self.lbl_units.grid(column=4, row=12, sticky=tk.W)

        # self.btn_change_name = tk.Button(root, text = 'Change Analyte Name', command = self.change_analyte_name)
        # self.btn_change_name.grid(row = 12, column = 3)
        # create analyte options in GUI
        self.mean_df = df[df['NAME'].str.contains('Mean')]
        self.analyte_options = self.mean_df['REPORTED_NAME'].unique()
        self.label_claim_IntVars = []
        self.target_IntVars = []
        self.unit_StringVars = []
        self.StringVars = []
        for i, f in enumerate(self.analyte_options):
            self.StringVars.append(tk.StringVar())
            self.check = tk.Checkbutton(root, text=str(f), variable=self.StringVars[i],
                                        onvalue=f, offvalue='None', command=self.analyte_list)
            self.check.deselect()
            self.check.grid(column=1, row=13 + i, sticky=tk.W)
            # label claim entries
            self.label_claim_IntVars.append(tk.StringVar())
            self.ent_label_claim = tk.Entry(root, width=20, textvariable=self.label_claim_IntVars[i])
            self.ent_label_claim.grid(column=2, row=13 + i)
            # target entries
            self.target_IntVars.append(tk.StringVar())
            self.ent_target = tk.Entry(root, width=20, textvariable=self.target_IntVars[i])
            self.ent_target.grid(column=3, row=13 + i)
            # unit entries
            self.unit_StringVars.append(tk.StringVar())
            self.ent_unit = tk.Entry(root, width=20, textvariable=self.unit_StringVars[i])
            self.ent_unit.grid(column=4, row=13 + i)

        # remove zero value results
        df.drop(df[df['FORMATTED_ENTRY'] == 0].index, inplace=True)
        self.new_df = df
        self.starlims_product = star_formula
        self.projects = df['PROJECT']

        return self.new_df, self.starlims_product

    '''def change_analyte_name(self):
        newWindow = tk.Toplevel(root)
        newWindow.title('Analyte Name Change')
        newWindow.geometry('400x300')

        self.lbl_analyte_label = tk.Label(newWindow, text = 'Analyte to Change')
        self.lbl_analyte_label.grid(row = 0, column = 0)

        self.lbl_change_label = tk.Label(newWindow, text = 'New Name')
        self.lbl_change_label.grid(row = 0, column = 2)

        self.btn_finish_change = tk.Button(newWindow, text = 'Make Changes', command = self.close_change_window)
        self.btn_finish_change.grid(column = 1, row = 5)

        self.change_analyte_StringVars = []

        for i, f in enumerate(self.analytes):
            self.change_analyte_StringVars.append(tk.StringVar())
            lbl_analyte_display = tk.Label(newWindow, text = str(f))
            lbl_analyte_display.grid(row = 1+i)

            ent_new_name = tk.Entry(newWindow, width = 20, textvariable = self.change_analyte_StringVars[i])
            ent_new_name.grid(row = 1+i, column = 2)

    def close_change_window(self):
        for i, f in enumerate(self.change_analyte_StringVars):
            self.analytes[i] = f.get()   
        self.new_df
        print(self.analytes)'''

    def product_list(self):
        '''Adds selected products to a list'''
        self.products = []
        for value in self.IntVars:
            if value.get() != 0:
                self.products.append(value.get())
        return self.products

    def analyte_list(self):
        '''Adds selected analytes to a list'''
        self.analytes = []
        for value in self.StringVars:
            if value.get() != 'None':
                self.analytes.append(value.get())

    def analyze(self):
        '''finish button return data to main code'''
        self.shelflife = self.ent_shelflife.get()
        self.accshelflife = self.ent_accshelflife.get()

        self.label_claim = []
        for value in self.label_claim_IntVars:
            if value.get() != '':
                self.label_claim.append(float(value.get()))

        self.targets = []
        for value in self.target_IntVars:
            if value.get() != '':
                self.targets.append(float(value.get()))

        self.units = []
        for value in self.unit_StringVars:
            if value.get() != '':
                self.units.append(value.get())

        root.destroy()
        return self.shelflife, self.accshelflife, self.analytes, self.label_claim, self.targets, self.units

    def condition_select(self):
        '''Adds selected conditions to a list '''
        self.selected_conditions = []
        for value in self.StringVars:
            if value.get() != 'None':
                self.selected_conditions.append(value.get())
        return self.selected_conditions


root = tk.Tk()
root.title('Stability Grapher')
root.geometry('900x700')

app = stabilityapp(root)

root.mainloop()

shelf_life = app.shelflife
accelerated_shelf_life = app.accshelflife
analyte = app.analytes
units = app.units
label_claim = app.label_claim
target = app.targets
select_conditions = app.selected_conditions
df = app.new_df


# filter by study
# df = df[df['PROJECT']=='S20-0086']


def datatables(cond):
    formula = app.products
    if len(formula) == 0:
        formula = app.starlims_product
        formula = int(formula)
    '''return data tables for stability reports'''
    lab_df = pd.read_csv(app.csv)
    star_df = pd.read_csv('starlims_data.csv')
    table_df = pd.concat([lab_df, star_df])
    table_df['PRODUCT'] = pd.to_numeric(table_df['PRODUCT'], errors='coerce')

    if type(formula) == list:
        sorted_df = table_df[table_df['PRODUCT'].isin(formula)]
    elif type(formula) == int:
        sorted_df = table_df[table_df['PRODUCT'] == formula]
    print(sorted_df.shape)
    # study_list = sorted_df['PROJECT'].unique()
    # for study in study_list:
    #   study_df = sorted_df[sorted_df['PROJECT']==study]

    for item in cond:
        cond_df = sorted_df[['PROJECT', 'TIMEPOINT', 'REPORTED_NAME', 'FORMATTED_ENTRY', 'UNITS', 'CONDITION']].copy()
        cond_df = cond_df[cond_df['CONDITION'].isin([item])]
        print(cond_df.shape)
        cond_df = cond_df.sort_values(['REPORTED_NAME', 'TIMEPOINT'])
        pd.set_option('display.max_rows', None, 'display.max_columns', None)
        cond_df.to_csv(str(formula) + '_' + '{}.csv'.format(item), index=False, header=True)


def plot_cinterval(cond):
    formula = app.products
    '''plot regression and confidence interval for each test and each condidion'''

    for test in analyte:
        index = analyte.index(test)
        analyte_df = df[df['REPORTED_NAME'] == test]
        if analyte_df.empty:
            print('analyte_df is empty!!! when test =', test)
            exit()
        for item in cond:
            plot_df = analyte_df[analyte_df['CONDITION'].str.contains(item)]
            if plot_df.empty:
                print('plot_df is empty!!! when condition =', item)
                exit()

            # create x and y arrays
            xvals = plot_df['TIMEPOINT'].values
            yvals = plot_df['FORMATTED_ENTRY'].values
            # Convert series to numpy arrays
            x = np.array(xvals)
            y = np.array(yvals)
            slope, intercept = np.polyfit(x, y, 1)  # linear model adjustment

            y_model = np.polyval([slope, intercept], x)  # modeling...

            x_mean = np.mean(x)
            y_mean = np.mean(y)
            n = x.size  # number of samples
            m = 2  # number of parameters
            dof = n - m  # degrees of freedom
            t = stats.t.ppf(0.95, dof)  # Students statistic of interval confidence

            residual = y - y_model

            std_error = (np.sum(residual ** 2) / dof) ** .5  # Standard deviation of the error

            # calculating r2
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = (np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2)) ** .5
            correlation_coef = numerator / denominator
            r2 = correlation_coef ** 2

            # mean squared error
            MSE = 1 / n * np.sum((y - y_model) ** 2)

            # to plot the adjusted model
            x_line = np.linspace(np.min(x), np.max(x), 100)
            y_line = np.polyval([slope, intercept], x_line)

            # confidence interval
            ci = t * std_error * (1 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2)) ** .5
            # predicting interval
            pi = t * std_error * (1 + 1 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2)) ** .5

            ############### Ploting
            plt.rcParams.update({'font.size': 14})
            fig = plt.figure()
            ax = fig.add_axes([.1, .1, .8, .8])

            # ax.plot(x, y, 'o', color = 'royalblue')
            ax.plot(x_line, y_line, color='royalblue')
            ax.fill_between(x_line, y_line + pi, y_line - pi, color='gainsboro', label='95% prediction interval')
            ax.fill_between(x_line, y_line + ci, y_line - ci, color='lavender', label='95% confidence interval')

            # rounding and position must be changed for each case and preference
            a = str(np.round(intercept, 4))
            b = str(np.round(slope, 4))
            r2s = str(np.round(r2, 4))
            MSEs = str(np.round(MSE))

            # create string for equation of the line
            equation = 'Equation for fitted line: ' + str(test) + ' (' + units[
                index] + ')' + ' = [' + b + '] Weeks' + ' + ' + a
            # add equation below plot
            plt.figtext(-0.01, -0.1, equation, horizontalalignment='left')
            plt.figtext(-0.01, -0.17, '$r^2$ = ' + r2s + '     mean squared error = ' + MSEs)

            sns.scatterplot(data=plot_df, x='TIMEPOINT', y='FORMATTED_ENTRY', hue='PROJECT', s=50)

            plt.xlabel('Timepoint (weeks)', fontsize=15)
            # add units to axis if they are not included in the test name
            if '()' in test == True:
                plt.ylabel(test, fontsize=15)
            else:
                plt.ylabel(test + ' ' + '(' + units[index] + ')', fontsize=15)
            # plot title
            if len(formula) == 0:
                formula = app.starlims_product
            title = str(formula) + ' ' + test + ' ' + item
            plt.title(title, x=0.55)
            min_x = 0
            max_x = plot_df['TIMEPOINT'].max()
            # use the lesser of either the label claim or result as the min y value
            if label_claim[index] * 0.99 < plot_df['FORMATTED_ENTRY'].min():
                min_y = label_claim[index] * 0.99
            else:
                min_y = plot_df['FORMATTED_ENTRY'].min()
            max_y = plot_df['FORMATTED_ENTRY'].max()
            y_range = (max_y - min_y) / 5
            plt.axhline(y=label_claim[index], linestyle='--', color='blue',
                        label='Label Claim = ' + str(label_claim[index]) + str(units[index]))
            if len(target) != 0:
                plt.axhline(y=target[index], linestyle='--', color='red',
                            label='Target = ' + str(target[index]) + str(units[index]))
            else:
                pass
            if item == '40C_75RH':
                shelf_life = int(app.accshelflife)
            else:
                shelf_life = int(app.shelflife)
            plt.axvline(x=shelf_life, linestyle='--', color='green',
                        label='Shelf Life = ' + str(shelf_life) + ' weeks')
            plt.xlim(min_x * 1.1, max_x * 1.1)
            plt.ylim(min_y - y_range, max_y + y_range)
            plt.legend(loc='upper center', bbox_to_anchor=[1.4, 1])
            # fit linear model
            # model = LinearRegression().fit(x, y)
            # create string for equation of the line
            # equation = 'Equation for fitted line: '+str(test)+' (' + units[index] + ')' +' = '+str(np.around(model.coef_,4))+' Weeks'+' + '+str(round(model.intercept_, 4))
            # add equation below plot
            # plt.figtext(-0.01, -0.08, equation , horizontalalignment = 'left')
            plt.savefig(str(formula) + test + item, bbox_inches='tight', pad_inches=0.3, dpi=150)


# call the function(s)
plot_cinterval(select_conditions)
datatables(select_conditions)

