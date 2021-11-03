import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_table_download_link(df, filename):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="' + filename + f'">Download xlsx file</a>' # decode b'abc' => abc


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', sorted(filenames))

    return os.path.join(folder_path, selected_filename)


def to_lower_case():
    return lambda text: str(text).lower()


@st.cache(allow_output_mutation=True)
def load_data(filename):
    data = pd.read_csv(filename)
    lowercase = to_lower_case()
    data.rename(lowercase, axis=1, inplace=True)

    return data


final_table = pd.DataFrame()
COUNTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

st.write("# Analyse Differences")
st.markdown(
    """
    Read csv data files
    """)

data_file = file_selector(folder_path=r'data')
st.write('You selected `%s`' % data_file)


if data_file:
    # 4. option to load template file
    use_template_file = st.radio('Upload a template file?', ('Yes', 'No'), 1)
    if use_template_file == 'Yes':
        st.set_option('deprecation.showfileUploaderEncoding', False)
        template_file = st.file_uploader('Select directions template a .csv file', type='csv')

    # read data file
    data_df = load_data(data_file)
    # convert non numeric values to nan
    data_df[data_df.columns[2:]] = data_df[data_df.columns[2:]].apply(lambda value: pd.to_numeric(value, errors='coerce'))
    group_col = data_df.columns[0]
    subject_col = data_df.columns[1]
    subject_list = data_df[subject_col]

    # for paired - calculate differences per subject per parameter
    per_subject_diffs = data_df.copy()
    per_subject_diffs = per_subject_diffs.groupby([subject_col])[per_subject_diffs.columns].diff()
    per_subject_diffs[subject_col] = subject_list
    per_subject_diffs = per_subject_diffs[per_subject_diffs[group_col] == 1].set_index(subject_col)
    per_subject_diffs = per_subject_diffs.drop(columns=group_col)

    # read template file
    if use_template_file == 'Yes':
        template_df = load_data(template_file)

    group_list = {'group_' + str(i): {'selected': 1} for i in data_df[group_col]}
    if use_template_file == 'Yes':
        param_list = {i: {'selected': False, 'direction': template_df.loc[0, i]} for i in template_df.columns}
    else:
        param_list = {i: {'selected': False, 'direction': 'both'} for i in data_df.columns[2:]}

    control_group = st.selectbox('Select control group', list(group_list.keys()))
    st.write('## Select groups for analysis')
    for group in group_list:
        if group != control_group:
            group_list[group]['selected'] = st.checkbox(group, group, group + '_key')

    # 1. collect group means and sd into dataframes
    group_mean = pd.DataFrame()
    group_std = pd.DataFrame()
    for group in list(i for i in group_list.keys() if group_list[i]['selected']):  # loop groups and calculate mean and sd + remove unwanted columns
        group_mean[group] = data_df[data_df[group_col] == int(group[-1])][param_list].mean()  # collect all of the groups means
        group_std[group] = data_df[data_df[group_col] == int(group[-1])][param_list].std()  # collect all of the groups SDs

    # 1.1 (Roee) - set mode to paired
    paired_test_mode = st.radio('Use paired tests?', ('Yes', 'No'), 1)

    if paired_test_mode == 'Yes':
        grouped_diffs = pd.DataFrame()
        group_mean = group_mean.rename(columns={group_mean.columns[0]: 'Pre', group_mean.columns[1]: 'Post'})
        delta_std = 1.0
        # group_mean['diff'] = group_mean['group_1'] - group_mean['group_0']
        # group_mean['diff_std=' + str(delta_std)] = group_mean['diff'].std() * delta_std
        # grouped_diffs['diff'] = group_mean['Post'] - group_mean['Pre']
        grouped_diffs['mean'] = per_subject_diffs.mean()
        grouped_diffs['count'] = per_subject_diffs.count()
        std_column_name = 'std_' + str(delta_std)
        grouped_diffs[std_column_name] = per_subject_diffs.std() * delta_std
        'Per parameter group diff', grouped_diffs
        'Per subject per parameter diff', per_subject_diffs

        # delta_std = st.slider('Select % of differences between subjects delta and mean group delta', 0.5, 2.0, 1.0, 0.1)
        stds = list(float(x / 10) for x in range(5, 21))
        std_df = pd.DataFrame(index=range(1, len(grouped_diffs) + 1))
        show_list = [1.0, 1.5, 2.0]
        affected = {}
        affected_full = {}

        for std in stds:
            grouped_diffs[std_column_name] = per_subject_diffs.std() * std
            grouped_diffs = grouped_diffs.rename(columns={std_column_name: 'std_' + str(std)})
            affected[std] = per_subject_diffs.copy()
            affected_full[std] = per_subject_diffs.copy()

            grouped_diffs['+1'] = 0
            grouped_diffs['-1'] = 0
            grouped_diffs['0'] = 0
            for i, row in grouped_diffs.iterrows():
                for j, subject in per_subject_diffs[i].dropna().iteritems():
                    if (row['mean'] - row['std_' + str(std)] > subject) or\
                            (subject > row['mean'] + row['std_' + str(std)]):
                        # affected[std]
                        affected[std].loc[j, i] = 1
                    else:
                        affected[std].loc[j, i] = 0
                    if subject > row['mean'] + row['std_' + str(std)]:
                        affected_full[std].loc[j, i] = 1
                        grouped_diffs.loc[i, '+1'] += 1
                    elif subject < row['mean'] - row['std_' + str(std)]:
                        affected_full[std].loc[j, i] = -1
                        grouped_diffs.loc[i, '-1'] += 1
                    else:
                        affected_full[std].loc[j, i] = 0
                        grouped_diffs.loc[i, '0'] += 1

            affected[std]['affected'] = affected[std].sum(axis=1)
            affected_full[std]['affected'] = affected[std]['affected']
            # grouped_diffs['affected'] = grouped_diffs['+1'] + grouped_diffs['-1']
            for parameters in range(1, len(grouped_diffs) + 1):
                std_df.loc[parameters, '{:.1f}'.format(std)] =\
                    len(affected[std].loc[affected[std]['affected'] >= parameters]) / len(per_subject_diffs) * 100
            if std in show_list:
                std, grouped_diffs
            # std, affected
        std_df

        str_stds = list(str(std) for std in stds)
        # sd_col, param_col = st.beta_columns(2)
        std = float(st.selectbox('select SD to view', str_stds, str_stds.index('1.0')))
        # params = param_col.selectbox('select number of parameters to view', range(1, len(grouped_diffs) + 1), 6)
        filename = 'affected_sd_' + str(std)
        affected_full[std]

        st.markdown(get_table_download_link(affected_full[std], filename), unsafe_allow_html=True)

        # delta_std = st.slider('Select % of differences between subjects delta and mean group delta', 0.5, 2.0, 1.0, 0.1)
        # grouped_diffs[std_column_name] = per_subject_diffs.std() * delta_std
        # grouped_diffs = grouped_diffs.rename(columns={std_column_name: 'std_' + str(delta_std)})
        # affected = per_subject_diffs.copy()
        #
        # grouped_diffs['+1'] = 0
        # grouped_diffs['-1'] = 0
        # grouped_diffs['0'] = 0
        # for i, row in grouped_diffs.iterrows():
        #     # i, row
        #     for j, subject in per_subject_diffs[i].dropna().iteritems():
        #         if subject > row['mean'] + row['std_' + str(delta_std)]:
        #             affected.loc[j, i] = 1
        #             grouped_diffs.loc[i, '+1'] += 1
        #         elif subject < row['mean'] - row['std_' + str(delta_std)]:
        #             affected.loc[j, i] = -1
        #             grouped_diffs.loc[i, '-1'] += 1
        #         else:
        #             affected.loc[j, i] = 0
        #             grouped_diffs.loc[i, '0'] += 1
        # grouped_diffs['affected'] = grouped_diffs['+1'] + grouped_diffs['-1']
        # 'grouped_diffs percentages', grouped_diffs
        # # 'grouped diffs counts', grouped_diffs
        # grouped_diffs['+1'] = grouped_diffs['+1'] / grouped_diffs['count'] * 100
        # grouped_diffs['-1'] = grouped_diffs['-1'] / grouped_diffs['count'] * 100
        # grouped_diffs['0'] = grouped_diffs['0'] / grouped_diffs['count'] * 100
        # grouped_diffs = grouped_diffs.round({'+1': 2, '-1': 2, '0': 2})
        # # calculate percent affected
        # 'grouped_diffs percentages', grouped_diffs
        # 'affected', affected

    else:
        # 2.1. set the difference between control and other groups mean to be included
        mean_difference = st.slider('Select mean differences between control and experiment groups', 0, 100, 30, 10)
        mean_difference = mean_difference / 100
        # 2.2. set the difference between control and other groups deviance to be included
        deviation_difference = st.slider('Select deviance differences between control and experiment groups', 0, 100, 30, 10)
        deviation_difference = deviation_difference / 100

        # 2.3. find parameters that should be included
        for group in list(i for i in group_list.keys() if group_list[i]['selected']):
            for param in param_list:  # calculate deviation difference for all the parameters columns
                if group_std[group][param] == 0:
                    calculated_deviation_difference = 0
                else:
                    calculated_deviation_difference = group_std[group][param] / group_std[control_group][param]
                if (((1 - deviation_difference) > calculated_deviation_difference > (1 + deviation_difference)) or
                        ((abs(group_mean[group][param] - group_mean[control_group][param]) /
                          group_mean[control_group][param]) > mean_difference)):
                    param_list[param]['selected'] = True

        # 3. define the parameters list
        st.write('## Select parameters for analysis')
        directions = ['both', 'up', 'down']
        col1, col2 = st.beta_columns(2)

        for param in param_list:
            col1, col2 = st.beta_columns(2)
            # 3. allow manual changes for parameter inclusion
            param_list[param]['selected'] = col1.checkbox(param, value=param_list[param]['selected'])
            # 4. allow manual changes for direction
            param_list[param]['direction'] = col2.selectbox(param + ' direction', directions,
                                                            index=directions.index(param_list[param]['direction']))
        # 4. save template to file
        if st.button('Save directions to template file?'):
            template_df = pd.DataFrame.from_dict(param_list)
            template_df = template_df.drop('selected')
            save_template_file = data_file[:-4] + '_template.csv'
            template_df.to_csv(save_template_file, index=False, header=True)
            st.write('File', save_template_file, 'saved.')

        included_group_list = list(i for i in group_list.keys() if group_list[i]['selected'])
        included_param_list = list(i for i in param_list.keys() if param_list[i]['selected'])

        group_list_for_dev = {}
        for group in included_group_list:
            if group != control_group:
                group_list_for_dev[group] = {}
                group_list_for_dev[group]['selected'] = st.checkbox(group, group)

        max_of_max = {'SD': 0.5, '# of params': 0, 'max diff': 0, 'weighed': 0}  # holds the optimized and weighed maximum difference between control and group

        # 5. find out the sd that has the largest difference between control and experiment groups
        true_columns = {}
        true_false = {}
        for_pie = {}
        group_len = {}
        for sd in range(5, 21):
            sd = sd / 10
            # create the final count files infrastructure in the size of included groups x number of parameters
            true_columns[sd] = pd.DataFrame()
            true_columns[sd]['count'] = COUNTS[:len(param_list)]
            for group in list(i for i in group_list.keys() if group_list[i]['selected']):
                true_columns[sd][group] = 0
            true_columns[sd] = true_columns[sd].set_index('count')
            true_false[sd] = pd.DataFrame()
            true_false[sd][[group_col, subject_col]] = data_df[[group_col, subject_col]]
            group_len[sd] = {}
            # check the inclusion or exclusion for each animal in each parameter against the control group mean and SD
            for group in included_group_list:  # loop thorough the included groups
                for param in included_param_list:  # calculate for each of the parameters
                    x = (data_df[param] - group_mean[control_group][param]) / group_std[control_group][param]
                    if param_list[param]['direction'] == 'both':
                        true_false[sd][param] = abs(x) >= sd
                    elif param_list[param]['direction'] == 'up':
                        true_false[sd][param] = x >= sd
                    else:
                        true_false[sd][param] = x <= -sd
                group_len[sd][group] = data_df[data_df[group_col] == int(group[-1])].shape[0]

            true_false[sd]['sum'] = true_false[sd][included_param_list].sum(axis=1)

            # calculate the percentage of affected animals per group per level (if 2 is selected there are high_start and low levels)
            for group in included_group_list:  # loop thorough the included groups
                percentages = []
                for count in true_columns[sd].index:
                    percentages.append(true_false[sd][(true_false[sd]['sum'] >= count) &
                                                      (true_false[sd][group_col] == int(group[-1]))].count()['sum'] /
                                       group_len[sd][group] * 100)
                true_columns[sd][group] = percentages

            # test for max difference only if control group value is under 20%
            true_sums_counts_cut = true_columns[sd][true_columns[sd][control_group] <= 20]

            # calculate the maximum difference between any of the groups with the control group
            true_sums_counts_cut.loc[:, 'max_diff'] = \
                true_sums_counts_cut.loc[:, included_group_list].max(1) - true_sums_counts_cut[control_group]

            # calculate the weighed index as max_diff divided by the number of parameters
            true_sums_counts_cut['weighed'] = true_sums_counts_cut['max_diff'] * true_sums_counts_cut.index

            # replace if weighed is grater than the value collected until now
            if true_sums_counts_cut['weighed'].max() > max_of_max['weighed']:
                max_index = true_sums_counts_cut['weighed'].idxmax()
                max_of_max = {'SD': sd,
                              '# of params': int(max_index),
                              'max diff': true_sums_counts_cut.loc[max_index, 'max_diff'],
                              'weighed': true_sums_counts_cut.loc[max_index, 'weighed']}

        'Best Diff = ', max_of_max
        # 6. show the sd slider with calculated sd value as default. allow selecting 2 limits (high, low)
        dev_high = st.slider('Select SD range for highly affected', 0.5, 2.0, max_of_max['SD'], 0.1)
        second_level = st.checkbox('Add 2nd level?')

        final_table[subject_col] = data_df[subject_col]  # final table
        if not second_level:  # only high level is selected
            st.line_chart(true_columns[dev_high])

            # 11 + 12. create pie charts for 1 level
            for group in true_columns[dev_high][included_group_list]:
                fig1, ax1 = plt.subplots()
                labels = 'Affected', 'Unaffected'
                explode = (0.1, 0)  # only "explode" the first slice_high
                color = ['0.7', '0.85']
                slice_high = int(true_columns[dev_high].loc[max_of_max['# of params'], group])
                sizes = (slice_high, 100 - slice_high)
                ax1.set_title('Group ' + group[-1] + ' / ' + 'SD: ' + str(dev_high), fontsize=18)
                ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, colors=color,
                        textprops={'fontsize': 14})
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig1)

                # 13. create the final table of subjects and their respective list of affected parameters
                final_table.loc[true_false[dev_high]['sum'].index, 'Affected'] = \
                    true_false[dev_high]['sum'] >= max_of_max['# of params']
                only_true_false_df = true_false[dev_high].drop([group_col, subject_col, 'sum'], axis=1)
                attended_params = only_true_false_df.apply(lambda row: row.index[row.astype(bool)].tolist(), 1)  #
                final_table.loc[true_false[dev_high]['sum'].index, 'Params'] = attended_params

        else:  # selected to use 2 levels
            dev_low = st.slider('Select SD range for medium affected', 0.5, dev_high, dev_high * 0.7, 0.1)

            for group in included_group_list:  # loop thorough the included groups
                for param in included_param_list:  # calculate for each of the parameters
                    x = (data_df[param] - group_mean[control_group][param]) / group_std[control_group][param]
                    if param_list[param]['direction'] == 'both':
                        true_false[dev_high][param + '_med'] = (abs(x) < dev_high) & (abs(x) >= dev_low)
                    elif param_list[param]['direction'] == 'up':
                        true_false[dev_high][param + '_med'] = (x < dev_high) & (x >= dev_low)
                    else:
                        true_false[dev_high][param + '_med'] = (x > -dev_high) & (x <= -dev_low)

            # 7. count and find percent of own group of affected animals for 2 levels - highly and medium affected
            true_false[dev_high]['sum'] = true_false[dev_high][included_param_list].sum(axis=1)
            true_false[dev_high]['perc_of_included_high'] = true_false[dev_high]['sum'] / len(included_param_list) * 100
            true_false[dev_high]['sum_med'] = true_false[dev_high].loc[:, [x for x in true_false[dev_high].columns if x.endswith('_med')]].sum(axis=1)
            true_false[dev_high]['perc_of_included_med'] = true_false[dev_high]['sum_med'] / len(included_param_list) * 100

            # 8. calculate the correct value of affected for each subject
            true_false[dev_high]['affect_value_corrected'] = true_false[dev_high]['perc_of_included_high'] +\
                                                             (true_false[dev_high]['perc_of_included_med'] / 2)

            # 9. calculate low-med point by the mean and sd of 'affect_value_corrected' for control group
            control_values = true_false[dev_high]['affect_value_corrected'][true_false[dev_high]['group'] == 0]
            medium_default = int(control_values.mean() + control_values.std())

            # 10. calculate med-high point by the mean of remaining value' - (max + min) / 2
            medium_start = st.slider('Select medium cut point', 0, 100, medium_default, 1)
            high_series = true_false[dev_high]['affect_value_corrected'][true_false[dev_high]['affect_value_corrected'] > medium_default]
            high_default = int((high_series.max() + high_series.min()) / 2)
            high_start = st.slider('Select high cut point', medium_start, 100, high_default + 1, 1)

            # 12. create pie charts for 2 levels
            for group in included_group_list:
                for_pie[group] = {}
                highly_affected_count = true_false[dev_high][(true_false[dev_high]['affect_value_corrected'] > high_start) &
                                                      (true_false[dev_high]['group'] == int(group[-1]))]['group'].count() / group_len[dev_high][group] * 100
                medium_affected_count = true_false[dev_high][(true_false[dev_high]['affect_value_corrected'] < high_start) &
                                                      (true_false[dev_high]['affect_value_corrected'] > medium_start) &
                                                      (true_false[dev_high]['group'] == int(group[-1]))]['group'].count() / group_len[dev_high][group] * 100

                for_pie[group]['highly affected'] = highly_affected_count
                for_pie[group]['mildly affected'] = medium_affected_count
                for_pie[group]['not affected'] = 100 - (highly_affected_count + medium_affected_count)

                labels = ['Highly Affected', 'Medium Affected', 'Not Affected']
                explode = (0.1, 0.1, 0)  # only "explode" the first slice_high
                color = ['0.7', '0.6', '0.85']
                sizes = [for_pie[group]['highly affected'], for_pie[group]['mildly affected'], for_pie[group]['not affected']]

                fig1, ax1 = plt.subplots()
                ax1.set_title('Group ' + group[-1] + ' / ' + 'SD (high_start / medium_start): ' + str(dev_high) +
                              ' / ' + str(dev_low), fontsize=18)

                ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, colors=color,
                        textprops={'fontsize': 14})
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot()

                # 13. create the final table of subjects and their respective list of affected parameters
                final_table.loc[true_false[dev_high]['sum'].index, 'Affected_high'] = \
                    true_false[dev_high]['sum'] >= max_of_max['# of params']
                final_table.loc[true_false[dev_high]['sum_med'].index, 'Affected_med'] = \
                    (max_of_max['# of params'] >= true_false[dev_high]['sum_med']) & \
                    (true_false[dev_high]['sum_med'] >= max_of_max['# of params'])
                high_columns = included_param_list
                med_columns = [c + '_med' for c in included_param_list if c + '_med' in true_false[dev_high].columns]
                # only_true_false_low_df = only_true_false_df[[columns_low]]
                attended_params_high = true_false[dev_high][high_columns].apply(lambda row: row.index[row.astype(bool)].tolist(), 1)  #
                attended_params_med = true_false[dev_high][med_columns].apply(lambda row: row.index[row.astype(bool)].tolist(), 1)  #

                final_table.loc[true_false[dev_high]['sum'].index, 'Params_high'] = attended_params_high
                final_table.loc[true_false[dev_high]['sum_med'].index, 'Params_med'] = attended_params_med

        # 13. create the final table of subjects and their respective list of affected parameters
        'final_table', final_table
        if st.button('Save table to file?'):
            if second_level:
                levels = 2
            else:
                levels = 1
            final_table_file = 'output\\final_table_' + str(levels) + '_levels' + data_file[2:]
            final_table.to_csv(final_table_file, index=False, header=True)
            st.write('File', final_table_file, 'saved.')
