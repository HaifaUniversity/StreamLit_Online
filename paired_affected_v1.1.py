import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


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
task_groups = {'DD': ['auc_dd', 'k', 'percentnow'],
               'DPX': ['per_acc', 'per_acc_correct_ax', 'per_acc_correct_bx', 'per_acc_correct_ay',
                       'per_acc_correct_by', 'ave_cue_rt_correct', 'ave_probe_rt_correct'],
               'RMET': ['rmet_per_acc', 'rmet_average_rt_correct', 'rmet_average_rt_incorrect', 'difficult_session',
                        'easy_session', 'negative_session', 'neutral_session', 'positive_session']}

st.write("# Analyse Paired Differences")
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
        group_mean[group] = data_df[data_df[group_col] == int(group[-1])][list(param_list)].mean()  # collect all of the groups means
        group_std[group] = data_df[data_df[group_col] == int(group[-1])][list(param_list)].std()  # collect all of the groups SDs
    group_std

    # 1.1 (Roee) - set mode to paired
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

    # for std in stds:
    std = 1
    grouped_diffs[std_column_name] = per_subject_diffs.std() * std
    grouped_diffs = grouped_diffs.rename(columns={std_column_name: 'std_' + str(std)})
    affected[std] = per_subject_diffs.copy()
    affected_full[std] = per_subject_diffs.copy()

    grouped_diffs['+1'] = 0
    grouped_diffs['-1'] = 0
    grouped_diffs['0'] = 0
    for i, row in grouped_diffs.iterrows():
        for j, subject in per_subject_diffs[i].dropna().items():
            if (row['mean'] - row['std_' + str(std)] > subject) or \
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
    for task, groups in task_groups.items():
        affected[std]['affected_' + task] = affected[std].loc[:, task_groups[task]].sum(axis=1)
        affected_full[std]['affected_' + task] = affected[std]['affected_' + task]
    # grouped_diffs['affected'] = grouped_diffs['+1'] + grouped_diffs['-1']
    for parameters in range(1, len(grouped_diffs) + 1):
        std_df.loc[parameters, '{:.1f}'.format(std)] = \
            len(affected[std].loc[affected[std]['affected'] >= parameters]) / len(per_subject_diffs) * 100
    if std in show_list:
        std, grouped_diffs
    # std, affected
    # show table of standard deviations with included affected subjects by number of parameters percentages

    affected_full[std]['affected_DD_DPX'] = affected_full[std]['affected_DD'] + affected_full[std]['affected_DPX']
    affected_full[std]['affected_DD_RMET'] = affected_full[std]['affected_DD'] + affected_full[std]['affected_RMET']
    affected_full[std]['affected_DPX_RMET'] = affected_full[std]['affected_DPX'] + affected_full[std]['affected_RMET']
    affected_full[std]['affected_DD_DPX_RMET'] = affected_full[std]['affected_DD'] + affected_full[std]['affected_DPX']\
                                                 + affected_full[std]['affected_RMET']

    affected_full[std]['bin_affected_DD'] = affected_full[std]['affected_DD'].astype(bool).astype(int)
    affected_full[std]['bin_affected_DPX'] = affected_full[std]['affected_DPX'].astype(bool).astype(int)
    affected_full[std]['bin_affected_RMET'] = affected_full[std]['affected_RMET'].astype(bool).astype(int)
    affected_full[std]['bin_affected_DD_DPX'] = affected_full[std]['bin_affected_DD'] &\
                                                affected_full[std]['bin_affected_DPX']
    affected_full[std]['bin_affected_DD_RMET'] = affected_full[std]['bin_affected_DD'] & \
                                             affected_full[std]['bin_affected_RMET']
    affected_full[std]['bin_affected_DPX_RMET'] = affected_full[std]['bin_affected_DPX'] & \
                                              affected_full[std]['bin_affected_RMET']

    affected_full[std]['bin_affected_DD_DPX_RMET'] = affected_full[std]['bin_affected_DD'] &\
                                                 affected_full[std]['bin_affected_DPX'] &\
                                                 affected_full[std]['bin_affected_RMET']
    affected_full_percentages = {}
    for col in ['bin_affected_DD', 'bin_affected_DPX', 'bin_affected_RMET', 'bin_affected_DD_DPX',
                'bin_affected_DD_RMET', 'bin_affected_DPX_RMET', 'bin_affected_DD_DPX_RMET']:
        affected_full_percentages[col] = affected_full[std][col].sum() / len(affected_full[std][col]) * 100
    affected_full_percentages
    st.write('Table of percentage of affected subjects by number of paramters')
    std_df


    # display final table with an option to choose standard deviation
    # str_stds = list(str(std) for std in stds)
    # std = float(st.selectbox('select SD to view', str_stds, str_stds.index('1.0')))
    affected_full[std]

    # download final table
    filename = 'affected_sd_' + str(std) + '.csv'
    download_data = convert_df(affected_full[std])
    st.download_button(label='Download CSV',
                       data=download_data,
                       file_name=filename,
                       mime='text/csv')
