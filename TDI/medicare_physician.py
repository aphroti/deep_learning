import pandas as pd
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm
from os.path import expanduser

pd.options.mode.chained_assignment = None
desired_width = 180
pd.set_option('display.width', desired_width)

############################################
# Import datasets downloaded from          #
# data.medicare.gov/data/physician-compare #
############################################
# write function that cleans column names
def clean_col_name(df):

    col_list = df.columns.tolist()
    for col in col_list:
        col_new = col.replace(' ', '_').replace("'", '').replace('®', '').lower()
        df = df.rename(columns={col: col_new})

    return df

# set import path
import_path = expanduser('~') + '/Downloads/'

# set number of rows to import
nrows = None

# main data frame
col_to_use = ['NPI', 'PAC ID', 'Last Name', 'First Name', 'Gender', 'Credential', 'Graduation year']
df = clean_col_name(pd.read_csv(import_path + 'Physician_Compare_National_Downloadable_File.csv', usecols=col_to_use, nrows=nrows))

# rating data frame
performance_group = clean_col_name(pd.read_csv(import_path + 'Physician_Compare_2015_Group_Public_Reporting___Performance_Scores.csv', nrows=nrows))
experience_group = clean_col_name(pd.read_csv(import_path + 'Physician_Compare_2015_Group_Public_Reporting_-_Patient_Experience.csv', nrows=nrows))
performance_individual = clean_col_name(pd.read_csv(import_path + 'Physician_Compare_2015_Individual_EP_Public_Reporting___Performance_Scores.csv', nrows=nrows))

###########################################
# How many clinicians are in the dataset? #
###########################################
# check if there are NULL NPIs
print('Number of NULL NPIs: {}'.format(sum(df.npi.isnull())))  # 0

# check if there are NULL PAC IDs
print('Number of NULL PAC IDs: {}'.format(sum(df.pac_id.isnull())))  # 0

# number of unique NPI numbers
print('Number of Unique NPIs: {}'.format(len(df.npi.unique())))  # 1070395

# number of unique PAC IDs
print('Number of Unique PAC IDs: {}'.format(len(df.pac_id.unique())))  # 1070399

# more PAD IDs than NPIs, check what's going on
# find unique PAC IDs under each unique NPI
npi_unique = df.groupby('npi').pac_id.unique().reset_index()

# find number of unique PAC IDs under each NPI
npi_unique.loc[:, 'length'] = npi_unique.pac_id.apply(lambda x: len(x))

# find clinicians with more than one PAC ID
print(npi_unique[npi_unique.length >= 2])  # four clinicians have two PAC IDs each

# choose to use NPI to identify clinicians
print('Number of clinicians in the dataset: {}'.format(len(df.npi.unique())))

###################################################
# What is the ratio of male to female clinicians? #
###################################################
# check if there are NULL/unknown genders
print('Number of NULL gender: {}'.format(sum(df.gender.isnull())))  # 0
print('Number of unknown gender: {}'.format(sum(df.gender == 'U')))  # 1

# calculate ratio
female = sum(df.gender == 'F')
male = sum(df.gender == 'M')
print('Ratio of male to female clinicians: {}'.format(round(male/female, 2)))

#####################################################
# What is the highest ratio of female clinicians to #
# male clinicians with a given type of credential?  #
#####################################################
# check if there are NULL credentials
print('Number of NULL credentials: {}'.format(sum(df.credential.isnull())))  # 1941656

# fill NULL credentials with "Unknown"
df.loc[df.credential.isnull(), 'credential'] = 'Unknown'

# get a list of unique credentials
credentials = df.credential.unique().tolist()

# write function that calculates male to female ratio
def calc_female_to_male(df):
    female = sum(df.gender == 'F')
    male = sum(df.gender == 'M')
    ratio = round(female/male, 2)
    return ratio

# calculate male to female ratio by credential type
ratio_df = df.groupby(['credential']).apply(lambda x: calc_female_to_male(x)).reset_index().rename(columns={0: 'female_to_male'})

# observed that "SCW" has only one female record
# find biggest female to male ratio among other credentials
ratio_df = ratio_df[ratio_df.female_to_male < math.inf]
highest_cred = ratio_df[ratio_df.female_to_male == max(ratio_df.female_to_male)].credential.tolist()[0]
print('Highest ratio of female to male credential: {}'.format(highest_cred))

#############################################################################
# How many states have fewer than 10 healthcare facilities in this dataset? #
#############################################################################
# use Group Public Reporting Performance Score table for this question
# check number of unique states
print('Number of unique states: {}'.format(len(performance_group.state.unique())))

# check if there are NULL Group PAC IDs
print('Number of NULL Group PAC IDs: {}'.format(len(performance_group.group_pac_id.isnull())))

# calculate number of facilities by state
state_count = performance_group.groupby('state').group_pac_id.unique().reset_index()
state_count.loc[:, 'length'] = state_count.group_pac_id.apply(lambda x: len(x))
print('Number of states wither fewer than 10 facilities: {}'.format(state_count[state_count.length < 10].shape[0]))

##########################################################################################
# Compute the average measure performance rate for each clinician (across all measures). #
# Consider the distribution of these average rates for individuals who have at least 10. #
# What is the standard deviation of that distribution?                                   #
##########################################################################################
# check number of unique NPIs in Individual Public Reporting Performance Score table
print('Number of unique NPIs: {}'.format(len(performance_individual.pac_id.unique())))

# calculate performance rate across all measures for each clinician
rating_df = performance_individual.groupby(['npi', 'measure_identifier']).measure_performance_rate.mean().reset_index()

# calculate number of measures rated for each clinician
measure_count_df = rating_df.groupby(['npi']).measure_identifier.count().reset_index()
measure_count_df.rename(columns={'measure_identifier': 'measure_count'}, inplace=True)

# find clinicians with more than 10 measures
clinicians_to_study = measure_count_df[measure_count_df.measure_count >= 10].npi.tolist()
print('Number of clinicians with more than 10 measures: {}'.format(len(clinicians_to_study)))

# study rating distribution across measures for each clinician by plotting
# superposed histograms of each clinician's ratings
rating_study = rating_df[rating_df.npi.isin(clinicians_to_study)]
# bins = np.linspace(0, 10, 100)
# for npi in tqdm(rating_study.npi.unique()):
#     rating_study_small = rating_study[rating_study.npi == npi]
#     rates_to_plot = rating_study_small.measure_performance_rate.tolist()
#     plt.hist(rates_to_plot, alpha=0.05)
# plt.show()

# from the superposed histogram, we see that the distribution of each clinician's ratings
# is a uniform distribution from 0 to 100
std = round((1/12 * (100 - 0)**2) ** 0.5, 1)
print("Standard deviation of each clinician's rates is {}".format(std))

####################################################################
# What is the absolute difference in the average performance rates #
# between doctors (MD) and nurse practitioners (NP)?               #
####################################################################
# calculate average rates across all measures for each clinician
rating_study = rating_study.groupby(['npi']).measure_performance_rate.mean().reset_index()

# create NPI to credential mapping
npi_to_credential = df[df.npi.isin(clinicians_to_study)].groupby('npi').credential.nth(1).reset_index()
npi_credential_map = dict(zip(npi_to_credential.npi, npi_to_credential.credential))
rating_study.loc[:, 'credential'] = rating_study.npi.map(npi_credential_map)

# slice ratings by MD and NP; calculate absolute difference in ratings
md_rating = rating_study[rating_study.credential == 'MD']
np_rating = rating_study[rating_study.credential == 'NP']
abs_diff = round(abs(md_rating.measure_performance_rate.mean() - np_rating.measure_performance_rate.mean()), 1)
print('Absolute difference between MD and NP rates is: {}'.format(abs_diff))

#########################################################################
# What is the p-value of the difference in MD and NP performance rates? #
#########################################################################
ttest = sp.stats.ttest_ind(md_rating.measure_performance_rate, np_rating.measure_performance_rate, equal_var=True)
pvalue = round(ttest[1], 4)
print('P-value of MD-NP performance rates difference is: {}'.format(pvalue))

##########################################################################################
# What is the p-value of the linear regression of performance rates vs. graduation year? #
# Consider the average performance rates of every MD who graduated between 1973 and 2003 #
##########################################################################################
# create NPI to graduate year mapping
npi_to_grad = df[df.npi.isin(clinicians_to_study)].groupby('npi').graduation_year.nth(1).reset_index()
npi_grad_map = dict(zip(npi_to_grad.npi, npi_to_grad.graduation_year))
md_rating.loc[:, 'graduation_year'] = md_rating.npi.map(npi_grad_map)

# filter doctors graduated between 1973 and 2003
md_rating = md_rating[(md_rating.graduation_year >= 1973) & (md_rating.graduation_year <= 2003)]

# fit regression between performance rates and graduation year; find p-value
X = sm.add_constant(md_rating.graduation_year)
y = md_rating.measure_performance_rate
model = sm.OLS(y, X)
print(model.fit().summary())

