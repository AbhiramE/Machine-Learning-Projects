import numpy as np
import pandas as pd


# project_id,name,desc,goal,keywords,disable_communication,
# country,currency,deadline,state_changed_at,created_at,
# launched_at,backers_count,final_status

# project_id,name,desc,goal,keywords,disable_communication,
# country,currency,deadline,state_changed_at,created_at,\
# launched_at


def goalDist(from_goal, to_goal):
    df = pd.read_csv('train.csv')
    new_df = df[from_goal <= df['goal']]
    new_df = new_df[to_goal > new_df['goal']][['goal', 'final_status']]

    print to_goal, len(new_df), new_df.groupby(['final_status']).size() / len(new_df)
    print "\n"


def countryDist():
    df = pd.read_csv('train.csv')
    new_df = df[0 == df['final_status']]
    new_df = new_df[['country', 'final_status']]

    failed_df = new_df.groupby(['country'], as_index=False).agg(['count'])

    new_df = df[1 == df['final_status']]
    new_df = new_df[['country', 'final_status']]
    success_df = new_df.groupby(['country'], as_index=False).agg(['count'])

    merged_df = pd.concat([failed_df, success_df], axis=1, join='inner')

    merged_df['success_ratio'] = merged_df.apply(lambda row: float(row[1]) / (row[0] + row[1]), axis=1)

    print merged_df


def currencyDist():
    df = pd.read_csv('train.csv')
    new_df = df[0 == df['final_status']]
    new_df = new_df[['currency', 'final_status']]

    failed_df = new_df.groupby(['currency'], as_index=False).agg(['count'])

    new_df = df[1 == df['final_status']]
    new_df = new_df[['currency', 'final_status']]
    success_df = new_df.groupby(['currency'], as_index=False).agg(['count'])

    merged_df = pd.concat([failed_df, success_df], axis=1, join='inner')

    merged_df['success_ratio'] = merged_df.apply(lambda row: float(row[1]) / (row[0] + row[1]), axis=1)

    print merged_df


# goal_lengths = [0, 1000, 10000, 20000, 100000, 1000000, 1000000000]
# for i in range(1, len(goal_lengths)):
#    goalDist(goal_lengths[i - 1], goal_lengths[i])


currencyDist()
