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


def durationDist(start, end):
    end_day = end
    start = start * 24 * 60 * 60
    end = end * 24 * 60 * 60

    df = pd.read_csv('train.csv')
    df['duration'] = df.apply(lambda row: float(row['deadline']) - row['launched_at'], axis=1)
    df = df[['duration', 'final_status']]

    new_df = df[start <= df['duration']]
    new_df = new_df[end > new_df['duration']]

    print end_day, len(new_df), new_df.groupby(['final_status']).size() / len(new_df)
    print "\n"


def communication_dist():
    df = pd.read_csv('train.csv')
    df = df[['disable_communication', 'final_status']]
    disabled_df = df[df['disable_communication'] == False]
    new_df = df[True == df['disable_communication']]

    print len(new_df), len(disabled_df)
    # print len(new_df), new_df.groupby(['final_status']).size()
    # print len(disabled_df), disabled_df.groupby(['final_status']).size()
    print "\n"


def keywords():
    df = pd.read_csv('train.csv')
    print df[['keywords', 'name']]


def state_change():
    df = pd.read_csv('train.csv')
    df['state_changed_before'] = df.apply(lambda row: row['deadline'] > row['state_changed_at'], axis=1)
    df = df[['state_changed_before', 'final_status']]

    print df.groupby(['final_status', 'state_changed_before']).size()


# goal_lengths = [0, 1000, 10000, 20000, 100000, 1000000, 1000000000]
# for i in range(1, len(goal_lengths)):
#    goalDist(goal_lengths[i - 1], goal_lengths[i])


# duration_lengths = [0, 5, 10, 15, 20, 25, 30, 35, 45, 60]
# for i in range(1, len(duration_lengths)):
#    durationDist(duration_lengths[i - 1], duration_lengths[i])

countryDist()
