from textblob import TextBlob, Word
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import random

dict = {

    'art': ["Ceramics", "Conceptual Art", "Digital Art", "Illustration", "Installations", "Mixed Media",
            "Painting", "Performance Art", "Public Art", "Sculpture", "Textiles", "Video Art", "draw", "paint"
                                                                                                       "art"],

    'comics': ["Webcomics", "Anthologies", "Comic Books", "Events", "Graphic Novels", "DC Comics",
               "Marvel", "comics"],

    'crafts': ["Candles", "Crochet", "DIY", "Embroidery", "Glass", "Knitting", "Pottery", "Printing", "Quilts",
               "Stationery", "Taxidermy", "Weaving", "Woodworking", 'crafts', 'Sculpt'],

    'dance': ['dance', 'salsa'],

    'design': ["Architecture", "Civic Design", "Graphic Design", "Interactive Design", "Product Design",
               "Typography", "design"],

    'fashion': ["Accessories", "Apparel", "Childrenswear", "Couture", "Footwear", "Jewelry", "Pet Fashion"
        , "Ready-to-wear", "fashion", "clothes", "styles"],

    'film & video': ['film', 'webseries', 'videos'],

    'food': ["Bacon", "Community Gardens", "Cookbooks", "Drinks", "Events", "Farmers", "Farms",
             "Food Trucks", "Restaurants", "Small Batch", "Spaces", "Vegan", "food", "ingredients",
             "cake"],

    'games': ["Gaming Hardware", "Live Games", "Mobile Games", "Playing Cards", "Puzzles", "Tabletop Games",
              "Video Games", "games", "game", "multiplayer", "fps", "rpg"],

    'journalism': ['journalism', 'news'],

    'technology': ["3D Printing", "Apps", "Camera Equipment", "DIY Electronics", "Fabrication Tools", "Flight",
                   "Gadgets", "Hardware", "Makerspaces", "Robots", "Software", "Space Exploration",
                   "Wearables"],

    'music': ["Blues", "Chiptune", "Classical Music", "Comedy", "Country", "Folk", "Electronic Music", "Faith",
              "Hip-Hop", "Indie Rock", "Jazz", "Latin", "Metal", "Pop", "Punk", "R&B", "Rock", "World Music",
              "itunes", "spotify", "music", "songs", "EP", "album", "lp"],

    'photography': ['photo', 'photobooks', 'landscape', 'photography'],

    'publishing': ["Academic", "Anthologies", "Art Books", "Calendars", "Children's Books", "Fiction",
                   "Letterpress", "Literary Journals", "Nonfiction", "Periodicals", "Poetry", "Radio", "Podcasts",
                   "Translations", "Young Adult", "Zines", "Literary Spaces", "novels", "story", "stories",
                   "writing", "posters", "publish", "author", "book", "written", "novella", "novel"],
    'theater': ['musical', 'festivals', 'plays', 'theater']
}

categories = dict.keys()
categories.append('other')
print categories

countries = ['AU', 'CA', 'DK', 'GB', 'IE', 'NL', 'NO', 'NZ', 'SE', 'US', 'DE']


def find_category(text1, text2):
    text1 += " " + text2

    for key in dict.keys():
        for word in dict[key]:
            if word.lower() in text1.lower():
                return key

    return 'other'


def add_categories():
    df = pd.read_csv('test.csv')
    matrix = df.as_matrix()
    new_matrix = []

    for row in matrix:
        category = find_category(str(row[1]), str(row[2]))
        new_row = np.append(row, category)

        if len(new_matrix) == 0:
            new_matrix = new_row
        else:
            new_matrix = np.vstack([new_matrix, new_row])

    pd.DataFrame(new_matrix).to_csv('final_test.csv', sep=",")


def add_features_and_drop_columns():
    df = pd.read_csv('final_test.csv')
    df = df.drop(df.columns[[0]], 1)
    df['duration'] = df.apply(lambda row: float(row['deadline']) - row['launched_at'], axis=1)
    df['duration<7'] = df.apply(lambda row: True if 0 < row['duration'] < 7 * 24 * 60 * 60 else False, axis=1)
    df['duration<15'] = df.apply(lambda row: True if 7 * 24 * 60 * 60 < row['duration'] < 15 * 24 * 60 * 60
    else False, axis=1)
    df['duration<25'] = df.apply(lambda row: True if 15 * 24 * 60 * 60 < row['duration'] < 25 * 24 * 60 * 60
    else False, axis=1)
    df['duration<35'] = df.apply(lambda row: True if 25 * 24 * 60 * 60 < row['duration'] < 35 * 24 * 60 * 60
    else False, axis=1)
    df['duration<45'] = df.apply(lambda row: True if 35 * 24 * 60 * 60 < row['duration'] < 45 * 24 * 60 * 60
    else False, axis=1)
    df['duration<60'] = df.apply(lambda row: True if 45 * 24 * 60 * 60 < row['duration'] <= 60 * 24 * 60 * 60
    else False, axis=1)
    df['state_changed_before'] = df.apply(lambda row: row['deadline'] > row['state_changed_at'], axis=1)

    df['encoded_category'] = df.apply(lambda row: categories.index(row['category']), axis=1)
    df['encoded_country'] = df.apply(lambda row:
                                     countries.index(row['country'])
                                     if row['country'] in countries
                                     else -1, axis=1)

    df = df.drop('duration', 1)
    df = df.drop('deadline', 1)
    df = df.drop('launched_at', 1)
    df = df.drop('keywords', 1)
    df = df.drop('currency', 1)

    cols = list(df.columns.values)
    print cols

    print df.head(5)

    df = df[['project_id', 'goal', 'disable_communication', 'state_changed_at',
             'created_at', 'duration<7', 'duration<15', 'duration<25', 'duration<35', 'duration<45',
             'duration<60', 'state_changed_before', 'encoded_category', 'encoded_country']]

    header = ['project_id', 'goal', 'disable_communication', 'state_changed_at',
              'created_at', 'duration<7', 'duration<15', 'duration<25', 'duration<35', 'duration<45',
              'duration<60', 'state_changed_before', 'encoded_category', 'encoded_country']

    df.to_csv('final_test.csv', sep=',', columns=header)


'''
def encode():
    df = pd.read_csv('new_train.csv')

    df = df.drop(df.columns[[0]], 1)

    df = df[['goal', 'disable_communication', 'state_changed_at', 'created_at',
             'duration<7', 'duration<15', 'duration<25', 'duration<35', 'duration<45', 'duration<60',
             'state_changed_before', 'encoded_category', 'encoded_country', 'final_status']]
    header = ['goal', 'disable_communication', 'state_changed_at', 'created_at',
              'duration<7', 'duration<15', 'duration<25', 'duration<35', 'duration<45', 'duration<60',
              'state_changed_before', 'encoded_category', 'encoded_country', 'final_status']

    print len(header)
    df.to_csv('final_train.csv', sep=',', columns=header)
'''

# add_features_and_drop_columns()
# add_categories()
add_features_and_drop_columns()
