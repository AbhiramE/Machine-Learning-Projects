import numpy as np
import pandas as pd

listings = pd.read_csv("../data/listings.csv")


def preprocess(lisitings):
    listings['price'] = listings['price'].map(lambda p: int(p[1:-3].replace(",", "")))
    listings['amenities'] = listings['amenities'].map(
        lambda a: "|".join([amn.replace("}", "").replace("{", "").replace('"', "")
                            for amn in a.split(",")]))

    amenities = np.unique(np.concatenate(listings['amenities'].map(lambda amns: amns.split("|"))))[1:]
    amenities_matrix = np.array([listings['amenities'].map(lambda amns: amn in amns) for amn in amenities])

    print amenities_matrix

    build_features(amenities, amenities_matrix)


def build_features(amenities, amenities_matrix):
    features = listings[['host_listings_count', 'host_total_listings_count', 'accommodates',
                         'bathrooms', 'bedrooms', 'beds', 'price', 'guests_included', 'number_of_reviews',
                         'review_scores_rating']]
    features = pd.concat([features, pd.DataFrame(data=amenities_matrix.T, columns=amenities)], axis=1)

    print list(features)

preprocess(listings)