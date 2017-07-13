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

    # Converting also t to true and f to false
    for tf_feature in ['host_is_superhost', 'host_identity_verified', 'host_has_profile_pic',
                       'is_location_exact', 'requires_license', 'instant_bookable',
                       'require_guest_profile_picture', 'require_guest_phone_verification']:
        features[tf_feature] = listings[tf_feature].map(lambda s: False if s == "f" else True)

    # Encoding all categorical features using in built Pandas function
    for categorical_feature in ['neighbourhood_cleansed', 'property_type', 'room_type', 'bed_type']:
        features = pd.concat([features, pd.get_dummies(listings[categorical_feature])], axis=1)

    # Filling the empty values with the median of that column
    for col in features.columns[features.isnull().any()]:
        features[col] = features[col].fillna(features[col].median())

    # Removing lisitings crazily prized
    data = features.query('price <= 600')

preprocess(listings)
