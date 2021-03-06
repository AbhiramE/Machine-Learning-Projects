import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

listings = pd.read_csv("../data/listings.csv")


def preprocess():
    listings['price'] = listings['price'].map(lambda p: int(p[1:-3].replace(",", "")))
    listings['amenities'] = listings['amenities'].map(
        lambda a: "|".join([amn.replace("}", "").replace("{", "").replace('"', "")
                            for amn in a.split(",")]))

    amenities = np.unique(np.concatenate(listings['amenities'].map(lambda amns: amns.split("|"))))[1:]
    amenities_matrix = np.array([listings['amenities'].map(lambda amns: amn in amns) for amn in amenities])

    print (amenities_matrix)

    return build_features(amenities, amenities_matrix)


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
    y = data['price']
    X = data.drop('price', axis='columns')

    return X, y


def pipeline(X_train, y_train, X_validate, y_validate):
    clf = GridSearchCV(SVR(), cv=5,
                       param_grid={'kernel': ('linear', 'rbf'), "C": [1e0, 1e1, 1e2, 1e3],
                                   "gamma": np.logspace(-2, 2, 5)})
    clf.fit(X_train, y_train)

    print("Grid search done")

    clf = SVR(kernel=clf.best_params_['kernel'], C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
    rfe = RFE(estimator=clf, n_features_to_select=50, step=1)
    steps = [('svr', clf),
             ('feature_selection', rfe)]
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    y_prediction = pipeline.predict(X_validate)
    rmse = mean_squared_error(y_validate, y_prediction)
    print(rmse ** 0.5)


X, y = preprocess()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2)
pipeline(X_train, y_train, X_validate, y_validate)
