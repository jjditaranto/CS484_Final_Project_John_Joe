import numpy as np
from sklearn import preprocessing
enc = preprocessing.LabelEncoder()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid


def main():
    # Regression Problem Technique

    # Part 2 Step 2
    # load selected feature data into numpy array and encode the array
    x_arr = np.loadtxt("InputValues.csv", skiprows=1, dtype=str, delimiter=',')
    y_arr = np.loadtxt("OutputMPG.csv", skiprows=1, dtype=str, delimiter=',')

    # Part 2 Step 4
    # Encoding the feature data array
    encoded_x_array = np.zeros_like(x_arr, dtype=np.double)
    encoded_x_array[:, 1] = enc.fit_transform(x_arr[:, 1])
    encoded_x_array[:, 2] = enc.fit_transform(x_arr[:, 2])
    encoded_x_array[:, 3] = enc.fit_transform(x_arr[:, 3])
    encoded_x_array[:, 4] = enc.fit_transform(x_arr[:, 4])
    encoded_x_array[:, 5] = enc.fit_transform(x_arr[:, 5])
    encoded_x_array[:, 6] = enc.fit_transform(x_arr[:, 6])
    encoded_x_array[:, 7] = enc.fit_transform(x_arr[:, 7])
    encoded_x_array[:, 8] = enc.fit_transform(x_arr[:, 8])
    encoded_x_array[:, 9] = enc.fit_transform(x_arr[:, 9])
    encoded_x_array[:, 10] = enc.fit_transform(x_arr[:, 10])
    encoded_x_array[:, 11] = enc.fit_transform(x_arr[:, 11])
    encoded_x_array[:, 12] = enc.fit_transform(x_arr[:, 12])
    encoded_x_array[:, 13] = enc.fit_transform(x_arr[:, 13])
    encoded_x_array[:, 14] = enc.fit_transform(x_arr[:, 14])
    encoded_x_array[:, 15] = enc.fit_transform(x_arr[:, 15])
    encoded_x_array[:, 16] = enc.fit_transform(x_arr[:, 16])
    encoded_x_array[:, 17] = enc.fit_transform(x_arr[:, 17])
    encoded_x_array[:, 18] = enc.fit_transform(x_arr[:, 18])
    encoded_x_array[:, 19] = enc.fit_transform(x_arr[:, 19])
    encoded_x_array[:, 20] = enc.fit_transform(x_arr[:, 20])
    encoded_x_array[:, 21] = enc.fit_transform(x_arr[:, 21])
    encoded_x_array[:, 22] = enc.fit_transform(x_arr[:, 22])
    #print(encoded_x_array)

    # Part 2 Step 4
    # Encoding the Output Array
    encoded_y_array = np.zeros_like(y_arr, dtype=np.double)
    encoded_y_array[:, 0] = enc.fit_transform(y_arr[:, 0])
    encoded_y_array[:, 1] = enc.fit_transform(y_arr[:, 1])
    #print(encoded_y_array)

    # Part 2 Step 3
    # split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(encoded_x_array, encoded_y_array, test_size=.25, random_state=3)

    # reshaping y training output arrays so they are 2D to avoid error with score function
    y_train_city = np.zeros((Y_train.shape[0], 1))
    y_train_city[:, 0] = Y_train[:, 0]
    y_train_hwy = np.zeros((Y_train.shape[0], 1))
    y_train_hwy[:, 0] = Y_train[:, 1]

    # reshaping y testing output arrays so they are 2D to avoid error with score function
    y_test_city = np.zeros((Y_test.shape[0], 1))
    y_test_city[:, 0] = Y_test[:, 0]
    y_test_hwy = np.zeros((Y_test.shape[0], 1))
    y_test_hwy[:, 0] = Y_test[:, 1]

    # Part 2 Step 5
    # using K nearest neighbors regressor to fit our data to it for City MPG output
    neigh_city = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', metric='minkowski')
    neigh_city.fit(X_train, y_train_city)
   #print(neigh_city)

    # Part 2 Step 5
    # using K nearest neighbors regressor to fit our data to it for Highway MPG output
    neigh_hwy = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', metric='minkowski')
    neigh_hwy.fit(X_train, y_train_hwy)
    #print(neigh_hwy)

    # Part 2 Step 5
    # Get Regressor Scores
    KNN_city_score = neigh_city.score(X_test, y_test_city, sample_weight=None)
    KNN_hwy_score = neigh_hwy.score(X_test, y_test_hwy, sample_weight=None)
    print("Full Supervised Model Using All Features")
    print("These scores represent the coefficient of determination.")
    print("Score for Training set (City MPG): ", KNN_city_score)
    print("Score for Training set (Highway MPG) : ", KNN_hwy_score, "\n")

    # Part 3
    # Step 1.2
    # Creating Parameter grid for GridSearchCV
    parameter_grid = [{'n_neighbors': [3, 5, 7, 10], 'weights': ['uniform', 'distance'], 'algorithm': ['auto',
                        'ball_tree', 'kd_tree', 'brute'], 'metric': ['minkowski']}]

    # Part 3 Step 1.3
    grid_search_city = GridSearchCV(estimator=neigh_city, param_grid=parameter_grid, cv=5)
    grid_search_city.fit(X_train, y_train_city)

    # Part 3 Step 1.4
    print("The best parameters for City MPG are: ", grid_search_city.best_params_)
    print("The best score for City MPG is: ", grid_search_city.best_score_, "\n")

    # Part 3 Step 1.3
    grid_search_hwy = GridSearchCV(estimator=neigh_hwy, param_grid=parameter_grid, cv=5)
    grid_search_hwy.fit(X_train, y_train_hwy)

    # Part 3 Step 1.4
    print("The best parameters for Highway MPG are: ", grid_search_hwy.best_params_)
    print("The best score for Highway MPG is: ", grid_search_hwy.best_score_, "\n")

    # Part 3 Step 1.4
    # using best parameters to score testing data
    best_neigh_city = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', metric='minkowski')
    best_neigh_city.fit(X_train, y_train_city)
    KNN_best_city_score = best_neigh_city.score(X_test, y_test_city, sample_weight=None)
    print("The score for the best City MPG parameters is: ", KNN_best_city_score)

    # Part 3 Step 1.4
    best_neigh_hwy = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', metric='minkowski')
    best_neigh_hwy.fit(X_train, y_train_hwy)
    KNN_best_hwy_score = best_neigh_hwy.score(X_test, y_test_hwy, sample_weight=None)
    print("The score for the best Highway MPG parameters is: ", KNN_best_hwy_score, "\n")

    # Part 2 Steps 2, 4 and 6
    # feature selection technique
    # load selected feature data into numpy array and encode the array
    selected_feature_arr = np.loadtxt("SelectedFeatureInput.csv", skiprows=1, dtype=str, delimiter=',')
    encoded_sel_array = np.zeros_like(selected_feature_arr, dtype=np.double)
    encoded_sel_array[:, 0] = enc.fit_transform(selected_feature_arr[:, 0])
    encoded_sel_array[:, 1] = enc.fit_transform(selected_feature_arr[:, 1])
    encoded_sel_array[:, 2] = enc.fit_transform(selected_feature_arr[:, 2])

    # Part 2 Step 3
    # split data into training and testing sets
    X_train_sel, X_test_sel, Y_train_sel, Y_test_sel = train_test_split(encoded_sel_array, encoded_y_array,
                                                                        test_size=.25, random_state=3)
    # reshaping y training output arrays so they are 2D to avoid error with score function
    y_train_city_sel = np.zeros((Y_train_sel.shape[0], 1))
    y_train_city_sel[:, 0] = Y_train_sel[:, 0]
    y_train_hwy_sel = np.zeros((Y_train_sel.shape[0], 1))
    y_train_hwy_sel[:, 0] = Y_train_sel[:, 1]

    # reshaping y testing output arrays so they are 2D to avoid error with score function
    y_test_city_sel = np.zeros((Y_test_sel.shape[0], 1))
    y_test_city_sel[:, 0] = Y_test_sel[:, 0]
    y_test_hwy_sel = np.zeros((Y_test_sel.shape[0], 1))
    y_test_hwy_sel[:, 0] = Y_test_sel[:, 1]

    # Part 2 Step 5
    # using K nearest neighbors regressor to fit our data to it for City MPG output
    neigh_city_sel = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', metric='minkowski')
    neigh_city_sel.fit(X_train_sel, y_train_city_sel)
    #print(neigh_city_sel)

    # Part 2 Step 5
    # using K nearest neighbors regressor to fit our data to it for Highway MPG output
    neigh_hwy_sel = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', metric='minkowski')
    neigh_hwy_sel.fit(X_train_sel, y_train_hwy_sel)
    #print(neigh_hwy_sel)

    print("Feature Selection Technique Results")
    print("These scores represent the coefficient of determination.")
    print("Score for Feature Selection Training set (City MPG): ", neigh_city_sel.score(X_test_sel, y_test_city_sel, sample_weight=None))
    print("Score for Feature Selection Training set (Highway MPG) : ", neigh_hwy_sel.score(X_test_sel, y_test_hwy_sel, sample_weight=None))

    # Part 3 Step 1.3
    # Creating Parameter grid for GridSearchCV
    grid_search_sel_city = GridSearchCV(estimator=neigh_city_sel, param_grid=parameter_grid, cv=5)
    grid_search_sel_city.fit(X_train_sel, y_train_city_sel)
    print("The best parameters for Feature Selection City MPG are: ", grid_search_sel_city.best_params_)
    print("The best score for Feature Selection City MPG is: ", grid_search_sel_city.best_score_, "\n")

    # Part 3 Step 1.3
    grid_search_sel_hwy = GridSearchCV(estimator=neigh_hwy_sel, param_grid=parameter_grid, cv=5)
    grid_search_sel_hwy.fit(X_train_sel, y_train_hwy_sel)
    print("The best parameters for Feature Selection Highway MPG are: ", grid_search_sel_hwy.best_params_)
    print("The best score for Feature Selection Highway MPG is: ", grid_search_sel_hwy.best_score_, "\n")

    # Part 3 Step 1.4
    # using best parameters to score testing data
    best_neigh_sel_city = KNeighborsRegressor(n_neighbors=7, weights='distance', algorithm='auto', metric='minkowski')
    best_neigh_sel_city.fit(X_train_sel, y_train_city_sel)
    KNN_best_sel_city_score = best_neigh_sel_city.score(X_test_sel, y_test_city_sel, sample_weight=None)
    print("The score for the best Feature Selection City MPG parameters is: ", KNN_best_sel_city_score)

    # Part 3 Step 1.4
    best_neigh_sel_hwy = KNeighborsRegressor(n_neighbors=7, weights='distance', algorithm='brute', metric='minkowski')
    best_neigh_sel_hwy.fit(X_train_sel, y_train_hwy_sel)
    KNN_best_sel_hwy_score = best_neigh_sel_hwy.score(X_test_sel, y_test_hwy_sel, sample_weight=None)
    print("The score for the best Feature Selection Highway MPG parameters is: ", KNN_best_sel_hwy_score, "\n")

main()
