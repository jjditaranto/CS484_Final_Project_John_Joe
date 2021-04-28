import numpy as np
from sklearn import preprocessing
enc = preprocessing.LabelEncoder()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


def main():
    dataStrings = np.loadtxt("CS484CleanDataset.csv", skiprows=1, dtype=str, delimiter=',')
    x_arr = np.loadtxt("InputValues.csv", skiprows=1, dtype=str, delimiter=',')
    y_arr = np.loadtxt("OutputMPG.csv", skiprows=1, dtype=str, delimiter=',')
   # # data = np.loadtxt("CS484CleanDataset.csv", skiprows=1, dtype={'names': ('Symboling', 'Make', 'Fuel-Type',
   #                                                                          'Aspiration', '# of Doors', 'Body-Style',
   #                                                                          'Drive-Wheels', 'Engine-Location',
   #                                                                          'Wheel-Base',
   #                                                                          'Length', 'Width', 'Height', 'Curb-Weight',
   #                                                                          'Engine Type', '# of Cylinders',
   #                                                                          'Engine Size',
   #                                                                          'Fuel System', 'Bore', 'Stroke',
   #                                                                          'Compression Ratio', 'Horsepower',
   #                                                                          'Peak RPM', 'City MPG',
   #                                                                          'Highway MPG', 'Price'),
   #                                                                'formats': ('i4', 'S1', '')})
    #print(dataStrings.shape)
    #print(dataStrings)
    #tts = train_test_split(dataStrings, test_size=None)
    #print(le.fit_transform(dataStrings))

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

    encoded_y_array = np.zeros_like(y_arr, dtype=np.double)
    encoded_y_array[:, 0] = enc.fit_transform(y_arr[:, 0])
    encoded_y_array[:, 1] = enc.fit_transform(y_arr[:, 1])
    #print(encoded_y_array)

    X_train, X_test, Y_train, Y_test = train_test_split(encoded_x_array, encoded_y_array, test_size=.25, random_state=3)
    #print(Y_test)
    print(X_test.shape)
    print(Y_test.shape)
   # print(Y_train.shape)
    neigh = KNeighborsRegressor()
    neigh.fit(X_train, Y_train)
    print("Score for Training set (City MPG): ", neigh.score(, Y_test, sample_weight=None))
    #print("Score for Training set (Highway MPG) : ", neigh.score(X_test, Y_test[:, 1], sample_weight=None))


main()
