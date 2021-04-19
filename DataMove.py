import numpy as np
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from sklearn.model_selection import train_test_split


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


    X_train, X_test, Y_train, Y_test = train_test_split(x_arr, y_arr, test_size=.25, random_state=3)
    print(Y_test)
    print(Y_test.shape)
    print(Y_train.shape)
main()
