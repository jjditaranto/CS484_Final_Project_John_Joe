import numpy as np


def main():
    dataStrings = np.loadtxt("CS484CleanDataset.csv", skiprows=1, dtype=str, delimiter=',')
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
    print(dataStrings.shape)
    print(dataStrings)

main()
