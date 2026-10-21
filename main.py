# Porównanie z CCA, uruchamiane tylko przy liczeniu wyników odniesienia.
# import numpy as np
# from src.cca import process
# from src.config import mat_files_filtered, path_filtered
# from src.preprocessing import read_file
#
# read_file(path_filtered, mat_files_filtered)
# x_features, y_labels, subject_ids = process(mat_files_filtered)
# print(x_features.shape)
# print(y_labels.shape)
# print(subject_ids.shape)
#
# i = 0
# print("Osoba:", subject_ids[i])
# print("Prawdziwy target:", y_labels[i])
# print("Wyniki CCA:", x_features[i])
# print("Przewidziany target:", np.argmax(x_features[i]))
