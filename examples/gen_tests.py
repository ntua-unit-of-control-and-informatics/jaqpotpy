import csv
import numpy as np

reader = csv.reader(
    open("/Users/pantelispanka/P1-01-TTD_target_download.txt"), delimiter="\t")
i = 0
for row in reader:
    print(row)
#     if row[0] == '':
#         i += 1
# print(i)

# A = np.matrix([
#     [0, 1, 0, 0],
#     [0, 0, 1, 1],
#     [0, 1, 0, 0],
#     [1, 0, 1, 0]],
#     dtype=float
# )
#
# print(A)
#
# X = np.matrix([
#             [i, -i]
#             for i in range(A.shape[0])
#         ], dtype=float)
#
# print(X)
#
# print( A * X )