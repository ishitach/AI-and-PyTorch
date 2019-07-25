labels = np.array([9,9,3,6,9,9,9,9,8,2])
counts = np.bincount(labels, minlength =10)
counts

query_result = np.argmax(counts)
query_result

!pip install pysft
