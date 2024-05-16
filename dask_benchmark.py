import dask
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
import dask_cudf
import cuml
from cuml.cluster import KMeans
import numpy as np
import cupy as cp
import cudf

def create_data(n_samples=1000000, n_features=50):
    # Generate random data on the GPU
    data = cp.random.random((n_samples, n_features))
    return dask_cudf.from_cudf(cudf.DataFrame(data), npartitions=10)

def main():
    # Connect to the Dask cluster
    client = Client("tcp://172.29.184.162:8786")  # Scheduler IP address

    print("Connected to Dask cluster")
    
    # Create random dataset
    data = create_data()

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=10, init="k-means||", max_iter=300)
    
    # Compute and measure time
    result = kmeans.fit(data)
    
    print("K-means clustering completed")
    print("Cluster centers:", result.cluster_centers_)

if __name__ == "__main__":
    main()
