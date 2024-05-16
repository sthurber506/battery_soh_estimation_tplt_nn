if __name__ == "__main__":
    # Initialize Dask cluster with LocalCUDACluster
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print("Connected to Dask cluster")

    # Increase task complexity
    n_samples = 10_000_000  # More samples
    n_features = 100
    n_clusters = 10

    # Generate random data
    X = np.random.rand(n_samples, n_features).astype(np.float32)

    # Distribute data using Dask
    dx = dask.array.from_array(X, chunks=(n_samples // 10, n_features))

    # K-means Clustering
    kmeans = KMeans(n_clusters=n_clusters, init="scalable-k-means++", random_state=0)

    # Measure execution time
    start_time = time.time()
    kmeans.fit(dx)
    end_time = time.time()

    print(f"K-means clustering completed in {end_time - start_time:.2f} seconds")
    print(f"Cluster centers: {kmeans.cluster_centers_}")

    with performance_report(filename="dask_benchmark_report.html"):
        kmeans.fit(dx)
