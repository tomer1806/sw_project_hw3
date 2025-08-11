import sys
import numpy as np
import subprocess
from sklearn.metrics import silhouette_score
import symnmf as symnmf_c

def error_exit(): # Print the standard error message and quit immediately.
    print("An Error Has Occurred")
    sys.exit(1)

def get_labels_from_centroids(X, centroids):
    #Assign each data point to the closest centroid based on Euclidean distance.
    # centroids shape: (k, d)
    # X shape: (N, d)
    distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)

def main():# Main function to run the comparison between SymNMF and K-means clustering.
    try: 
        if len(sys.argv) != 3:
            error_exit()

        k = int(sys.argv[1])
        file_name = sys.argv[2]

        # Load dataset from file
        X = np.loadtxt(file_name, delimiter=',')
        N = X.shape[0]

        # Silhouette score not defined for k <= 1 or k >= N
        if k <= 1 or k >= N:
            error_exit()

        # ===== SymNMF section =====
        W = symnmf_c.norm(X.tolist())
        if W is None:
            error_exit()

        # Deterministic random seed for reproducibility
        np.random.seed(1234)
        m = np.mean(np.array(W))
        H_init = np.random.uniform(0, 2 * np.sqrt(m / k), size=(N, k))

        H_final = symnmf_c.symnmf(H_init.tolist(), W)
        if H_final is None:
            error_exit()

        nmf_labels = np.argmax(np.array(H_final), axis=1)
        nmf_score = silhouette_score(X, nmf_labels)

        # Kmeans Run section
        # Prepare dataset as plain text for kmeans.py stdin
        input_str = "\n".join(",".join(map(str, row)) for row in X)

        # Run kmeans.py with k as an argument
        cmd = [sys.executable, "kmeans.py", str(k)]
        proc = subprocess.run(cmd, input=input_str, capture_output=True, text=True)

        if proc.returncode != 0:
            error_exit()

        # Parse centroids from stdout
        lines = proc.stdout.strip().split("\n")
        kmeans_centroids = np.array([list(map(float, line.split(","))) for line in lines if line])

        kmeans_labels = get_labels_from_centroids(X, kmeans_centroids)
        kmeans_score = silhouette_score(X, kmeans_labels)

        # Output the results
        print(f"nmf: {nmf_score:.4f}")
        print(f"kmeans: {kmeans_score:.4f}")

    except Exception:
        error_exit()

if __name__ == "__main__":
    main()
