import sys
import numpy as np
import subprocess
from sklearn.metrics import silhouette_score
import symnmf as symnmf_c
import os

def error_exit():
    #Prints the standard error message and quits immediately
    print("An Error Has Occurred")
    sys.exit(1)

def get_labels_from_centroids(X, centroids):
    #Assigns each data point to its closest centroid.
    distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)

def run_symnmf_analysis(X, k):
    #Runs the SymNMF clustering algorithm and gets its silhouette score.
    N = X.shape[0]
    W = symnmf_c.norm(X.tolist())
    if W is None: error_exit()

    # Create a random initial H matrix for the optimization
    np.random.seed(1234)
    m = np.mean(np.array(W))
    H_init = np.random.uniform(0, 2 * np.sqrt(m / k), size=(N, k))
    
    H_final = symnmf_c.symnmf(H_init.tolist(), W)
    if H_final is None: error_exit()
    
    # Assign a cluster based on the highest value in each row of H
    nmf_labels = np.argmax(np.array(H_final), axis=1)
    return silhouette_score(X, nmf_labels)

def run_kmeans_analysis(X, k):
    #Runs the K-means script as a subprocess and gets its silhouette score.
    #Passes the data to the 'kmeans.py' script and recieves final cluster for the final score
 
    # Prepare the dataset as a string to be passed to the kmeans script
    input_str = "\n".join(",".join(map(str, row)) for row in X)
    
    # Find the kmeans.py script, assuming it's in the same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kmeans_path = os.path.join(script_dir, 'kmeans.py')
    
    cmd = [sys.executable, kmeans_path, str(k)]
    proc = subprocess.run(cmd, input=input_str, capture_output=True, text=True)

    if proc.returncode != 0: error_exit()

    #Parse the centroids from the script's output
    lines = proc.stdout.strip().split("\n")
    centroids = np.array([list(map(float, line.split(","))) for line in lines if line])
    
    kmeans_labels = get_labels_from_centroids(X, centroids)
    return silhouette_score(X, kmeans_labels)

def main():
    #Main of comparing SymNMF and K-means.
    #loads data, runs both clustering algorithms, and prints silhouette scores
    try: 
        if len(sys.argv) != 3: error_exit()
        k = int(sys.argv[1])
        file_name = sys.argv[2]
        X = np.loadtxt(file_name, delimiter=',', ndmin = 2)
        if k <= 1 or k >= X.shape[0]: error_exit()
        nmf_score = run_symnmf_analysis(X, k)
        kmeans_score = run_kmeans_analysis(X, k)
        print(f"nmf: {nmf_score:.4f}")
        print(f"kmeans: {kmeans_score:.4f}")

    except Exception:
        error_exit()

if __name__ == "__main__":
    main()