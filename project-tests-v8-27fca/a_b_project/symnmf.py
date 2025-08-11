import sys
import numpy as np
import symnmf as symnmf_c

def error_exit(): # Exit the program with an error message
    print("An Error Has Occurred")
    sys.exit(1)

def print_matrix_helper(m): # Helper function to print a matrix if it is not None
    for row in m:
        print(','.join(['%.4f' % val for val in row]))

def main():
    try:
        if len(sys.argv) != 4:
            error_exit()
        k = int(sys.argv[1])
        goal = sys.argv[2]
        file_name = sys.argv[3]
        
        # Read data from file
        try:
            X = np.loadtxt(file_name, delimiter=',')
        except Exception: # If file reading fails, exit with error
            error_exit()
            
        N = X.shape[0]
        if k >= N: # Checks if k is less than amount of vectors
            error_exit()
        # Convert numpy array to list of lists for C module
        X_list = X.tolist()

        if goal == 'sym': # Call C implementation for sym()
            result_matrix = symnmf_c.sym(X_list)
        elif goal == 'ddg': # Call C implementation for ddg()
            result_matrix = symnmf_c.ddg(X_list)
        elif goal == 'norm': # Call C implementation for norm()
            result_matrix = symnmf_c.norm(X_list)
        elif goal == 'symnmf': # Call C implementation for symnmf()
            # Initialize W
            W = symnmf_c.norm(X_list)
            W_np = np.array(W)
            # Initialize H
            np.random.seed(1234)
            m = np.mean(W_np)
            H_init = np.random.uniform(0, 2 * np.sqrt(m / k), size=(N, k))
            H_init_list = H_init.tolist()
            
            # Call C implementation for optimization
            result_matrix = symnmf_c.symnmf(H_init_list, W)
        else:
            error_exit()
        # If result_matrix is None, exit with error
        if result_matrix is None:
            error_exit()
        print_matrix_helper(result_matrix)

    except (ValueError, IndexError):
        error_exit()

if __name__ == "__main__":
    main()