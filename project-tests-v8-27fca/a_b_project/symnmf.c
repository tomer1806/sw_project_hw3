#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

/* Initialize constants */
#define MAX_ITER 300
#define EPSILON 1e-4
#define BETA 0.5


/* Prints an error message and terminates the program */
static void error_exit() {
    printf("An Error Has Occurred\n");
    exit(1);
}

/**
 * Allocates a 2D matrix (rows x cols) and initializes with zeros.
 * Returns A pointer to the allocated matrix.
 */
static double** allocate_matrix(int rows, int cols) {
    double **matrix = (double **) malloc(rows * sizeof(double *));
    int i;
    if (!matrix) error_exit();
    for (i = 0; i < rows; i++) {
        matrix[i] = (double *) calloc(cols, sizeof(double));
        if (!matrix[i]) error_exit();
    }
    return matrix;
}

/* Frees the memory allocated for a 2D matrix */
static void free_matrix(double **matrix, int rows) {
    if (matrix != NULL) {
        int i;
        for (i = 0; i < rows; i++) {
            free(matrix[i]);
        }
        free(matrix);
    }
}

/* Prints a 2D matrix to stdout with 4-decimal precision */
static void print_matrix_helper(double **matrix, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%.4f", matrix[i][j]);
            if (j < cols - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

/**
 * Calculates the squared difference between two matrices.
 * Returns The squared Frobenius norm
 */
static double two_matrices_diff(double **A, double **B, int n, int k) {
    int i, j;
    double sum = 0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            sum += pow(A[i][j] - B[i][j], 2);
        }
    }
    return sum;
}

/**
 * Multiplies two matrices A (r1 x c1) and B (c1 x c2).
 * Returns The resulting matrix C (r1 x c2).
 */
static double** matrix_multiply(double **A, double **B, int r1, int c1, int c2) {
    double **C = allocate_matrix(r1, c2);
    int i, j, l;
    for (i = 0; i < r1; i++) {
        for (j = 0; j < c2; j++) {
            for (l = 0; l < c1; l++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
    return C;
}

/**
 * Transposes a matrix and Returns the transposed matrix
 */
static double** transpose_matrix(double **A, int rows, int cols) {
    double **T = allocate_matrix(cols, rows);
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

/**
 * Finds the Vector dimension (d) of the data points in a file
 */
static void count_data_dimension(const char *file_name, int *d_out) {
    FILE *fp = fopen(file_name, "r");
    char *line = NULL, *token;
    size_t len = 0;
    int d = 0;

    if (fp == NULL) error_exit();
    if (getline(&line, &len, fp) != -1) {
        token = strtok(line, ",");
        while (token != NULL) {
            d++;
            token = strtok(NULL, ",");
        }
    }
    *d_out = d;
    fclose(fp);
    if (line) free(line);
}

/**
 * Reads data points from a file into a dynamically allocated matrix.
 * Returns A 2D matrix containing the data points.
 */
static double** read_data_to_matrix(const char *file_name, int d, int *n_out) {
    FILE *fp = fopen(file_name, "r");
    char *line = NULL, *token;
    size_t len = 0;
    int n = 0, cap = 10;
    int i;
    double **data;
    
    if (fp == NULL) error_exit();
    data = (double**) malloc(cap * sizeof(double*));
    if(!data) error_exit();

    while (getline(&line, &len, fp) != -1) {
        if (n >= cap) {
            cap *= 2;
            data = (double**) realloc(data, cap * sizeof(double*));
            if(!data) error_exit();
        }
        data[n] = (double*) malloc(d * sizeof(double));
        if(!data[n]) error_exit();
        
        token = strtok(line, ",");
        for(i = 0; token != NULL && i < d; i++){
            data[n][i] = atof(token);
            token = strtok(NULL, ",");
        }
        n++;
    }
    fclose(fp);
    if (line) free(line);
    *n_out = n;
    return data;
}

/* Computes the symmetric similarity matrix A from data points X */
void sym(double **X, double **A, int n, int d) {
    int i, j, l;
    double dist_sq;
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            if (i == j) {
                A[i][j] = 0;
                continue;
            }
            dist_sq = 0;
            for (l = 0; l < d; l++) {
                dist_sq += pow(X[i][l] - X[j][l], 2);
            }
            A[i][j] = exp(-dist_sq / 2.0);
            A[j][i] = A[i][j];
        }
    }
}
/* Computes the Diagonal degree matrix */
void ddg(double **A, double **D, int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            D[i][j] = 0;
        }
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            D[i][i] += A[i][j];
        }
    }
}
/* Computes the normalized similarity matrix */
void norm(double **A, double **D, double **W, int n) {
    int i, j;
    double d_inv_sqrt;
    for(i=0; i<n; i++){
        if(D[i][i] == 0) error_exit();
        d_inv_sqrt = 1.0 / sqrt(D[i][i]);
        D[i][i] = d_inv_sqrt;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            W[i][j] = D[i][i] * A[i][j] * D[j][j];
        }
    }
}

/* Computes the Symmetric Non-negative Matrix Factorization */
void symnmf(double **W, double **H, double **H_final, int n, int k) {
    int iter = 0, i, j;
    double **WH, **HT, **HHT, **HHTH, **H_next;
    int converged = 0;
    
    H_next = allocate_matrix(n, k);

    for(iter = 0; iter < MAX_ITER; iter++) {
        /* Perform matrix calculations for the update rule */
        WH = matrix_multiply(W, H, n, n, k);
        HT = transpose_matrix(H, n, k);
        HHT = matrix_multiply(H, HT, n, k, n);
        HHTH = matrix_multiply(HHT, H, n, n, k);

        /* Apply the update rule to calculate H_next */
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                if(HHTH[i][j] == 0) error_exit();
                H_next[i][j] = H[i][j] * (1 - BETA + BETA * (WH[i][j] / HHTH[i][j]));
            }
        }

        /* Check for convergence before updating H */
        if (two_matrices_diff(H_next, H, n, k) < EPSILON) {
            converged = 1;
        }

        /* ALWAYS update H to the newest matrix (H_next) */
        for (i = 0; i < n; i++) {
            memcpy(H[i], H_next[i], k * sizeof(double));
        }
        
        /* Free intermediate matrices */
        free_matrix(WH, n); 
        free_matrix(HT, k); 
        free_matrix(HHT, n); 
        free_matrix(HHTH, n);
        
        /* Break the loop AFTER the final update if convergence was met */
        if (converged) {
            break;
        }
    }

    /* Now, H correctly holds the final converged matrix. Copy it to H_final. */
    for (i = 0; i < n; i++) {
        memcpy(H_final[i], H[i], k * sizeof(double));
    }
    
    free_matrix(H_next, n);
}
/* Main function to handle command line arguments and call appropriate functions */
int main(int argc, char *argv[]) {
    char *goal, *file_name;
    int n = 0, d = 0;
    double **X, **A, **D, **W;

    if (argc != 3) error_exit();

    goal = argv[1];
    file_name = argv[2];

    count_data_dimension(file_name, &d);
    if(d == 0) error_exit();
    X = read_data_to_matrix(file_name, d, &n);
    if(n == 0) error_exit();

    A = allocate_matrix(n, n);
    sym(X, A, n, d);

    if (strcmp(goal, "sym") == 0) {
        print_matrix_helper(A, n, n);
    } else if (strcmp(goal, "ddg") == 0) {
        D = allocate_matrix(n, n);
        ddg(A, D, n);
        print_matrix_helper(D, n, n);
        free_matrix(D, n);
    } else if (strcmp(goal, "norm") == 0) {
        D = allocate_matrix(n, n);
        W = allocate_matrix(n, n);
        ddg(A, D, n);
        norm(A, D, W, n);
        print_matrix_helper(W, n, n);
        free_matrix(D, n);
        free_matrix(W, n);
    } else {
        error_exit();
    }
    
    free_matrix(X, n);
    free_matrix(A, n);
    
    return 0;
}