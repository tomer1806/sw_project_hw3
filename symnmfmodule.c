#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"
#include <stdlib.h> 

static void error_exit_py() {
    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
}

static double** allocate_matrix(int rows, int cols) {
    /* Allocate a 2D matrix (rows x cols) and initialize with zeros */ 
    double **matrix = (double **) malloc(rows * sizeof(double *));
    if (!matrix) {
        error_exit_py();
        return NULL;
    }
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double *) calloc(cols, sizeof(double));
        if (!matrix[i]) {
            error_exit_py();
            return NULL;
        }
    }
    return matrix;
}

static void free_matrix(double **matrix, int rows) {
    /* Frees the memory allocated for a 2D matrix */
    if (matrix != NULL) {
        for (int i = 0; i < rows; i++) {
            free(matrix[i]);
        }
        free(matrix);
    }
}

static double** py_list_to_c_matrix(PyObject* py_list, int *rows, int *cols) {
    /* Converts a Python list of lists to a C 2D matrix */
    if (!PyList_Check(py_list)) return NULL;
    *rows = PyList_Size(py_list);
    if (*rows == 0) {
        *cols = 0;
        return allocate_matrix(0,0);
    }

    PyObject *first_row = PyList_GetItem(py_list, 0);
    if (!PyList_Check(first_row)) return NULL;
    *cols = PyList_Size(first_row);
    
    double **matrix = allocate_matrix(*rows, *cols);
    if (!matrix) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for C matrix");
        return NULL;
    }
    
    for (int i = 0; i < *rows; i++) {
        PyObject *row_list = PyList_GetItem(py_list, i);
        for (int j = 0; j < *cols; j++) {
            PyObject *item = PyList_GetItem(row_list, j);
            matrix[i][j] = PyFloat_AsDouble(item);
        }
    }
    return matrix;
}

static PyObject* c_matrix_to_py_list(double** matrix, int rows, int cols) {
    /* Converts a C 2D matrix to a Python list of lists */
    PyObject *py_list = PyList_New(rows);
    if (!py_list) return NULL;

    for (int i = 0; i < rows; i++) {
        PyObject *row_list = PyList_New(cols);
        if (!row_list) return NULL;
        for (int j = 0; j < cols; j++) {
            PyObject *num = PyFloat_FromDouble(matrix[i][j]);
            PyList_SET_ITEM(row_list, j, num);
        }
        PyList_SET_ITEM(py_list, i, row_list);
    }
    return py_list;
}

static PyObject* sym_api(PyObject *self, PyObject *args) {
    /* Python API for the sym function */  
    PyObject *X_py;
    if (!PyArg_ParseTuple(args, "O", &X_py)) return NULL;

    int n, d;
    double **X = py_list_to_c_matrix(X_py, &n, &d);
    double **A = allocate_matrix(n, n);
    
    sym(X, A, n, d);
    
    PyObject* A_py = c_matrix_to_py_list(A, n, n);
    free_matrix(X, n);
    free_matrix(A, n);
    
    return A_py;
}

static PyObject* ddg_api(PyObject *self, PyObject *args) {
    /* Python API for the ddg function */
    PyObject *X_py;
    if (!PyArg_ParseTuple(args, "O", &X_py)) return NULL;

    int n, d;
    double **X = py_list_to_c_matrix(X_py, &n, &d);
    double **A = allocate_matrix(n, n);
    double **D = allocate_matrix(n, n);

    sym(X, A, n, d);
    ddg(A, D, n);
    
    PyObject* D_py = c_matrix_to_py_list(D, n, n);
    free_matrix(X, n);
    free_matrix(A, n);
    free_matrix(D, n);
    
    return D_py;
}

static PyObject* norm_api(PyObject *self, PyObject *args) {
    /* Python API for the norm function */
    PyObject *X_py;
    if (!PyArg_ParseTuple(args, "O", &X_py)) return NULL;

    int n, d;
    double **X = py_list_to_c_matrix(X_py, &n, &d);
    double **A = allocate_matrix(n, n);
    double **D = allocate_matrix(n, n);
    double **W = allocate_matrix(n, n);

    sym(X, A, n, d);
    ddg(A, D, n);
    norm(A, D, W, n);
    
    PyObject* W_py = c_matrix_to_py_list(W, n, n);
    free_matrix(X, n);
    free_matrix(A, n);
    free_matrix(D, n);
    free_matrix(W, n);

    return W_py;
}

static PyObject* symnmf_api(PyObject *self, PyObject *args) {
    /* Python API for the symnmf function */
    PyObject *W_py, *H_init_py;
    int n, k;
    if (!PyArg_ParseTuple(args, "OO", &H_init_py, &W_py)) return NULL;

    double **H_init = py_list_to_c_matrix(H_init_py, &n, &k);
    double **W = py_list_to_c_matrix(W_py, &n, &n);
    double **H_final = allocate_matrix(n, k);
    
    symnmf(W, H_init, H_final, n, k);

    PyObject* H_final_py = c_matrix_to_py_list(H_final, n, k);
    free_matrix(W, n);
    free_matrix(H_init, n);
    free_matrix(H_final, n);

    return H_final_py;
}

static PyMethodDef symnmf_methods[] = {
    /* Define the methods that can be called from Python */
    {"sym",    (PyCFunction)sym_api,    METH_VARARGS, "Calculate similarity matrix"},
    {"ddg",    (PyCFunction)ddg_api,    METH_VARARGS, "Calculate diagonal degree matrix"},
    {"norm",   (PyCFunction)norm_api,   METH_VARARGS, "Calculate normalized similarity matrix"},
    {"symnmf", (PyCFunction)symnmf_api, METH_VARARGS, "Perform full symNMF algorithm"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmf_module = {
    /* Define the module */ 
    PyModuleDef_HEAD_INIT,
    "symnmf", 
    "Python interface for the symNMF C library",
    -1,
    symnmf_methods
};

PyMODINIT_FUNC PyInit_symnmf(void) {
    /* Initialize the module */
    return PyModule_Create(&symnmf_module);
}
