#ifndef SYMNMF_H
#define SYMNMF_H

/* Sym function to build the similarity Matrix A */
void sym(double **X, double **A, int n, int d);

/* ddg function to build diagonal degree matrix D based on similarity matrix A */
void ddg(double **A, double **D, int n);

/* norm function to build normalized similarity matrix W from A and D */
void norm(double **A, double **D, double **W, int n);

/* symnmf function to run SymNMF */
void symnmf(double **W, double **H_init, double **H_final, int n, int k);

#endif
