/*
This program serves as the testbed for various computational physics tools.
It solves the 1D heat equation using an explicit finite difference method stencil 

Written by Tim Linke as part of the MIT Plasma Science and Fusion Center Computational Physics School for Fusion Research
August 18 - 23, 2025
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double l2norm(const double * restrict T);

int frequency = 10; //total print count in array
int n = 100;

int main() {

    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    printf("Running on %d threads.\n", num_threads);

    double start = omp_get_wtime();

    // Material parameters
    int nsteps = 1e7;
    double alpha = 0.00000001;  // thermal diffusivity
    double length = 0.01;
    double dx = length / (n + 1);
    double dt = 0.5 / nsteps;  // time interval
    double r = alpha * dt / (dx*dx);
    if(r > 0.5) {
        printf("ERROR: simulation unstable. Ensure that r <= 0.5");
        return 0;
    }

    // printf("Temperature along ablator location i:\n");
    // for (int i = 0; i < n; i+=n/10) printf("%d ", i);
    // printf("\n");

    // Allocate arrays
    double *T     = malloc(sizeof(double) * n);
    double *T_tmp = malloc(sizeof(double) * n);
    double *tmp;

    // Initial condition: preheated 400 K
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        T[i] = 400.0;
        T_tmp[i] = 400.0;
    }

    // Parameters for a boundary laser pulse at x=0
    double laser_amplitude = 1e3;  // arbitrary choice
    double pulse_duration  = 0.2;   // fraction of total time

    double tic = omp_get_wtime();
    double time = 0.0;
    //#pragma omp target enter data map(to:T[0:n], T_tmp[0:n]) //this significantly improves memory performance
    for (int t = 0; t < nsteps; ++t) {
        time = t * dt;

        //#pragma omp target 
        //#pragma omp loop
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double left  = (i > 0) ? T[i-1] : T[0];
            double right = (i < n-1) ? T[i+1] : T[n-2];
            T_tmp[i] = T[i] + r * (left - 2*T[i] + right);
        }

        //Apply laser boundary at i=0 with simple flux injection
        if (time < pulse_duration) {
            T_tmp[0] += laser_amplitude * dt / dx;
        }

        // Swap pointers
        tmp = T;
        T = T_tmp;
        T_tmp = tmp;
        
        // if(nsteps % 10000 == 0) {
        //     for (int i = 0; i < n; i+=n/frequency) printf("%f ", T[i]);
        //     printf("\n");
        // }
    }   
    //#pragma omp target exit data map(from: T[0:n]) 

    double toc = omp_get_wtime();
    double stop = omp_get_wtime();
    double norm = l2norm(T);

    // for (int i = 0; i < n; i+=n/frequency) printf("%f ", T[i]);
    // printf("\n");
    printf("%lf  %f  %lf %lf\n", alpha, laser_amplitude, T[0], T[n-1]);

    printf("Results\n\n");
    //printf("Error (L2norm): %E\n", norm);
    printf("Solve time (s): %lf\n", toc - tic);
    printf("Total time (s): %lf\n", stop - start);

    free(T);
    free(T_tmp);
}


// Computes the L2-norm of the computed array to a standard solution
double l2norm(const double * restrict T) {

    // Sample solution - useful when trying different parallelization strategies
    double T_sol[10] = {20507.377292, 9430.552828, 1286.664601, 422.849656, 400.177644, 400.000428, 400.000000, 400.000000, 400.000000, 400.000000};
    double l2norm = 0.0;
    int index = 0;
    #pragma omp parallel for
    for (int i = 0; i < n; i+=n/frequency) {
        l2norm += (T[i] - T_sol[index]) * (T[i] - T_sol[index]);
        index++;
    }

    return sqrt(l2norm);
}


