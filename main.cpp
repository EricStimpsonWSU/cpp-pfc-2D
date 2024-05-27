#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <fstream>
#include <complex>
#include <vector>
#include <array>
#include <algorithm>
#include <fftw3.h>
#include <fpng.h>

using namespace std;

// PFC (Phase Field Crystal) model
// single-component 2D system
// 2D hexagonal lattice structure
// note: need to call forward FFT for psi-->psiq at the end of each time step (although psiq is known)
//       (or real data treatment for j=1 (qy=0) in fftw), for convergence at late time
//
// pseudo-specral method
// using a predictor-corrector scheme of Cross et al., Chaos 1994
// here we 1) do not fix the number of iterations
//         2) expand around small alpha_1 to get cf_1 and cf2_1 when alpha_1 -> 0
// with the transient time

#define INIT_HOMO // homogenious initial condition
// #define NOISE_DYNAMICS // dynamics with conserved Gaussian noise (nabla dot zeta, i.e. div[zeta])
#define SAVE_CONF
#define image_reverse // x,y reverse for image output (default)

const int LY = 128;
const int LX = 128; // Array dimensions

const int N = 2048;
const double PI = 3.141592653589793238460;
const double L = 2.0 * PI;
const double dx = L / N;

// define a structure for pfc model parameters
struct pfc_parms
{
    double q0;    // wave number
    double q02;   // wave number squared
    double qx0;   // wave number x
    double qy0;   // wave number y
    int n_dx;     // number of grid points per lattice period
    double m0_x;  // number of lattice periods along x
    double dx;    // grid spacing x
    double dy;    // grid spacing y
    double dt;    // time step
    double tmax;  // maximum time
    double tcp;   // time to checkpoint
    int nim;      // number of images
    double dt_t;  // transient time step
    double t_t;   // transient time
    int n_t;      // number of transient time steps
    int n_iter;   // number of iterations
    double tol;   // tolerance for convergence
    double err;   // error
    double eps;   // epsilon
    double n0;    // n0
    double b_l;   // b_l
    double b_s;   // b_s
    double g;     // g
    double scale; // scale for backward FFT
    int seed;     // seed for random number generator
};

// define a structure for pfc model checkpoint
struct pfc_checkpoint
{
    int n;                     // time step
    double t;                  // time
    double dt;                 // time step
    double *psi_1D;            // density
    double *phi2_1D;           // density^2
    fftw_complex *psiq_1D;     // density in Fourier space
    fftw_complex *d2nq_1D;     // second derivative of density in q space
    array<double, LY * LX> f;  // free energy
    array<double, LY * LX> mu; // chemical energy
    double fMean;
    double muMean;
    array<double, LX> qx;
    array<double, LY / 2 + 1> qy;
    array<array<double, LX>, LY / 2 + 1> q2, exp_1, cf_1, cf2_1;
    fftw_plan r2c;
    fftw_plan c2r;
    int nimages; // number of images
};

// define a structure for pfc temps (temporary variables)
struct pfc_temps
{
    array<double, LY * LX> nonlin1;
    array<fftw_complex, (LY / 2 + 1) * LX> nonlin1_q;
};

// prototypes
void test_pfc_2D();
void init_pfc_parms_thin_film(pfc_parms &parms);
void output_pfc_parms(pfc_parms &parms);
void init_pfc_checkpoint(pfc_parms &parms, pfc_checkpoint &checkpoint);
void output_pfc_checkpoint(pfc_checkpoint &checkpoint);
void free_pfc_checkpoint(pfc_checkpoint &checkpoint);
void psi2png(int l, int m, double *psi, string name);
void f_mu(pfc_parms &parms, pfc_checkpoint &checkpoint);
void nonlin1_calc(pfc_parms &parms, pfc_checkpoint &checkpoint, pfc_temps &temps);

int main(int argc, char const *argv[])
{
    cout << "Hello World!" << endl;

    // conduct tests
    test_pfc_2D();

    cout << "Goodbye World!" << endl;
    return 0;
}

// tests
void test_pfc_2D()
{

#ifdef NOISE_DYNAMICS
    complex **zeta_nq; // 2D Gaussian noise
#endif
    // parameters for film (unstrained, bulk)
    pfc_parms parms;
    init_pfc_parms_thin_film(parms);
    output_pfc_parms(parms);

    // initialize checkpoint
    pfc_checkpoint checkpoint;
    init_pfc_checkpoint(parms, checkpoint);
    output_pfc_checkpoint(checkpoint);
    psi2png(LY, LX, checkpoint.psi_1D, "test.png");
    f_mu(parms, checkpoint);

    // output checkpoint.fMean, checkpoint.muMean
    cout << "<f>: " << checkpoint.fMean << ", <mu>: " << checkpoint.muMean << endl;

    // initialize temps
    pfc_temps temps;

    // calculate non linear term
    nonlin1_calc(parms, checkpoint, temps);

    free_pfc_checkpoint(checkpoint);
}

void init_pfc_parms_thin_film(pfc_parms &parms)
{
    parms.q0 = 1;
    parms.q02 = 1;
    parms.qx0 = (sqrt(3.0) / 2) * parms.q0;
    parms.qy0 = parms.q0;
    parms.n_dx = 8;                                       // number of grid points per lattice period
    parms.m0_x = LX / parms.n_dx;                         // number of lattice periods along x
    parms.dx = 2 * PI / (parms.n_dx * parms.qy0);         // discretization of space
    parms.dy = parms.dx;                                  // discretization of space
    parms.dt = 0.5;                                       // time step (seconds)
    parms.tmax = 10000;                                   // maximum time (seconds)
    parms.tcp = 500;                                      // time to checkpoint (image save)
    parms.nim = static_cast<int>(parms.tmax / parms.tcp); // number of images
    parms.dt_t = 0.01;                                    // time step (seconds)
    parms.t_t = 1.0;                                      // transient time (seconds)
    parms.n_t = static_cast<int>(parms.t_t / parms.dt_t); // number of transient time steps
    parms.n_iter = 100;                                   // number of iterations
    parms.tol = 1.0e-3;                                   // tolerance for convergence
    parms.err = parms.tol / 10;                           // error
    parms.eps = 0.02;                                     // epsilon
    parms.n0 = -0.02;                                     // n0 (bulk)
    parms.b_s = 10.;                                      // b_s
    parms.b_l = (1 - parms.eps) * parms.b_s;              // b_l
    parms.g = 0.5 * sqrt(3 / parms.b_s);                  // g
    parms.scale = 1. / (LY * LX);                         // inverse FFT scale
    parms.seed = static_cast<int>(time(0));               // seed for random number generator
}

void output_pfc_parms(pfc_parms &parms)
{
    // output parameters to a file
    ofstream out("pfc_params.txt");
    out << setprecision(numeric_limits<double>::max_digits10);
    out << "q0 = " << parms.q0 << endl;
    out << "q02 = " << parms.q02 << endl;
    out << "qx0 = " << parms.qx0 << endl;
    out << "qy0 = " << parms.qy0 << endl;
    out << "n_dx = " << parms.n_dx << endl;
    out << "m0_x = " << parms.m0_x << endl;
    out << "dx = " << parms.dx << endl;
    out << "dy = " << parms.dy << endl;
    out << "dt = " << parms.dt << endl;
    out << "tmax = " << parms.tmax << endl;
    out << "tcp = " << parms.tcp << endl;
    out << "nim = " << parms.nim << endl;
    out << "dt_t = " << parms.dt_t << endl;
    out << "t_t = " << parms.t_t << endl;
    out << "n_t = " << parms.n_t << endl;
    out << "n_iter = " << parms.n_iter << endl;
    out << "tol = " << parms.tol << endl;
    out << "err = " << parms.err << endl;
    out << "eps = " << parms.eps << endl;
    out << "n0 = " << parms.n0 << endl;
    out << "b_s = " << parms.b_s << endl;
    out << "b_l = " << parms.b_l << endl;
    out << "g = " << parms.g << endl;
    out << "seed = " << parms.seed << endl;
    out.close();
}

void init_pfc_checkpoint(pfc_parms &parms, pfc_checkpoint &checkpoint)
{
    checkpoint.psi_1D = fftw_alloc_real(LY * LX);
    checkpoint.phi2_1D = fftw_alloc_real(LY * LX);
    checkpoint.psiq_1D = fftw_alloc_complex((LY / 2 + 1) * LX);
    checkpoint.d2nq_1D = fftw_alloc_complex((LY / 2 + 1) * LX);

    // Create the FFTW plan
    checkpoint.r2c = fftw_plan_dft_r2c_2d(LY, LX, checkpoint.psi_1D, checkpoint.psiq_1D, FFTW_MEASURE);
    checkpoint.c2r = fftw_plan_dft_c2r_2d(LY, LX, checkpoint.psiq_1D, checkpoint.psi_1D, FFTW_MEASURE);

    checkpoint.n = 0;
    checkpoint.t = 0.0;
    checkpoint.nimages = 0;

    // random number test
    srand(parms.seed); // randomimze from seed
    cout << "random double [between 0 and 1] = " << (double)rand() / RAND_MAX << endl;

    // Create a vector representations of psi and psiq
    // vector<double> psi(checkpoint.psi_1D, checkpoint.psi_1D + LY * LX);
    // vector<double> psiq(reinterpret_cast<double *>(checkpoint.psiq_1D), reinterpret_cast<double *>(checkpoint.psiq_1D) + LY * (LX / 2 + 1) * 2);

    // initialize psi with noise
    double n0 = -0.02; // n0_sol=-0.042253274 (from 1D ampl. eqs.); note: if setting n0=-0.04, all liquid
    double noise = 0.1;
    double noise0 = noise;
    for (int j = 0; j < LY; j++)
    {
        for (int i = 0; i < LX; i++)
        {
            if (abs(n0) < 1.0e-10)
            {
                // psi[j][i] = n0 + noise0 * ((double)rand() / RAND_MAX - 0.5);
                checkpoint.psi_1D[j * LX + i] = n0 + noise0 * ((double)rand() / RAND_MAX - 0.5);
            }
            else
            {
                // psi[j][i] = n0 + noise * ((double)rand() / RAND_MAX - 0.5);
                checkpoint.psi_1D[j * LX + i] = n0 + noise * ((double)rand() / RAND_MAX - 0.5);
            }
        }
    }

    // get initial psiq
    fftw_execute(checkpoint.r2c);
    checkpoint.psiq_1D[0][0] = n0 * LX * LY; // set the zero mode to n0 so that <psi>=n0
    checkpoint.psiq_1D[0][1] = 0.0;
    fftw_execute(checkpoint.c2r);
    for (int j = 0; j < LY; j++)
    {
        for (int i = 0; i < LX; i++)
        {
            // psi[j][i] *= scale;
            checkpoint.psi_1D[j * LX + i] *= parms.scale;
        }
    }

    // if transient time is not zero, use transient time step
    if (parms.t_t > 0.0)
    {
        checkpoint.dt = parms.dt_t;
    }

    // what do we think qx and qy are meant for?
    // initialize qx, qy
    for (int i = 0; i < LX; i++)
    {
        if (i < LX / 2 + 1)
        {
            checkpoint.qx[i] = 2.0 * PI * i / (parms.dx * LX);
        }
        else
        {
            checkpoint.qx[i] = 2.0 * PI * (i - LX) / (parms.dx * LX);
        }
    }
    for (int j = 0; j < LY / 2 + 1; j++)
    {
        checkpoint.qy[j] = 2.0 * PI * j / (parms.dy * LY);
    }

    // build q2, exp_1, cf_1, cf2_1
    for (int i = 0; i < LX; i++)
    {
        for (int j = 0; j < LY / 2 + 1; j++)
        {
            checkpoint.q2[j][i] = checkpoint.qx[i] * checkpoint.qx[i] + checkpoint.qy[j] * checkpoint.qy[j];
            auto alpha_1 = -checkpoint.q2[j][i] * (-parms.eps + (checkpoint.q2[j][i] - parms.q02) * (checkpoint.q2[j][i] - parms.q02));
            auto alpha_dt = alpha_1 * checkpoint.dt;
            checkpoint.exp_1[j][i] = exp(alpha_dt);
            if (abs(alpha_dt) < 2.0e-5)
            {
                checkpoint.cf_1[j][i] = checkpoint.dt * (1.0 + 0.5 * alpha_dt * (1.0 + alpha_dt / 3.0));
                checkpoint.cf2_1[j][i] = 0.5 * checkpoint.dt * (1.0 + alpha_dt * (1.0 + 0.250 * alpha_dt) / 3.0);
            }
            else
            {
                checkpoint.cf_1[j][i] = (checkpoint.exp_1[j][i] - 1) / alpha_1;
                checkpoint.cf2_1[j][i] = (checkpoint.exp_1[j][i] - (1 + alpha_dt)) / (alpha_1 * alpha_dt);
            }
        }
    }
}

void output_pfc_checkpoint(pfc_checkpoint &checkpoint)
{
    // write q2, exp_1, cf_1, cf2_1 to csv files
    ofstream out_q2("q2.csv");
    ofstream out_exp_1("exp_1.csv");
    ofstream out_cf_1("cf_1.csv");
    ofstream out_cf2_1("cf2_1.csv");
    for (int j = 0; j < LY / 2 + 1; j++)
    {
        for (int i = 0; i < LX; i++)
        {
            out_q2 << checkpoint.q2[j][i] << ",";
            out_exp_1 << checkpoint.exp_1[j][i] << ",";
            out_cf_1 << checkpoint.cf_1[j][i] << ",";
            out_cf2_1 << checkpoint.cf2_1[j][i] << ",";
        }
        out_q2 << endl;
        out_exp_1 << endl;
        out_cf_1 << endl;
        out_cf2_1 << endl;
    }
    out_q2.close();
    out_exp_1.close();
    out_cf_1.close();
    out_cf2_1.close();
}

void free_pfc_checkpoint(pfc_checkpoint &checkpoint)
{
    fftw_free(checkpoint.psi_1D);
    fftw_free(checkpoint.phi2_1D);
    fftw_free(checkpoint.psiq_1D);
    fftw_free(checkpoint.d2nq_1D);

    // destroy the plans
    fftw_destroy_plan(checkpoint.r2c);
    fftw_destroy_plan(checkpoint.c2r);
}

void psi2png(int l, int m, double *psi, string name)
{
    // create a PNG image
    vector<uint8_t> image(l * m * 3);

    // copy r1a to vector<double>
    vector<double> r1a(l * m);
    for (int i = 0; i < l; i++)
    {
        for (int j = 0; j < m; j++)
        {
            r1a[i * m + j] = psi[i * m + j];
        }
    }

    // get the min and max values of r1a
    double min = *min_element(r1a.begin(), r1a.end());
    double max = *max_element(r1a.begin(), r1a.end());

    // use max precision for doubles on output
    cout << setprecision(numeric_limits<double>::max_digits10);
    cout << "min = " << min << ", max = " << max << ", range = " << max - min << endl;

    // scale r1a to [0,1]
    for (int i = 0; i < l; i++)
    {
        for (int j = 0; j < m; j++)
        {
            if (max - min < 1.0e-10)
            {
                r1a[i * m + j] = 0.0;
            }
            else
            {
                r1a[i * m + j] = (r1a[i * m + j] - min) / (max - min);
            }
        }
    }

    // copy r1a to image
    for (int i = 0; i < l; i++)
    {
        for (int j = 0; j < m; j++)
        {
            int idx = 3 * (i * m + j);
            image[idx] = static_cast<uint8_t>(r1a[i * m + j] * 255);
            image[idx + 1] = static_cast<uint8_t>(r1a[i * m + j] * 255);
            image[idx + 2] = static_cast<uint8_t>(r1a[i * m + j] * 255);
        }
    }

    fpng::fpng_encode_image_to_file(name.c_str(), image.data(), l, m, 3, 0);

    // for (int i = 0; i < l; i++)
    // {
    //     for (int j = 0; j < m; j++)
    //     {
    //         int idx = 3 * (i * m + j);
    //         image[idx] = static_cast<uint8_t>(r1a[i * m + j] * 255);
    //         image[idx + 1] = static_cast<uint8_t>(r1a[i * m + j] * 255);
    //         image[idx + 2] = static_cast<uint8_t>(r1a[i * m + j] * 255);
    //     }
    // }
    // fpng::write(name.c_str(), l, m, r1a);
}

void f_mu(pfc_parms &parms, pfc_checkpoint &checkpoint)
{
    for (int j = 0; j < LY; j++)
    {
        for (int i = 0; i < LX; i++)
        {
            auto index = j * LX + i;
            checkpoint.phi2_1D[index] = checkpoint.psi_1D[index] * checkpoint.psi_1D[index];
            checkpoint.f[index] = (0.25 * checkpoint.phi2_1D[index] - 0.5 * parms.eps - parms.g * checkpoint.psi_1D[index]) * checkpoint.psi_1D[index];
            checkpoint.mu[index] = (checkpoint.psi_1D[index] - parms.g) * checkpoint.phi2_1D[index] - parms.eps * checkpoint.psi_1D[index];
        }
    }

    for (int j = 0; j < (LY / 2 + 1); j++)
    {
        for (int i = 0; i < LX; i++)
        {
            auto index = j * LX + i;
            checkpoint.d2nq_1D[index][0] = (parms.q02 - checkpoint.q2[j][i]) * (parms.q02 - checkpoint.q2[j][i]) * checkpoint.psiq_1D[index][0];
        }
    }
    fftw_execute_dft_c2r(checkpoint.c2r, checkpoint.d2nq_1D, checkpoint.phi2_1D);
    checkpoint.fMean = 0;
    checkpoint.muMean = 0;
    for (int j = 0; j < LY; j++)
    {
        for (int i = 0; i < LX; i++)
        {
            auto index = j * LX + i;
            checkpoint.phi2_1D[index] *= parms.scale;
            checkpoint.f[index] += 0.5 * checkpoint.phi2_1D[index] * checkpoint.psi_1D[index];
            checkpoint.mu[index] += checkpoint.phi2_1D[index];
            checkpoint.fMean += checkpoint.f[index] * parms.scale;
            checkpoint.muMean += checkpoint.mu[index] * parms.scale;
        }
    }
}

void nonlin1_calc(pfc_parms &parms, pfc_checkpoint &checkpoint, pfc_temps &temps)
{
    // calculate non linear term.
    for (int j = 0; j < LY; j++)
    {
        for (int i = 0; i < LX; i++)
        {
            auto index = j * LX + i;
            temps.nonlin1[index] = checkpoint.psi_1D[index] * checkpoint.psi_1D[index] * (checkpoint.psi_1D[index] - parms.g);
        }
    }

    // tramsform
    fftw_execute_dft_r2c(checkpoint.r2c, temps.nonlin1.data(), temps.nonlin1_q.data());
}