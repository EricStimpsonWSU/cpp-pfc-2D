#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <fstream>
#include <complex>
#include <vector>
#include <array>
#include <algorithm>
#include <filesystem>
#include <cstdio>
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

// Define only one of the initialization methods
#ifdef DEBUG
#define INIT_FILE // initialize using 0.psi.dat file
#else
#define INIT_HOMO // homogenious initial condition
#endif

// Define as many of the validates as desired
#define VALID_TOL 1.e-9

// #define VALID_PSIQ_0

// #define VALID_Q2_0
// #define VALID_EXP_0
// #define VALID_CF_1_0
// #define VALID_CF2_1_0

// #define VALID_PSI2_0
// #define VALID_F_0
// #define VALID_MU_0
// #define VALID_D2NQ2_0
// #define VALID_PSI2_POSTC2R_0
// #define VALID_F_POSTC2R_0
// #define VALID_MU_POSTC2R_0

// #define VALID_NONLIN1_1
// #define VALID_NONLIN1_Q_1
// #define VALID_SIGN_PSIQ_1
// #define VALID_PSI0Q_PRECONJ_1
// #define VALID_PSI0Q_1
// #define VALID_PSI_1
// #define VALID_PSIQ_1
// #define VALID_PSI0Q_1_1
// #define VALID_NONLIN1_Q_1_2
// #define VALID_NONLIN1_Q_1_3
// #define VALID_PSIQ_1_2
// #define VALID_PSIQ_1_3
// #define VALID_PSI_1_2
// #define VALID_PSI_1_3

// #define NOISE_DYNAMICS // dynamics with conserved Gaussian noise (nabla dot zeta, i.e. div[zeta])
#define SAVE_CONF
#define image_reverse // x,y reverse for image output (default)

const int LX = 128; // Array dimensions
const int LY = 128;

const int N = 2048;
const double PI = 3.141592653589793238460;
const double L = 2.0 * PI;
const double dx = L / N;

// define a structure for pfc model parameters
struct pfc_parms
{
    double q0;                          // wave number
    double q02;                         // wave number squared
    double qx0;                         // wave number x
    double qy0;                         // wave number y
    int n_dx;                           // number of grid points per lattice period
    double m0_x;                        // number of lattice periods along x
    double dx;                          // grid spacing x
    double dy;                          // grid spacing y
    double transient_time;              // transient time
    double transient_dt;                // transient time step
    int transient_time_steps;           // number of transient time steps
    double total_time;                  // total run rime
    double dt;                          // time step (after transient time)
    int total_time_steps;               // number of time steps (transient and after transient)
    double checkpoint_time;             // time to checkpoint
    int checkpoint_time_steps;          // stpes per checkpoint
    int nim;                            // number of images
    int predictor_corrector_iterations; // number of iterations
    double tol;                         // tolerance for convergence
    double err;                         // error
    double eps;                         // epsilon
    double n0;                          // n0
    double b_l;                         // b_l
    double b_s;                         // b_s
    double g;                           // g
    double scale;                       // scale for backward FFT
    int seed;                           // seed for random number generator
};

// define a structure for pfc model checkpoint
struct pfc_checkpoint
{
    int n;                          // time step
    double time;                    // time
    double dt;                      // time step
    vector<double> psi, psi2;       // density, density^2
    vector<array<double, 2>> psi_q; // density in Fourier space
    vector<array<double, 2>> d2n_q; // second derivative of density in q space
    vector<double> f;               // free energy
    vector<double> mu;              // chemical energy
    double fMean;                   // mean free energy
    double muMean;                  // mean chemical energy
    vector<double> qx;
    vector<double> qy;
    vector<double> q2;
    vector<double> exp_1;
    vector<double> cf_1;
    vector<double> cf2_1;
    fftw_plan r2c;
    fftw_plan c2r;
    int nimages; // number of images
};

// define a structure for pfc temps (temporary variables)
struct pfc_temps
{
    vector<double> nonlin1;
    vector<array<double, 2>> nonlin1_q;
    vector<double> psi0;
    vector<array<double, 2>> psi0_q;
    vector<array<double, 2>> psi0_tmp_q;
    vector<array<double, 2>> psi0_sign_q;
};

// prototypes
// tests
void test_pfc_2D();

// PFC initialization / cleanup
void init_pfc_parms_thin_film(pfc_parms &parms);
void output_pfc_parms(pfc_parms &parms);
void init_pfc_checkpoint(pfc_parms &parms, pfc_checkpoint &checkpoint);
void output_pfc_checkpoint(pfc_checkpoint &checkpoint);
void free_pfc_checkpoint(pfc_checkpoint &checkpoint);
void init_pfc_temps(pfc_parms &parms, pfc_checkpoint &checkpoint, pfc_temps &temps);

// PFC algorithms
void f_mu(pfc_parms &parms, pfc_checkpoint &checkpoint);
void sheq(pfc_parms &parms, pfc_checkpoint &checkpoint, pfc_temps &temps);
void nonlin1_calc(pfc_parms &parms, pfc_checkpoint &checkpoint, pfc_temps &temps);

// File outputs
void delete_files();
void write_f_mu(pfc_checkpoint &checkpoint);
void psi2png(int l, int m, double *psi, string name);

// Validatations
void validate_real(const vector<double> &checkArray, const string &filename, int LX, int LY, double tol);
void validate_complex(const vector<array<double, 2>> &checkArray, const string &filename, int LX, int LY, double tol);

// main
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
    // delete files
    delete_files();

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

    psi2png(LY, LX, checkpoint.psi.data(), "test.png");
    f_mu(parms, checkpoint);

    // write fMean, muMean to file
    write_f_mu(checkpoint);

    // initialize temps
    pfc_temps temps;
    init_pfc_temps(parms, checkpoint, temps);

    // PFC time loop (running)
    while (checkpoint.n < parms.total_time_steps)
    {
        checkpoint.n++;
        // if time is less than transient time, use transient time step, otherwise standard time step
        if (checkpoint.n < parms.transient_time_steps)
        {
            checkpoint.dt = parms.transient_dt;
            checkpoint.time = checkpoint.n * parms.transient_dt;
        }
        else
        {
            checkpoint.dt = parms.dt;
            checkpoint.time = (checkpoint.n - parms.transient_time_steps) * parms.dt + parms.transient_time;

            if (checkpoint.n == parms.transient_time_steps)
            {
                for (int index = 0; index < LX * (LY / 2 + 1); index++)
                {
                    auto alpha_1 = -checkpoint.q2[index] * (-parms.eps + (checkpoint.q2[index] - parms.q02) * (checkpoint.q2[index] - parms.q02));
                    auto alpha_dt = alpha_1 * checkpoint.dt;
                    checkpoint.exp_1[index] = exp(alpha_dt);
                    if (abs(alpha_dt) < 2.0e-5)
                    {
                        checkpoint.cf_1[index] = checkpoint.dt * (1.0 + 0.5 * alpha_dt * (1.0 + alpha_dt / 3.0));
                        checkpoint.cf2_1[index] = 0.5 * checkpoint.dt * (1.0 + alpha_dt * (1.0 + 0.250 * alpha_dt) / 3.0);
                    }
                    else
                    {
                        checkpoint.cf_1[index] = (checkpoint.exp_1[index] - 1) / alpha_1;
                        checkpoint.cf2_1[index] = (checkpoint.exp_1[index] - (1 + alpha_dt)) / (alpha_1 * alpha_dt);
                    }
                }
            }
        }
        sheq(parms, checkpoint, temps);

        // calculate free energy
        if (checkpoint.n == parms.transient_time_steps ||
            ((checkpoint.n > parms.transient_time_steps) && ((checkpoint.n + static_cast<int>(parms.transient_time / parms.dt)) % parms.checkpoint_time_steps == 0)))
        {
            f_mu(parms, checkpoint);
            write_f_mu(checkpoint);
            psi2png(LY, LX, checkpoint.psi.data(), "test1.png");
        }
    }

    free_pfc_checkpoint(checkpoint);
}

// PFC initialization / cleanup
void init_pfc_parms_thin_film(pfc_parms &parms)
{
    parms.q0 = 1;
    parms.q02 = 1;
    parms.qx0 = (sqrt(3.0) / 2) * parms.q0;
    parms.qy0 = parms.q0;
    parms.n_dx = 8;                                                                                  // number of grid points per lattice period
    parms.m0_x = LX / parms.n_dx;                                                                    // number of lattice periods along x
    parms.dx = 2 * PI / (parms.n_dx * parms.qy0);                                                    // discretization of space
    parms.dy = parms.dx;                                                                             // discretization of space
    parms.transient_time = 1.;                                                                       // transient time (seconds)
    parms.transient_dt = 0.01;                                                                       // time step (seconds)
    parms.transient_time_steps = static_cast<int>(round(parms.transient_time / parms.transient_dt)); // number of transient time steps
    parms.total_time = 10000.;                                                                       // total run time
    parms.dt = 0.5;                                                                                  // time step (seconds)
    parms.total_time_steps = parms.transient_time_steps +
                             static_cast<int>(round((parms.total_time - parms.transient_time) / parms.dt)); // total time steps
    parms.checkpoint_time = 50.;                                                                            // time to checkpoint (image save)
    parms.checkpoint_time_steps = static_cast<int>(round(parms.checkpoint_time / parms.dt));                // time steps between checkpoints
    parms.nim = static_cast<int>(parms.total_time / parms.checkpoint_time);                                 // number of images
    parms.predictor_corrector_iterations = 100;                                                             // number of iterations
    parms.tol = 1.0e-3;                                                                                     // tolerance for convergence
    parms.err = parms.tol / 10;                                                                             // error
    parms.eps = 0.02;                                                                                       // epsilon
    parms.n0 = -0.02;                                                                                       // n0 (bulk)
    parms.b_s = 10.;                                                                                        // b_s
    parms.b_l = (1 - parms.eps) * parms.b_s;                                                                // b_l
    parms.g = 0.5 * sqrt(3 / parms.b_s);                                                                    // g
    parms.scale = 1. / (LX * LY);                                                                           // inverse FFT scale
    parms.seed = static_cast<int>(time(0));                                                                 // seed for random number generator
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
    out << "transient_dt = " << parms.transient_dt << endl;
    out << "transient_time = " << parms.transient_time << endl;
    out << "transient_time_steps = " << parms.transient_time_steps << endl;
    out << "total_time = " << parms.total_time << endl;
    out << "checkpoint_time = " << parms.checkpoint_time << endl;
    out << "nim = " << parms.nim << endl;
    out << "predictor_corrector_iterations = " << parms.predictor_corrector_iterations << endl;
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
    checkpoint.psi.resize(LX * LY);
    checkpoint.psi2.resize(LX * LY);
    checkpoint.psi_q.resize((LY / 2 + 1) * LX);
    checkpoint.d2n_q.resize((LY / 2 + 1) * LX);
    checkpoint.f.resize(LX * LY);
    checkpoint.mu.resize(LX * LY);
    checkpoint.qx.resize(LX);
    checkpoint.qy.resize(LY / 2 + 1);
    checkpoint.q2.resize((LY / 2 + 1) * LX);
    checkpoint.exp_1.resize((LY / 2 + 1) * LX);
    checkpoint.cf_1.resize((LY / 2 + 1) * LX);
    checkpoint.cf2_1.resize((LY / 2 + 1) * LX);

    // Create the FFTW plan
    checkpoint.r2c = fftw_plan_dft_r2c_2d(LX, LY, checkpoint.psi.data(), reinterpret_cast<fftw_complex *>(checkpoint.psi_q.data()), FFTW_MEASURE);
    checkpoint.c2r = fftw_plan_dft_c2r_2d(LX, LY, reinterpret_cast<fftw_complex *>(checkpoint.psi_q.data()), checkpoint.psi.data(), FFTW_MEASURE);

    checkpoint.n = 0;
    checkpoint.time = 0.0;
    checkpoint.nimages = 0;

    // random number test
    srand(parms.seed); // randomimze from seed
    cout << "random double [between 0 and 1] = " << (double)rand() / RAND_MAX << endl;

#ifdef INIT_HOMO
    // initialize psi with noise
    double n0 = -0.02; // n0_sol=-0.042253274 (from 1D ampl. eqs.); note: if setting n0=-0.04, all liquid
    double noise = 0.1;
    double noise0 = noise;
    for (int index = 0; index < LX * LY; index++)
    {
        if (abs(n0) < 1.0e-10)
        {
            // psi[index] = n0 + noise0 * ((double)rand() / RAND_MAX - 0.5);
            checkpoint.psi[index] = n0 + noise0 * ((double)rand() / RAND_MAX - 0.5);
        }
        else
        {
            // psi[index] = n0 + noise * ((double)rand() / RAND_MAX - 0.5);
            checkpoint.psi[index] = n0 + noise * ((double)rand() / RAND_MAX - 0.5);
        }
    }

    // get initial psiq
    fftw_execute(checkpoint.r2c);
    checkpoint.psi_q[0][0] = n0 * LX * LY; // set the zero mode to n0 so that <psi>=n0
    checkpoint.psi_q[0][1] = 0.0;
    // c2r overwrites input
    fftw_execute(checkpoint.c2r);

    for (int index = 0; index < LX * LY; index++)
    {
        checkpoint.psi[index] *= parms.scale;
    }
#elif defined(INIT_FILE)
    // initialize psi from "0.psi.dat"
    cout << "initialize psi from '0.psi.dat'" << endl;
    ifstream in("0.psi.dat");
    for (int index = 0; index < LX * LY; index++)
    {
        in >> checkpoint.psi[index];
    }
#endif
    fftw_execute_dft_r2c(checkpoint.r2c, checkpoint.psi.data(), reinterpret_cast<fftw_complex *>(checkpoint.psi_q.data()));

#ifdef VALID_PSIQ_0
    validate_complex(checkpoint.psi_q, "0.psiq.dat", LX, LY / 2 + 1, VALID_TOL);
#endif

    // if transient time is not zero, use transient time step
    if (parms.transient_time > 0.0)
    {
        checkpoint.dt = parms.transient_dt;
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
            auto index = i * (LY / 2 + 1) + j;
            checkpoint.q2[index] = checkpoint.qx[i] * checkpoint.qx[i] + checkpoint.qy[j] * checkpoint.qy[j];
            auto alpha_1 = -checkpoint.q2[index] * (-parms.eps + (checkpoint.q2[index] - parms.q02) * (checkpoint.q2[index] - parms.q02));
            auto alpha_dt = alpha_1 * checkpoint.dt;
            checkpoint.exp_1[index] = exp(alpha_dt);
            if (abs(alpha_dt) < 2.0e-5)
            {
                checkpoint.cf_1[index] = checkpoint.dt * (1.0 + 0.5 * alpha_dt * (1.0 + alpha_dt / 3.0));
                checkpoint.cf2_1[index] = 0.5 * checkpoint.dt * (1.0 + alpha_dt * (1.0 + 0.250 * alpha_dt) / 3.0);
            }
            else
            {
                checkpoint.cf_1[index] = (checkpoint.exp_1[index] - 1) / alpha_1;
                checkpoint.cf2_1[index] = (checkpoint.exp_1[index] - (1 + alpha_dt)) / (alpha_1 * alpha_dt);
            }
        }
    }

#ifdef VALID_Q2_0
    validate_real(checkpoint.q2, "0.q2.dat", LX, LY / 2 + 1, VALID_TOL);
#endif
#ifdef VALID_EXP_0
    validate_real(checkpoint.exp_1, "0.exp_1.dat", LX, LY / 2 + 1, VALID_TOL);
#endif
#ifdef VALID_CF_1_0
    validate_real(checkpoint.cf_1, "0.cf_1.dat", LX, LY / 2 + 1, VALID_TOL);
#endif
#ifdef VALID_CF2_1_0
    validate_real(checkpoint.cf2_1, "0.cf2_1.dat", LX, LY / 2 + 1, VALID_TOL);
#endif
}

void output_pfc_checkpoint(pfc_checkpoint &checkpoint)
{
    // write q2, exp_1, cf_1, cf2_1 to csv files
    ofstream out_q2("q2.csv");
    ofstream out_exp_1("exp_1.csv");
    ofstream out_cf_1("cf_1.csv");
    ofstream out_cf2_1("cf2_1.csv");
    for (int i = 0; i < LX; i++)
    {
        for (int j = 0; j < LY / 2 + 1; j++)
        {
            auto index = i * (LY / 2 + 1) + j;
            out_q2 << checkpoint.q2[index] << ",";
            out_exp_1 << checkpoint.exp_1[index] << ",";
            out_cf_1 << checkpoint.cf_1[index] << ",";
            out_cf2_1 << checkpoint.cf2_1[index] << ",";
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
    // destroy the plans
    fftw_destroy_plan(checkpoint.r2c);
    fftw_destroy_plan(checkpoint.c2r);
}

void init_pfc_temps(pfc_parms &parms, pfc_checkpoint &checkpoint, pfc_temps &temps)
{
    temps.nonlin1.resize(LX * LY);
    temps.nonlin1_q.resize((LY / 2 + 1) * LX);
    temps.psi0.resize(LX * LY);
    temps.psi0_q.resize((LY / 2 + 1) * LX);
    temps.psi0_tmp_q.resize((LY / 2 + 1) * LX);
    temps.psi0_sign_q.resize((LY / 2 + 1) * LX);
}

// PFC algorithms
void f_mu(pfc_parms &parms, pfc_checkpoint &checkpoint)
{
    for (int index = 0; index < LX * LY; index++)
    {
        checkpoint.psi2[index] = checkpoint.psi[index] * checkpoint.psi[index];
        checkpoint.f[index] = (0.25 * checkpoint.psi2[index] - 0.5 * parms.eps - parms.g * checkpoint.psi[index] / 3) * checkpoint.psi2[index];
        checkpoint.mu[index] = (checkpoint.psi[index] - parms.g) * checkpoint.psi2[index] - parms.eps * checkpoint.psi[index];
    }

#ifdef VALID_PSI2_0
    validate_real(checkpoint.psi2, "0.phi2.dat", LX, LY, VALID_TOL);
#endif
#ifdef VALID_F_0
    validate_real(checkpoint.f, "0.f.dat", LX, LY, VALID_TOL);
#endif
#ifdef VALID_MU_0
    validate_real(checkpoint.mu, "0.mu.dat", LX, LY, VALID_TOL);
#endif

    for (int index = 0; index < LX * (LY / 2 + 1); index++)
    {
        checkpoint.d2n_q[index][0] = (parms.q02 - checkpoint.q2[index]) * (parms.q02 - checkpoint.q2[index]) * checkpoint.psi_q[index][0];
        checkpoint.d2n_q[index][1] = (parms.q02 - checkpoint.q2[index]) * (parms.q02 - checkpoint.q2[index]) * checkpoint.psi_q[index][1];
    }

#ifdef VALID_D2NQ2_0
    validate_complex(checkpoint.d2n_q, "0.d2nq2.dat", LX, LY / 2 + 1, VALID_TOL);
#endif

    // back FFT (overwrites data in input)
    // cout << "call dft c2r in sheq" << endl;
    fftw_execute_dft_c2r(checkpoint.c2r, reinterpret_cast<fftw_complex *>(checkpoint.d2n_q.data()), checkpoint.psi2.data());
    checkpoint.fMean = 0;
    checkpoint.muMean = 0;

    for (int index = 0; index < LX * LY; index++)
    {
        checkpoint.psi2[index] *= parms.scale;
        checkpoint.f[index] += 0.5 * checkpoint.psi2[index] * checkpoint.psi[index];
        checkpoint.mu[index] += checkpoint.psi2[index];
        checkpoint.fMean += checkpoint.f[index] * parms.scale;
        checkpoint.muMean += checkpoint.mu[index] * parms.scale;
    }

#ifdef VALID_PSI2_POSTC2R_0
    validate_real(checkpoint.psi2, "0.phi2.postc2r.dat", LX, LY, VALID_TOL);
#endif
#ifdef VALID_F_POSTC2R_0
    validate_real(checkpoint.f, "0.f.postc2r.dat", LX, LY, VALID_TOL);
#endif
#ifdef VALID_MU_POSTC2R_0
    validate_real(checkpoint.mu, "0.mu.postc2r.dat", LX, LY, VALID_TOL);
#endif
}

void sheq(pfc_parms &parms, pfc_checkpoint &checkpoint, pfc_temps &temps)
{
    // calculate nonlinear terms
    nonlin1_calc(parms, checkpoint, temps);
#ifdef VALID_NONLIN1_1
    validate_complex(temps.nonlin1_q, "1.Nonlin1_q.dat", LX, LY / 2 + 1, VALID_TOL);
#endif

    // first step: predictor values of psiq
    for (int index = 0; index < LX * (LY / 2 + 1); index++)
    {
        temps.psi0_q[index][0] = checkpoint.exp_1[index] * checkpoint.psi_q[index][0] + checkpoint.cf_1[index] * temps.nonlin1_q[index][0];
        temps.psi0_q[index][1] = checkpoint.exp_1[index] * checkpoint.psi_q[index][1] + checkpoint.cf_1[index] * temps.nonlin1_q[index][1];

        if (parms.predictor_corrector_iterations > 0)
        {
            temps.psi0_sign_q[index][0] = -checkpoint.cf2_1[index] * temps.nonlin1_q[index][0];
            temps.psi0_sign_q[index][1] = -checkpoint.cf2_1[index] * temps.nonlin1_q[index][1];
        }
    }
#ifdef VALID_PSI0Q_PRECONJ_1
    validate_complex(temps.psi0_q, "1.psi0q.preconj.dat", LX, LY / 2 + 1, VALID_TOL);
#endif

#ifdef VALID_SIGN_PSIQ_1
    validate_complex(temps.psi0_sign_q, "1.sigN_psiq.dat", LX, LY / 2 + 1, VALID_TOL);
#endif

    // set the first half of first row of temps.psi0_q to complex conjugate of the second half
    for (int i = LX / 2 + 1; i < LX; i++)
    {
        auto index1 = i * (LY / 2 + 1);
        auto index2 = (LX - i) * (LY / 2 + 1);
        temps.psi0_q[index1][0] = temps.psi0_q[index2][0];
        temps.psi0_q[index1][1] = -temps.psi0_q[index2][1];
    }
#ifdef VALID_PSI0Q_1
    validate_complex(temps.psi0_q, "1.psi0q.dat", LX, LY / 2 + 1, VALID_TOL);
#endif

    // copy temps.psi0_q to temps.psi0_tmp_q
    for (int index = 0; index < LX * (LY / 2 + 1); index++)
    {
        temps.psi0_tmp_q[index][0] = temps.psi0_q[index][0];
        temps.psi0_tmp_q[index][1] = temps.psi0_q[index][1];
    }
#ifdef VALID_PSI_1
    validate_complex(temps.psi0_tmp_q, "1.psi0q.dat", LX, LY / 2 + 1, VALID_TOL);
#endif

    // back FFT (overwrites data in input)
    // cout << "call dft c2r in sheq" << endl;
    fftw_execute_dft_c2r(checkpoint.c2r, reinterpret_cast<fftw_complex *>(temps.psi0_tmp_q.data()), temps.psi0.data());

    // scale & copy back
    for (int index = 0; index < LX * LY; index++)
    {
        temps.psi0[index] *= parms.scale;
        checkpoint.psi[index] = temps.psi0[index];
    }
#ifdef VALID_PSI_1
    validate_real(checkpoint.psi, "1.psi.dat", LX, LY, VALID_TOL);
#endif

    // copy temps.psi0_q to checkpoint.psi_q
    for (int index = 0; index < LX * (LY / 2 + 1); index++)
    {
        checkpoint.psi_q[index][0] = temps.psi0_q[index][0];
        checkpoint.psi_q[index][1] = temps.psi0_q[index][1];
    }
#ifdef VALID_PSI_1
    validate_complex(checkpoint.psi_q, "1.psiq.dat", LX, LY / 2 + 1, VALID_TOL);
#endif

    // Second step: corrector
    if (parms.predictor_corrector_iterations > 0)
    {
        for (int index = 0; index < LX * (LY / 2 + 1); index++)
        {
            temps.psi0_q[index][0] += temps.psi0_sign_q[index][0];
            temps.psi0_q[index][1] += temps.psi0_sign_q[index][1];
        }

#ifdef VALID_PSI0Q_1_1
        validate_complex(temps.psi0_q, "1.1.psi0q.dat", LX, LY / 2 + 1, VALID_TOL);
#endif

        for (int iCorr = 0; iCorr < parms.predictor_corrector_iterations - 1; iCorr++)
        {
            nonlin1_calc(parms, checkpoint, temps);

#ifdef VALID_NONLIN1_Q_1_2
            if (iCorr + 2 == 2)
                validate_complex(temps.nonlin1_q, "1.2.Nonlin1_q.dat", LX, LY / 2 + 1, VALID_TOL);
#endif

#ifdef VALID_NONLIN1_Q_1_3
            if (iCorr + 2 == 3)
                validate_complex(temps.nonlin1_q, "1.3.Nonlin1_q.dat", LX, LY / 2 + 1, VALID_TOL);
#endif

            // refactor : could I be working with temps.psi0_q?
            for (int index = 0; index < LX * (LY / 2 + 1); index++)
            {
                checkpoint.psi_q[index][0] = temps.psi0_q[index][0] + checkpoint.cf2_1[index] * temps.nonlin1_q[index][0];
                checkpoint.psi_q[index][1] = temps.psi0_q[index][1] + checkpoint.cf2_1[index] * temps.nonlin1_q[index][1];
            }

            // set the first half of first row of temps.psi0_q to complex conjugate of the second half
            for (int i = LX / 2 + 1; i < LX; i++)
            {
                auto index1 = i * (LY / 2 + 1);
                auto index2 = (LX - i) * (LY / 2 + 1);
                checkpoint.psi_q[index1][0] = checkpoint.psi_q[index2][0];
                checkpoint.psi_q[index1][1] = -checkpoint.psi_q[index2][1];
            }

            // copy checkpoint.psi_q to temps.psi0_tmp_q
            for (int index = 0; index < LX * (LY / 2 + 1); index++)
            {
                temps.psi0_tmp_q[index][0] = checkpoint.psi_q[index][0];
                temps.psi0_tmp_q[index][1] = checkpoint.psi_q[index][1];
            }

#ifdef VALID_PSIQ_1_2
            if (iCorr + 2 == 2)
                validate_complex(checkpoint.psi_q, "1.2.psiq.dat", LX, LY / 2 + 1, VALID_TOL);
#endif

#ifdef VALID_PSIQ_1_3
            if (iCorr + 2 == 3)
                validate_complex(checkpoint.psi_q, "1.3.psiq.dat", LX, LY / 2 + 1, VALID_TOL);
#endif

            // back FFT (overwrites data in input)
            // cout << "call dft c2r in sheq:correction loop" << endl;
            fftw_execute_dft_c2r(checkpoint.c2r, reinterpret_cast<fftw_complex *>(temps.psi0_tmp_q.data()), checkpoint.psi.data());

            // scale
            for (int index = 0; index < LX * LY; index++)
            {
                checkpoint.psi[index] *= parms.scale;
            }

#ifdef VALID_PSI_1_2
            if (iCorr + 2 == 2)
                validate_real(checkpoint.psi, "1.2.psi.dat", LX, LY, VALID_TOL);
#endif

#ifdef VALID_PSI_1_3
            if (iCorr + 2 == 3)
                validate_real(checkpoint.psi, "1.3.psi.dat", LX, LY, VALID_TOL);
#endif

            // all but first time through corrector loop...
            auto conv = 0.;
            if (iCorr > 0)
            {
                auto max_conv_psi = 0.;
                for (int index = 0; index < LX * LY; index++)
                {
                    auto abs_psi = abs(checkpoint.psi[index]);
                    auto conv = 0.;
                    if (abs_psi > 1.e5)
                    {
                        cout << "psi diverges at t=" << checkpoint.time << ", iter=" << checkpoint.n << ", iCorr=" << iCorr << endl;
                        cout << "i=" << index % LY << ", j=" << static_cast<int>(index / LY) << ", psi(j,i)=" << checkpoint.psi[index] << endl;
                        // throw exception
                        throw;
                    }
                    if (abs_psi > parms.err)
                    {
                        conv = abs(checkpoint.psi[index] - temps.psi0[index]);
                    }
                    else
                    {
                        conv = abs(checkpoint.psi[index] - temps.psi0[index]) / abs_psi;
                    }
                    if (conv > max_conv_psi)
                    {
                        max_conv_psi = conv;
                    }
                }

                if (max_conv_psi < parms.tol)
                {
                    return;
                }
            }

            for (int index = 0; index < LX * LY; index++)
            {
                temps.psi0[index] = checkpoint.psi[index];
            }
        }

        if (parms.predictor_corrector_iterations > 0)
        {
            // should have converged...
            // write non-convergence to file
            ofstream out_nc("non_convergence.txt", ios::app);
            out_nc << "Exceeding maximum iterations=" << parms.predictor_corrector_iterations << endl;
            out_nc << "iter=" << checkpoint.n << ", t=" << checkpoint.time << ", tol=" << parms.tol << endl;
            auto min_psi = 999.;
            auto max_psi = -999.;
            auto max_conv_psi = 0.;
            for (int index = 0; index < LX * LY; index++)
            {
                auto psi = checkpoint.psi[index];
                auto abs_psi = abs(psi);
                min_psi = (min_psi < psi) ? min_psi : psi;
                max_psi = (max_psi > psi) ? max_psi : psi;
                max_conv_psi = (max_conv_psi > abs_psi) ? max_conv_psi : abs_psi;
            }
            out_nc << "max_conv_psi=" << max_conv_psi << ", range=(" << min_psi << ", " << max_psi << ")" << endl
                   << endl;
            out_nc.close();
        }
    }
}

void nonlin1_calc(pfc_parms &parms, pfc_checkpoint &checkpoint, pfc_temps &temps)
{
    // calculate non linear term.
    for (int index = 0; index < LX * LY; index++)
    {
        temps.nonlin1[index] = checkpoint.psi[index] * checkpoint.psi[index] * (checkpoint.psi[index] - parms.g);
    }

    // tramsform
    fftw_execute_dft_r2c(checkpoint.r2c, temps.nonlin1.data(), reinterpret_cast<fftw_complex *>(temps.nonlin1_q.data()));

    for (int index = 0; index < LX * (LY / 2 + 1); index++)
    {
        temps.nonlin1_q[index][0] *= -checkpoint.q2[index];
        temps.nonlin1_q[index][1] *= -checkpoint.q2[index];
    }
}

// File outputs
void delete_files()
{
    // delete f_mu.csv if it exists
    if (filesystem::exists("f_mu.csv"))
    {
        filesystem::remove("f_mu.csv");
    }
}

void write_f_mu(pfc_checkpoint &checkpoint)
{
    // write f, mu to csv files (append to existing file or create)
    ofstream out_f_mu("f_mu.csv", ios::app);
    out_f_mu << setprecision(numeric_limits<double>::max_digits10);
    out_f_mu << checkpoint.time << "," << checkpoint.fMean << "," << checkpoint.muMean << "," << endl;
    out_f_mu.close();
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
}

// Validatations
void validate_real(const vector<double> &checkArray, const string &filename, int LX, int LY, double tol)
{
    cout << "validating check array '" << filename << "'" << endl;
    string line;
    ifstream file(filename);

    for (int i = 0; i < LX; i++)
    {
        for (int j = 0; j < LY; j++)
        {
            auto index = i * LY + j;
            double value;
            file >> value;
            if (abs(checkArray[index] - value) > tol)
            {
                // Output using maximum precision
                cout << setprecision(numeric_limits<double>::max_digits10);
                cout << "TOLERANCE EXCEEDED VALIDATING " << filename << " ::> checkArray[" << i << "," << j << "]="
                     << checkArray[index] << " != " << value << ")" << endl;
            }
        }
    }
}

void validate_complex(const vector<array<double, 2>> &checkArray, const string &filename, int LX, int LY, double tol)
{
    cout << "validating check array '" << filename << "'" << endl;
    vector<array<double, 2>> data;
    string line;
    ifstream file(filename);

    while (getline(file, line))
    {
        double x, y;
        sscanf_s(line.c_str(), " (%lf,%lf)", &x, &y);
        array<double, 2> point = {x, y};
        data.push_back(point);
    }
    for (int i = 0; i < LX; i++)
    {
        for (int j = 0; j < LY; j++)
        {
            auto index = i * LY + j;
            if (abs(checkArray[index][0] - data[index][0]) + abs(checkArray[index][1] - data[index][1]) > tol)
            {
                // Output using maximum precision
                cout << setprecision(numeric_limits<double>::max_digits10);
                cout << "TOLERANCE EXCEEDED VALIDATING " << filename << " ::> checkArray[" << i << "," << j << "]=("
                     << checkArray[index][0] << "," << checkArray[index][1] << ") != ("
                     << data[index][0] << "," << data[index][1] << ")" << endl;
            }
        }
    }
}
