#ifndef PWBASISK_H
#define PWBASISK_H

#include "pw_basis.h"
namespace ModulePW
{

///
/// Special pw_basis class.
/// It includes different k-points
/// plane waves: <r|g,k> = 1/sqrt(V) * exp(i(k+g)r)
/// f(r) = 1/sqrt(V) * \sum_g{c(g)*exp(i(k+g)r)}
///
class PW_Basis_K : public PW_Basis
{

public:
    PW_Basis_K();
    ~PW_Basis_K();

    //init parameters of pw_basis_k class
    void initparameters(
        bool gamma_only_in,
        double ecut_in,
        int nk_in, //number of k points in this pool
        ModuleBase::Vector3<double> *kvec_d, // Direct coordinates of k points
        int distribution_type_in
    );


public:
    int nks;//number of k points in this pool
    ModuleBase::Vector3<double> *kvec_d; // Direct coordinates of k points
    ModuleBase::Vector3<double> *kvec_c; // Cartesian coordinates of k points
    int *npwk; //[nks] number of plane waves of different k-points
    int npwk_max; //max npwk among all nks k-points
    double gk_ecut; //Energy cut off for (g+k)^2/2

public:
    //prepare for transforms between real and reciprocal spaces
    void setuptransform();
    
    //create igl2isz_k map array for fft
    void setupIndGk();

    int *igl2isz_k; //[npwk_max*nks] map (ig,ik) to (is,iz) 

    //create Direct coordinate, Cartesian coordinate, norm2 of plane waves in each processor
    void collect_local_pw();

public:
    void real2recip(double * in, std::complex<double> * out, int ik); //in:(nplane,nx*ny)  ; out(nz, ns)
    void real2recip(std::complex<double> * in, std::complex<double> * out, int ik); //in:(nplane,nx*ny)  ; out(nz, ns)
    void recip2real(std::complex<double> * in, double *out, int ik); //in:(nz, ns)  ; out(nplane,nx*ny)
    void recip2real(std::complex<double> * in, std::complex<double> * out, int ik); //in:(nz, ns)  ; out(nplane,nx*ny)

#ifdef __MIX_PRECISION
    void real2recip(float * in, std::complex<float> * out, int ik); //in:(nplane,nx*ny)  ; out(nz, ns)
    void real2recip(std::complex<float> * in, std::complex<float> * out, int ik); //in:(nplane,nx*ny)  ; out(nz, ns)
    void recip2real(std::complex<float> * in, float *out, int ik); //in:(nz, ns)  ; out(nplane,nx*ny)
    void recip2real(std::complex<float> * in, std::complex<float> * out, int ik); //in:(nz, ns)  ; out(nplane,nx*ny)
#endif

public:
    //operator:

    //calculate g+k
    ModuleBase::Vector3<double> get_GPlusK_cartesian(const int ik, const int ig) const;
    //calculate g+k.x/y/z
    double get_GPlusK_cartesian_projection(const int ik, const int ig, const int axis) const;
    //calculate (g+k)^2
    double get_SquareGPlusK_cartesian(const int ik, const int ig) const;
};

}
#endif //PlaneWave_K class

