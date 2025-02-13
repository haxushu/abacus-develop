#ifndef LCAO_DESCRIPTOR_H
#define LCAO_DESCRIPTOR_H

#ifdef __DEEPKS

#include "../module_base/intarray.h"
#include "../module_base/complexmatrix.h"
#include "../src_pw/global.h"
#include <unordered_map>

#include "torch/script.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/api/include/torch/linalg.h"

#include "../src_lcao/LCAO_matrix.h"
#include "../module_base/intarray.h"
#include "../module_base/complexmatrix.h"
#include "../src_lcao/global_fp.h"
#include "../src_pw/global.h"
#include "../src_io/winput.h"
#include "../module_base/matrix.h"

///
/// The LCAO_Deepks contains subroutines for implementation of the DeePKS method in atomic basis.
/// In essential, it is a machine-learned correction term to the XC potential
/// in the form of delta_V=|alpha> V(D) <alpha|, where D is a list of descriptors
/// The subroutines may be roughly grouped into 3 types
/// 1. generation of projected density matrices pdm=sum_i,occ <phi_i|alpha><alpha|phi_i>
///    and then descriptors D=eig(pdm)
///    as well as their gradients with regard to atomic position, gdmx = d/dX (pdm)
///    and grad_vx = d/dX (D)
/// 2. loading the model, which requires interfaces with pytorch
/// 3. applying the correction potential, delta_V, in Kohn-Sham Hamiltonian and calculation of energy, force, stress
/// 
/// For details of DeePKS method, you can refer to [DeePKS paper](https://pubs.acs.org/doi/10.1021/acs.jctc.0c00872).
///
///
// caoyu add 2021-03-29
// wenfei modified 2022-1-5
//
class LCAO_Deepks
{

//-------------------
// public variables
//-------------------
public:
    
    ///(Unit: Ry) Correction energy provided by NN
    double E_delta = 0.0;
    ///(Unit: Ry)  \f$tr(\rho H_\delta), \rho = \sum_i{c_{i, \mu}c_{i,\nu}} \f$ (for gamma_only)
    double e_delta_band = 0.0;
    ///Correction term to the Hamiltonian matrix: \f$\langle\psi|V_\delta|\psi\rangle\f$ (for gamma only)

    double* H_V_delta;
    ///Correction term to Hamiltonian, for multi-k
    ///In R space:
    double* H_V_deltaR;
    ///In k space:
    std::complex<double>** H_V_delta_k;

    ///(Unit: Ry/Bohr) Total Force due to the DeePKS correction term \f$E_{\delta}\f$
    ModuleBase::matrix	F_delta;

//-------------------
// private variables
//-------------------
private:

	int lmaxd = 0; //max l of descirptors
	int nmaxd = 0; //#. descriptors per l
	int inlmax = 0; //tot. number {i,n,l} - atom, n, l

	// deep neural network module that provides corrected Hamiltonian term and
	// related derivatives.
	torch::jit::script::Module module;

    // saves <psi(0)|alpha(R)>, for gamma only
    std::vector<std::vector<std::unordered_map<int,std::vector<std::vector<double>>>>> nlm_save;
    
    // saves <psi(0)|alpha(R)>, for multi_k
    typedef std::tuple<int,int,int,int> key_tuple;
    std::vector<std::map<key_tuple,std::unordered_map<int,std::vector<std::vector<double>>>>> nlm_save_k;

    // projected density matrix
	double** pdm;	//[tot_Inl][2l+1][2l+1]	caoyu modified 2021-05-07
	std::vector<torch::Tensor> pdm_tensor;

	// descriptors
	std::vector<torch::Tensor> d_tensor;

	//gedm:dE/dD, [tot_Inl][2l+1][2l+1]	(E: Hartree)
	std::vector<torch::Tensor> gedm_tensor;

	//gdmx: dD/dX		\sum_{mu,nu} 2*c_mu*c_nu * <dpsi_mu/dx|alpha_m><alpha_m'|psi_nu>
	double*** gdmx;	//[natom][tot_Inl][2l+1][2l+1]
	double*** gdmy;
	double*** gdmz;

	///dE/dD, autograd from loaded model(E: Ry)
	double** gedm;	//[tot_Inl][2l+1][2l+1]

    //gvx:d(d)/dX, [natom][3][natom][des_per_atom]
    torch::Tensor gvx_tensor;

    //d(d)/dD, autograd from torch::linalg::eigh
    std::vector<torch::Tensor> gevdm_vector;

    //dD/dX, tensor form of gdmx
    std::vector<torch::Tensor> gdmr_vector;

    ///size of descriptor(projector) basis set
    int n_descriptor;

	// \sum_L{Nchi(L)*(2L+1)}
	int des_per_atom;

	ModuleBase::IntArray* alpha_index;
	ModuleBase::IntArray* inl_index;	//caoyu add 2021-05-07
	int* inl_l;	//inl_l[inl_index] = l of descriptor with inl_index

//-------------------
// subroutines, grouped according to the file they are in:
//-------------------

//-------------------
// LCAO_deepks.cpp
//-------------------

//This file contains constructor and destructor of the class LCAO_deepks, 
//as well as subroutines for initializing and releasing relevant data structures 

//Other than the constructor and the destructor, it contains 3 types of subroutines:
//1. subroutines that are related to calculating descriptors:
//  - init : allocates some arrays
//  - init_index : records the index (inl)
//  - allocate_nlm : allocates data structures (nlm_save) which is used to store <chi|alpha>
//2. subroutines that are related to calculating force label:
//  - init_gdmx : allocates gdmx; it is a private subroutine
//  - del_gdmx : releases gdmx
//3. subroutines that are related to V_delta:
//  - allocate_V_delta : allocates H_V_delta; if calculating force, it also calls
//      init_gdmx, as well as allocating F_delta
//  - allocate_V_deltaR : allcoates H_V_deltaR, for multi-k calculations

public:

    explicit LCAO_Deepks();
    ~LCAO_Deepks();

    ///Allocate memory and calculate the index of descriptor in all atoms. 
    ///(only for descriptor part, not including scf)
    void init(const LCAO_Orbitals &orb,
        const int nat,
        const int ntype,
        std::vector<int> na);

    void allocate_V_delta(const int nat,const int nloc, const int nks=1);
    void allocate_V_deltaR(const int nnr);

private:

    // arrange index of descriptor in all atoms
	void init_index(const int ntype,
        const int nat,
        std::vector<int> na,
        const int tot_inl,
        const LCAO_Orbitals &orb);
    // data structure that saves <psi|alpha>
    void allocate_nlm(const int nat);

    // array for storing gdmx, used for calculating gvx
	void init_gdmx(const int nat);
	void del_gdmx(const int nat);

//-------------------
// LCAO_deepks_psialpha.cpp
//-------------------

//wenfei 2022-1-5
//This file contains one subroutine for calculating the overlap
//between atomic basis and projector alpha : chi_mu|alpha>;
//it will be used in calculating pdm, gdmx, H_V_delta, F_delta

public:

    //calculates <chi|alpha>
    void build_psialpha(const bool& cal_deri/**< [in] 0 for 3-center intergration, 1 for its derivation*/,
        const UnitCell_pseudo &ucell,
        const LCAO_Orbitals &orb,
        Grid_Driver &GridD,
        const Parallel_Orbitals &ParaO,
        const ORB_gen_tables &UOT);

//-------------------
// LCAO_deepks_pdm.cpp
//-------------------

//This file contains subroutines for calculating pdm,
//which is defind as sum_mu,nu rho_mu,nu (<chi_mu|alpha><alpha|chi_nu>);
//as well as gdmx, which is the gradient of pdm, defined as
//sum_mu,nu rho_mu,nu d/dX(<chi_mu|alpha><alpha|chi_nu>)

//There are four subroutines in this file:
//1. cal_projected_DM, which is used for calculating pdm for gamma point calculation
//2. cal_projected_DM_k, counterpart of 1, for multi-k
//3. cal_gdmx, calculating gdmx for gamma point
//4. cal_gdmx_k, counterpart of 3, for multi-k

public:

    //S_alpha_mu * DM  * S_nu_beta
    ///calculate projected density matrix:
    ///pdm = sum_i,occ <phi_i|alpha1><alpha2|phi_k>
    void cal_projected_DM(const ModuleBase::matrix& dm/**< [in] density matrix*/,
        const UnitCell_pseudo &ucell,
        const LCAO_Orbitals &orb,
        Grid_Driver &GridD,
        const Parallel_Orbitals &ParaO);
    void cal_projected_DM_k(const std::vector<ModuleBase::ComplexMatrix>& dm,
        const UnitCell_pseudo &ucell,
        const LCAO_Orbitals &orb,
        Grid_Driver &GridD,
        const Parallel_Orbitals &ParaO,
        const K_Vectors &kv);

    //calculate the gradient of pdm with regard to atomic positions
    //d/dX D_{Inl,mm'}

    void cal_gdmx(const ModuleBase::matrix& dm,
        const UnitCell_pseudo &ucell,
        const LCAO_Orbitals &orb,
        Grid_Driver &GridD,
        const Parallel_Orbitals &ParaO);	//dD/dX, precondition of cal_gvx
    void cal_gdmx_k(const std::vector<ModuleBase::ComplexMatrix>& dm,
        const UnitCell_pseudo &ucell,
        const LCAO_Orbitals &orb,
        Grid_Driver &GridD,
        const Parallel_Orbitals &ParaO,
        const K_Vectors &kv);	//dD/dX, precondition of cal_gvx

//-------------------
// LCAO_deepks_vdelta.cpp
//-------------------

//This file contains subroutines related to V_delta, which is the deepks contribution to Hamiltonian
//defined as |alpha>V(D)<alpha|
//It also contains subroutine related to calculating e_delta_bands, which is basically
//tr (rho * V_delta)

//Four subroutines are contained in the file:
//1. add_v_delta : adds deepks contribution to hamiltonian, for gamma only
//2. add_v_delta_k : counterpart of 1, for multi-k
//3. cal_e_delta_band : calculates e_delta_bands for gamma only
//4. cal_e_delta_band_k : counterpart of 4, for multi-k

public:

    ///add dV to the Hamiltonian matrix
    void add_v_delta(const UnitCell_pseudo &ucell,
        const LCAO_Orbitals &orb,
        Grid_Driver &GridD,
        const Parallel_Orbitals &ParaO);
    void add_v_delta_k(const UnitCell_pseudo &ucell,
        const LCAO_Orbitals &orb,
        Grid_Driver &GridD,
        const Parallel_Orbitals &ParaO,
        const int nnr_in);

    ///calculate tr(\rho V_delta)
    void cal_e_delta_band(const std::vector<ModuleBase::matrix>& dm/**<[in] density matrix*/,
        const Parallel_Orbitals &ParaO);
    void cal_e_delta_band_k(const std::vector<ModuleBase::ComplexMatrix>& dm/**<[in] density matrix*/,
        const Parallel_Orbitals &ParaO,
        const int nks);

//-------------------
// LCAO_deepks_fdelta.cpp
//-------------------

//This file contains subroutines for calculating F_delta,
//which is defind as sum_mu,nu rho_mu,nu d/dX (<chi_mu|alpha>V(D)<alpha|chi_nu>)

//There are two subroutines in this file:
//1. cal_f_delta_gamma, which is used for gamma point calculation
//2. cal_f_delta_k, which is used for multi-k calculation

public:

    //for gamma only, pulay and HF terms of force are calculated together
    void cal_f_delta_gamma(const ModuleBase::matrix& dm/**< [in] density matrix*/,
        const UnitCell_pseudo &ucell,
        const LCAO_Orbitals &orb,
        Grid_Driver &GridD,
        const Parallel_Orbitals &ParaO,
        const bool isstress, ModuleBase::matrix& svnl_dalpha);

    //for multi-k, pulay and HF terms of force are calculated together
    void cal_f_delta_k(const std::vector<ModuleBase::ComplexMatrix>& dm/**<[in] density matrix*/,
        const UnitCell_pseudo &ucell,
        const LCAO_Orbitals &orb,
        Grid_Driver &GridD,
        const Parallel_Orbitals &ParaO,
        const K_Vectors &kv,
        const bool isstress, ModuleBase::matrix& svnl_dalpha);

//-------------------
// LCAO_deepks_torch.cpp
//-------------------

//This file contains interfaces with libtorch,
//including loading of model and calculating gradients

//The file contains 5 subroutines:
//1. cal_descriptor : obtains descriptors which are eigenvalues of pdm
//      by calling torch::linalg::eigh
//2. cal_gvx : gvx is used for training with force label, which is gradient of descriptors, 
//      calculated by d(des)/dX = d(pdm)/dX * d(des)/d(pdm) = gdmx * gvdm
//      using einsum
//3. cal_gvdm : d(des)/d(pdm)
//      calculated using torch::autograd::grad
//4. load_model : loads model for applying V_delta
//5. cal_gedm : calculates d(E_delta)/d(pdm)
//      this is the term V(D) that enters the expression H_V_delta = |alpha>V(D)<alpha|
//      caculated using torch::autograd::grad

public:

    ///Calculates descriptors
    ///which are eigenvalues of pdm in blocks of I_n_l
	void cal_descriptor(void);

    ///calculates gradient of descriptors w.r.t atomic positions
    ///----------------------------------------------------
    ///m, n: 2*l+1
    ///v: eigenvalues of dm , 2*l+1
    ///a,b: natom 
    /// - (a: the center of descriptor orbitals
    /// - b: the atoms whose force being calculated)
    ///gvdm*gdmx->gvx
    ///----------------------------------------------------
    void cal_gvx(const int nat);

    //load the trained neural network model
    void load_model(const std::string& model_file);

    ///calculate partial of energy correction to descriptors
    void cal_gedm(const int nat);

private:
    void cal_gvdm(const int nat);

//-------------------
// LCAO_deepks_io.cpp
//-------------------

//This file contains subroutines that deals with io,
//as well as interface with libnpy, since many arrays must be saved in numpy format

//There are 6 subroutines in this file,
//two of which prints in human-readable formats:
//1. print_descriptor : prints descriptors into descriptor.dat
//2. print_F_delta : prints F_delta into F_delta.dat

//four of which prints in numpy formats:
//3. save_npy_d : descriptor ->dm_eig.npy
//4. save_npy_gvx : gvx ->grad_vx.npy
//5. save_npy_e : energy
//6. save_npy_f : force

public:

    ///print descriptors based on LCAO basis
    void print_descriptor(const int nat);
    
    ///print the force related to\f$V_\delta\f$ for each atom
    void print_F_delta(const std::string &fname/**< [in] the name of output file*/, const UnitCell_pseudo &ucell);

	///----------------------------------------------------------------------
	///The following 4 functions save the `[dm_eig], [e_base], [f_base], [grad_vx]`
	///of current configuration as `.npy` file, when `deepks_scf = 1`.
	///After a full group of consfigurations are calculated,
    ///we need a python script to `load` and `torch.cat` these `.npy` files,
    ///and get `l_e_delta,npy` and `l_f_delta.npy` corresponding to the exact E, F data.
    /// 
    /// Unit of energy: Ry
    ///
    /// Unit of force: Ry/Bohr
    ///----------------------------------------------------------------------
	void save_npy_d(const int nat);
	void save_npy_e(const double &e/**<[in] \f$E_{base}\f$ or \f$E_{tot}\f$, in Ry*/,const std::string &e_file);
	void save_npy_f(const ModuleBase::matrix &fbase/**<[in] \f$F_{base}\f$ or \f$F_{tot}\f$, in Ry/Bohr*/,
        const std::string &f_file, const int nat);
    void save_npy_gvx(const int nat);

//-------------------
// LCAO_deepks_mpi.cpp
//-------------------

//This file contains only one subroutine, allsum_deepks
//which is used to perform allsum on a two-level pointer
//It is used in a few places in the deepks code

#ifdef __MPI

public:
    //reduces a dim 2 array
    void allsum_deepks(
        int inlmax, //first dimension
        int ndim, //second dimension
        double** mat); //the array being reduced 
#endif

};

namespace GlobalC
{
    extern LCAO_Deepks ld;
}

#endif
#endif
