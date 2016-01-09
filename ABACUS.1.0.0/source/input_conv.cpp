#include "src_pw/global.h"
#include "src_pw/tools.h"
#include "input.h"
#include "input_conv.h"
#include "src_ions/ions_move_basic.h"
#include "src_pw/optical.h"
#include "src_lcao/bfield.h"
#include "src_lcao/force_lcao.h"
#include "src_lcao/local_orbital_charge.h"
#include "src_lcao/lcao_orbitals.h"
#include "src_pw/efield.h"
#include "src_pw/vdwd2.h"
//xiaohui modified 2013-03-23, adding "//" before #include...
//#include "../../src_develop/src_siao/selinv.h"
//#include "../src_develop/src_dc/dc_info.h"
//#include "../../src_develop/src_md/md.h"

void Input_Conv::Convert(void)
{
    TITLE("Input_Conv","Convert");
	timer::tick("Input_Conv","Convert",'B');
//----------------------------------------------------------
// main parameters / electrons / spin ( 10/16 )
//----------------------------------------------------------
//  suffix
    if(INPUT.atom_file!="")global_atom_card = INPUT.atom_file;//xiaohui modify 2015-02-01
	global_wannier_card = INPUT.wannier_card;
    if(INPUT.kpoint_file!= "")global_kpoint_card = INPUT.kpoint_file;//xiaohui modify 2015-02-01
    if(INPUT.pseudo_dir != "")global_pseudo_dir = INPUT.pseudo_dir + "/"; // mohan add slash 2013-04-13 (xiaohui add 2013-06-23)
    global_pseudo_type = INPUT.pseudo_type; // mohan add this on 2013-05-20 (xiaohui add 2013-06-23)
    global_epm_pseudo_card = INPUT.epm_pseudo_card;
	ucell.latName = INPUT.latname; 
	ucell.ntype = INPUT.ntype;
	ucell.nelec = INPUT.nelec;
//        ucell.lmaxmax = INPUT.lmaxmax;
//	calculation : 2
    NBANDS = INPUT.nbands;				// 1
    NBANDS_ISTATE = INPUT.nbands_istate;// 1
	NPOOL = INPUT.npool; 				// mohan add 2010-06-09
	CALCULATION = INPUT.calculation;	// 2
    BERRY_PHASE = INPUT.berry_phase;	// 3

	EFIELD = INPUT.efield;
	Efield::edir = INPUT.edir;
	Efield::emaxpos = INPUT.emaxpos;
	Efield::eopreg = INPUT.eopreg;
	Efield::eamp = INPUT.eamp;

#ifdef __FP
	BFIELD = INPUT.bfield; // mohan add 2011-04-08
	bfid.tesla_x = INPUT.bfield_teslax;
	bfid.tesla_y = INPUT.bfield_teslay;
	bfid.tesla_z = INPUT.bfield_teslaz;
	bfid.Gauge_Origin_x = INPUT.bfield_gauge_x;
	bfid.Gauge_Origin_y = INPUT.bfield_gauge_y;
	bfid.Gauge_Origin_z = INPUT.bfield_gauge_z;
	bfid.convert(); //Zhiyuan add at 2011-12-26, for converting unit to Rydberg
	bfid.check();
#endif

	Optical::opt_epsilon2=INPUT.opt_epsilon2;	// mohan add 2010-03-24	
	Optical::opt_nbands=INPUT.opt_nbands;		// number of bands for optical transition.	
	LDA_PLUS_U = INPUT.lda_plus_u;		// 5
	DFT_FUNCTIONAL = INPUT.dft_functional;	// 6.5
    NSPIN = INPUT.nspin;				// 7
    CURRENT_SPIN = 0;					// 8
	VNA = INPUT.vna;
	GRID_SPEED = INPUT.grid_speed;		//
    FORCE = INPUT.force;				// 8.1
    FORCE_THR = INPUT.force_thr;		// 8.2
#ifdef __FP
	Force_LCAO::force_invalid_threshold_ev = INPUT.force_thr_ev2;
#endif
	BFGS_Basic::w1 = INPUT.bfgs_w1;
	BFGS_Basic::w2 = INPUT.bfgs_w2;
	Ions_Move_Basic::trust_radius_max = INPUT.trust_radius_max;
	Ions_Move_Basic::trust_radius_min = INPUT.trust_radius_min;
	Ions_Move_Basic::trust_radius_ini = INPUT.trust_radius_ini;
	Ions_Move_Basic::out_stru = INPUT.out_stru; //mohan add 2012-03-23
	STRESS = INPUT.stress;
	MOVE_IONS = INPUT.ion_dynamics;
	OUT_LEVEL = INPUT.out_level;
	Ions_Move_CG::CG_THRESHOLD = INPUT.cg_threshold; // pengfei add 2013-09-09
//----------------------------------------------------------
// new function (5/5)
//----------------------------------------------------------
    SYMMETRY = INPUT.symmetry;						// 9
	MLWF_FLAG = INPUT.mlwf_flag;					// 9.1
    //LOCAL_BASIS = INPUT.local_basis; xiaohui modify 2013-09-01				// 10
    //LINEAR_SCALING = INPUT.linear_scaling; xiaohui modify 2013-09-01			// 11
	BASIS_TYPE = INPUT.basis_type; //xiaohui add 2013-09-01
	KS_SOLVER = INPUT.ks_solver; //xiaohui add 2013-09-01
	SEARCH_RADIUS = INPUT.search_radius;			// 11.1
	SEARCH_PBC = INPUT.search_pbc;					// 11.2
    SPARSE_MATRIX = INPUT.sparse_matrix;			// 11.3
	ATOM_DISTRIBUTION = INPUT.atom_distribution;	// 11.4
//----------------------------------------------------------
// planewave (8/8)
//----------------------------------------------------------
    pw.set(
        INPUT.gamma_only,
        INPUT.ecutwfc,
        INPUT.ecutrho,
        INPUT.nx,
        INPUT.ny,
        INPUT.nz,
        INPUT.ncx,
        INPUT.ncy,
        INPUT.ncz,
		INPUT.bx,
		INPUT.by,
		INPUT.bz
    );
	GAMMA_ONLY_LOCAL = INPUT.gamma_only_local;
//----------------------------------------------------------
// diagonalization  (5/5)
//----------------------------------------------------------
    //DIAGO_TYPE = INPUT.diago_type; xiaohui modify 2013-09-01				// 12
	DIAGO_PROC = INPUT.diago_proc;				// 12.1 mohan add 2012-01-13
    DIAGO_CG_MAXITER = INPUT.diago_cg_maxiter;	// 13
	DIAGO_CG_PREC = INPUT.diago_cg_prec;		// 13.1
    DIAGO_DAVID_NDIM = INPUT.diago_david_ndim;	// 14
    ETHR = INPUT.ethr;							// 15
	NB2D = INPUT.nb2d;
	NURSE = INPUT.nurse;						// 21
	COLOUR = INPUT.colour;
	T_IN_H = INPUT.t_in_h;						// 23
	VL_IN_H = INPUT.vl_in_h;					// 24
	VNL_IN_H = INPUT.vnl_in_h;					// 25
	ZEEMAN_IN_H = INPUT.zeeman_in_h;			//
	TEST_FORCE = INPUT.test_force;				// 26
    FS_REF_ENERGY = INPUT.fs_ref_energy; 		// 16
//----------------------------------------------------------
// iteration (1/3)
//----------------------------------------------------------
    DRHO2 = INPUT.dr2;							// 17

//----------------------------------------------------------
// wavefunction / charge / potential / (2/4)
//----------------------------------------------------------
    RESTART_MODE = INPUT.restart_mode;
    wf.start_wfc = INPUT.start_wfc;
	wf.mem_saver = INPUT.mem_saver; //mohan add 2010-09-07
	en.printe    = INPUT.printe; // mohan add 2011-03-16


	//DC_Info::dcnx = INPUT.dc_nx;
	//DC_Info::dcny = INPUT.dc_ny;
	//DC_Info::dcnz = INPUT.dc_nz;

//----------------------------------------------------------
// about vdwD2									//Peize Lin add 2014-03-31
//----------------------------------------------------------	
	if(INPUT.vdwD2)
	{
		VdwD2::vdwD2 = INPUT.vdwD2;
		VdwD2::scaling = INPUT.vdwD2_scaling;
		VdwD2::damping = INPUT.vdwD2_d;
		VdwD2::C6_input(INPUT.vdwD2_C6_file, INPUT.vdwD2_C6_unit);
		VdwD2::R0_input(INPUT.vdwD2_R0_file, INPUT.vdwD2_R0_unit);
		VdwD2::model=INPUT.vdwD2_model;
		if(INPUT.vdwD2_model=="radius")
		{
			if(INPUT.vdwD2_radius_unit=="Bohr")
			{
				VdwD2::radius = INPUT.vdwD2_radius;
			}
			else
			{
				VdwD2::radius = INPUT.vdwD2_radius * BOHR_TO_A;			
			}
		}
		else if(INPUT.vdwD2_model=="period")
		{
			VdwD2::period[0]=INPUT.vdwD2_period[0];
			VdwD2::period[1]=INPUT.vdwD2_period[1];
			VdwD2::period[2]=INPUT.vdwD2_period[2];
		}
	}
	
//----------------------------------------------------------
// about selinv
//----------------------------------------------------------
//xiaohui modified 2013-03-23, adding "//" before "Selinv"
/*	Selinv::Npole = INPUT.selinv_npole;
	Selinv::temp = INPUT.selinv_temp;
	Selinv::gap = INPUT.selinv_gap;
	Selinv::deltaE = INPUT.selinv_deltae;
	Selinv::mu = INPUT.selinv_mu;
	Selinv::threshold=INPUT.selinv_threshold;
	Selinv::niter=INPUT.selinv_niter;

//----------------------------------------------------------
// about MD 
//----------------------------------------------------------
	MD::md_dt=INPUT.md_dt;	
	MD::md_restart=INPUT.md_restart;
	MD::md_tolv=INPUT.md_tolv;
	MD::md_thermostat=INPUT.md_thermostat;
	MD::md_temp0=INPUT.md_temp0;
	MD::md_tstep=INPUT.md_tstep;
	MD::md_delt=INPUT.md_delt;
*/
	timer::tick("Input_Conv","Convert",'B');
    return;
}

#ifdef __EPM
// last step in Input.init()
void Input_Conv::Convert_EPM(void)
{
    TITLE("Input","Convert_EPM");

    EPM_SPIN_ORBITAL = epm_spin_orbital;

    return;
}
#else
void Input_Conv::Convert_FP(void) 
{
    TITLE("Input","Convert_FP");
	timer::tick("Input_Conv","Convert_FP",'B');
//----------------------------------------------------------
// main parameters / electrons / spin ( 2/16 )
//----------------------------------------------------------
//	electrons::nelup = INPUT.nelup;
//	electrons::neldw = INPUT.neldw;

//----------------------------------------------------------
// occupation (3/3)
//----------------------------------------------------------
    Occupy::decision(INPUT.occupations,INPUT.smearing,INPUT.degauss);
//----------------------------------------------------------
// charge mixing(3/3)
//----------------------------------------------------------
    //chr.set_mixing(INPUT.mixing_mode, INPUT.mixing_beta, INPUT.mixing_ndim);
    chr.set_mixing(INPUT.mixing_mode, INPUT.mixing_beta, INPUT.mixing_ndim, INPUT.mixing_gg0); //mohan modify 2014-09-27, add mixing_gg0

//----------------------------------------------------------
// iteration (2/3)
//----------------------------------------------------------
    NITER = INPUT.niter;
    NSTEP = INPUT.nstep;

//----------------------------------------------------------
// wavefunction / charge / potential / (2/4)
//----------------------------------------------------------
    pot.start_pot = INPUT.start_pot;
	pot.extra_pot = INPUT.charge_extrap;//xiaohui modify 2015-02-01
    chr.out_charge = INPUT.out_charge;
	pot.out_potential = INPUT.out_potential;
    wf.out_wf = INPUT.out_wf;
	en.out_dos = INPUT.out_dos;
        en.out_band = INPUT.out_band;
#ifdef __FP
	LOC.out_dm = INPUT.out_dm;
	ParaO.out_hs = INPUT.out_hs;
	ParaO.out_lowf = INPUT.out_lowf;
#endif

	en.dos_emin_ev = INPUT.dos_emin_ev;
	en.dos_emax_ev = INPUT.dos_emax_ev;
	en.dos_edelta_ev = INPUT.dos_edelta_ev;
        en.bcoeff = INPUT.b_coef;
#ifdef __FP
//----------------------------------------------------------
// About LCAO
//----------------------------------------------------------
	ORB.ecutwfc = INPUT.lcao_ecut;
	ORB.dk = INPUT.lcao_dk;
	ORB.dR = INPUT.lcao_dr;
	ORB.Rmax = INPUT.lcao_rmax; 
#endif

	timer::tick("Input_Conv","Convert_FP",'B');
    return;
}
#endif

