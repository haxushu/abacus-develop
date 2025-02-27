#include "ELEC_scf.h"
#include "../src_pw/global.h"
#include "../src_io/chi0_hilbert.h"
#include "../src_io/print_info.h"
#include "../src_pw/symmetry_rho.h"
#include "dftu.h"
#include "LCAO_evolve.h"
#include "ELEC_cbands_k.h"
#include "ELEC_cbands_gamma.h"
#include "ELEC_evolve.h"
#include "input_update.h"
#include "../src_pw/occupy.h"
//new
#include "../src_pw/H_Ewald_pw.h"
#ifdef __DEEPKS
    #include "../module_deepks/LCAO_deepks.h"	//caoyu add 2021-06-04
#endif

ELEC_scf::ELEC_scf(){}
ELEC_scf::~ELEC_scf(){}

int ELEC_scf::iter=0;

void ELEC_scf::scf(const int &istep)
{
	ModuleBase::TITLE("ELEC_scf","scf");
	ModuleBase::timer::tick("ELEC_scf","scf");

	// (1) calculate ewald energy.
	// mohan update 2021-02-25
	H_Ewald_pw::compute_ewald(GlobalC::ucell, GlobalC::pw);

	// mohan add 2012-02-08
    set_ethr();

	// the electron charge density should be symmetrized,
	// here is the initialization
	Symmetry_rho srho;
	for(int is=0; is<GlobalV::NSPIN; is++)
	{
		srho.begin(is, GlobalC::CHR,GlobalC::pw, GlobalC::Pgrid, GlobalC::symm);
	}

//	std::cout << scientific;
//	std::cout << std::setiosflags(ios::fixed);

	if(GlobalV::OUT_LEVEL=="ie" ||GlobalV::OUT_LEVEL=="m")
	{
		if(GlobalV::COLOUR && GlobalV::MY_RANK==0)
		{
			printf( "\e[33m%-7s\e[0m", "ITER");
			printf( "\e[33m%-15s\e[0m", "ETOT(Ry)");
			if(GlobalV::NSPIN==2)
			{
				printf( "\e[33m%-10s\e[0m", "TMAG");
				printf( "\e[33m%-10s\e[0m", "AMAG");
			}
			printf( "\e[33m%-14s\e[0m", "DRHO2");
			printf( "\e[33m%-15s\e[0m", "ETOT(eV)");
			printf( "\e[33m%-11s\e[0m\n", "TIME(s)");
		}
		else
		{
			std::cout << " " << std::setw(7)<< "ITER";

			if(GlobalV::NSPIN==2)
			{
				std::cout<<std::setw(10)<<"TMAG";
				std::cout<<std::setw(10)<<"AMAG";
			}

			std::cout << std::setw(15) << "ETOT(eV)";
			std::cout << std::setw(15) << "EDIFF(eV)";
			std::cout << std::setw(11) << "DRHO2";
			std::cout << std::setw(11) << "TIME(s)" << std::endl;
		}
	}// end GlobalV::OUT_LEVEL


	for(iter=1; iter<=GlobalV::NITER; iter++)
	{
        Print_Info::print_scf(istep, iter);

		std::string ufile = "CHANGE";
		Update_input UI;
		UI.init(ufile);

		if(INPUT.dft_plus_u) GlobalC::dftu.iter_dftu = iter;
#ifdef __MPI
		auto clock_start = MPI_Wtime();
#else
		auto clock_start = std::chrono::system_clock::now();
#endif
		conv_elec = false;//mohan add 2008-05-25

		// mohan add 2010-07-16
		// used for pulay mixing.
		if(iter==1)
		{
			GlobalC::CHR.set_new_e_iteration(true);
		}
		else
		{
			GlobalC::CHR.set_new_e_iteration(false);
		}

		// set converged threshold,
		// automatically updated during self consistency, only for CG.
        this->update_ethr(iter);
        if(GlobalV::FINAL_SCF && iter==1)
        {
            init_mixstep_final_scf();
            //GlobalC::CHR.irstep=0;
            //GlobalC::CHR.idstep=0;
            //GlobalC::CHR.totstep=0;
        }

		// mohan update 2012-06-05
		GlobalC::en.calculate_harris(1);

		// mohan move it outside 2011-01-13
		// first need to calculate the weight according to
		// electrons number.
		// mohan add iter > 1 on 2011-04-02
		// because the GlobalC::en.ekb has not value now.
		// so the smearing can not be done.
		if(iter>1)Occupy::calculate_weights();

		if(GlobalC::wf.start_wfc == "file")
		{
			if(iter==1)
			{
				std::cout << " WAVEFUN -> CHARGE " << std::endl;

				// The occupation should be read in together.
				// Occupy::calculate_weights(); //mohan add 2012-02-15

				// calculate the density matrix using read in wave functions
				// and the ncalculate the charge density on grid.
				GlobalC::LOC.sum_bands();
				// calculate the local potential(rho) again.
				// the grid integration will do in later grid integration.


				// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
				// a puzzle remains here.
				// if I don't renew potential,
				// The dr2 is very small.
				// OneElectron, Hartree and
				// Exc energy are all correct
				// except the band energy.
				//
				// solved by mohan 2010-09-10
				// there are there rho here:
				// rho1: formed by read in orbitals.
				// rho2: atomic rho, used to construct H
				// rho3: generated by after diagonalize
				// here converged because rho3 and rho1
				// are very close.
				// so be careful here, make sure
				// rho1 and rho2 are the same rho.
				// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
				GlobalC::pot.vr = GlobalC::pot.v_of_rho(GlobalC::CHR.rho, GlobalC::CHR.rho_core);
				GlobalC::en.delta_escf();
				if (ELEC_evolve::td_vext == 0)
				{
					GlobalC::pot.set_vr_eff();
				}
				else
				{
					GlobalC::pot.set_vrs_tddft(istep);
				}
			}
		}

		// fuxiang add 2016-11-1
		// need reconstruction in near future -- mohan 2021-02-09
		// the initialization of wave functions should be moved to
		// somewhere else
		if(ELEC_evolve::tddft == 1 && iter == 2)
		{
			this->WFC_init = new std::complex<double>**[GlobalC::kv.nks];
			for(int ik=0; ik<GlobalC::kv.nks; ik++)
			{
				this->WFC_init[ik] = new std::complex<double>*[GlobalV::NBANDS];
			}
			for(int ik=0; ik<GlobalC::kv.nks; ik++)
			{
				for(int ib=0; ib<GlobalV::NBANDS; ib++)
				{
					this->WFC_init[ik][ib] = new std::complex<double>[GlobalV::NLOCAL];
				}
			}
			if(istep>=1)
			{
				for (int ik=0; ik<GlobalC::kv.nks; ik++)
				{
					for (int ib=0; ib<GlobalV::NBANDS; ib++)
					{
						for (int i=0; i<GlobalV::NLOCAL; i++)
						{
							WFC_init[ik][ib][i] = GlobalC::LOWF.WFC_K[ik][ib][i];
						}
					}
				}
			}
			else
			{
				for (int ik=0; ik<GlobalC::kv.nks; ik++)
				{
					for (int ib=0; ib<GlobalV::NBANDS; ib++)
					{
						for (int i=0; i<GlobalV::NLOCAL; i++)
						{
							WFC_init[ik][ib][i] = std::complex<double>(0.0,0.0);
						}
					}
				}
			}
		}

		// calculate exact-exchange
		switch(GlobalC::xcf.iexch_now)						// Peize Lin add 2018-10-30
		{
			case 5:    case 6:   case 9:
				if( !GlobalC::exx_global.info.separate_loop )
				{
					GlobalC::exx_lcao.cal_exx_elec();
				}
				break;
		}

		if(INPUT.dft_plus_u)
		{
			GlobalC::dftu.cal_slater_UJ(istep, iter); // Calculate U and J if Yukawa potential is used
		}

		// (1) calculate the bands.
		// mohan add 2021-02-09
		if(GlobalV::GAMMA_ONLY_LOCAL)
		{
			ELEC_cbands_gamma::cal_bands(istep, GlobalC::UHM);
		}
		else
		{
			if(ELEC_evolve::tddft && istep >= 1 && iter > 1)
			{
				ELEC_evolve::evolve_psi(istep, GlobalC::UHM, this->WFC_init);
			}
			else
			{
				ELEC_cbands_k::cal_bands(istep, GlobalC::UHM);
			}
		}


//		for(int ib=0; ib<GlobalV::NBANDS; ++ib)
//		{
//			std::cout << ib+1 << " " << GlobalC::wf.ekb[0][ib] << std::endl;
//		}

		//-----------------------------------------------------------
		// only deal with charge density after both wavefunctions.
		// are calculated.
		//-----------------------------------------------------------
		if(GlobalV::GAMMA_ONLY_LOCAL && GlobalV::NSPIN == 2 && GlobalV::CURRENT_SPIN == 0) continue;


		if(conv_elec)
		{
			ModuleBase::timer::tick("ELEC_scf","scf");
			return;
		}

		GlobalC::en.eband  = 0.0;
		GlobalC::en.ef     = 0.0;
		GlobalC::en.ef_up  = 0.0;
		GlobalC::en.ef_dw  = 0.0;

		// demet is included into eband.
		//if(GlobalV::DIAGO_TYPE!="selinv")
		{
			GlobalC::en.demet  = 0.0;
		}

		// (2)
		GlobalC::CHR.save_rho_before_sum_band();

		// (3) sum bands to calculate charge density
		Occupy::calculate_weights();

		if (GlobalV::ocp == 1)
		{
			for (int ik=0; ik<GlobalC::kv.nks; ik++)
			{
				for (int ib=0; ib<GlobalV::NBANDS; ib++)
				{
					GlobalC::wf.wg(ik,ib)=GlobalV::ocp_kb[ik*GlobalV::NBANDS+ib];
				}
			}
		}


		for(int ik=0; ik<GlobalC::kv.nks; ++ik)
		{
			GlobalC::en.print_band(ik);
		}

		// if selinv is used, we need this to calculate the charge
		// using density matrix.
		GlobalC::LOC.sum_bands();

		// add exx
		// Peize Lin add 2016-12-03
		GlobalC::en.set_exx();

		// Peize Lin add 2020.04.04
		if(Exx_Global::Hybrid_Type::HF==GlobalC::exx_lcao.info.hybrid_type
			|| Exx_Global::Hybrid_Type::PBE0==GlobalC::exx_lcao.info.hybrid_type
			|| Exx_Global::Hybrid_Type::HSE==GlobalC::exx_lcao.info.hybrid_type)
		{
			if(GlobalC::restart.info_load.load_H && GlobalC::restart.info_load.load_H_finish && !GlobalC::restart.info_load.restart_exx)
			{
				GlobalC::exx_global.info.set_xcfunc(GlobalC::xcf);
				GlobalC::exx_lcao.cal_exx_elec();
				GlobalC::restart.info_load.restart_exx = true;
			}
		}


		// if DFT+U calculation is needed, this function will calculate
		// the local occupation number matrix and energy correction
		if(INPUT.dft_plus_u)
		{
			if(GlobalV::GAMMA_ONLY_LOCAL) GlobalC::dftu.cal_occup_m_gamma(iter);
			else GlobalC::dftu.cal_occup_m_k(iter);

		 	GlobalC::dftu.cal_energy_correction(istep);
			GlobalC::dftu.output();
		}

		// (4) mohan add 2010-06-24
		// using new charge density.
		GlobalC::en.calculate_harris(2);

		// (5) symmetrize the charge density
		Symmetry_rho srho;
		for(int is=0; is<GlobalV::NSPIN; is++)
		{
			srho.begin(is, GlobalC::CHR,GlobalC::pw, GlobalC::Pgrid, GlobalC::symm);
		}

		// (6) compute magnetization, only for spin==2
        GlobalC::ucell.magnet.compute_magnetization();

		// resume codes!
		//-------------------------------------------------------------------------
		// this->GlobalC::LOWF.init_Cij( 0 ); // check the orthogonality of local orbital.
		// GlobalC::CHR.sum_band(); use local orbital in plane wave basis to calculate bands.
		// but must has evc first!
		//-------------------------------------------------------------------------

		// (7) calculate delta energy
		GlobalC::en.deband = GlobalC::en.delta_e();

		// (8) Mix charge density
		GlobalC::CHR.mix_rho(dr2,0,GlobalV::DRHO2,iter,conv_elec);

		// Peize Lin add 2020.04.04
		if(GlobalC::restart.info_save.save_charge)
		{
			for(int is=0; is<GlobalV::NSPIN; ++is)
			{
				GlobalC::restart.save_disk("charge", is);
			}
		}

		// (9) Calculate new potential according to new Charge Density.

		if(conv_elec || iter==GlobalV::NITER)
		{
			if(GlobalC::pot.out_potential<0) //mohan add 2011-10-10
			{
				GlobalC::pot.out_potential = -2;
			}
		}

		if(!conv_elec)
		{
			// option 1
			GlobalC::pot.vr = GlobalC::pot.v_of_rho(GlobalC::CHR.rho, GlobalC::CHR.rho_core);
			GlobalC::en.delta_escf();

			// option 2
			//------------------------------
			// mohan add 2012-06-08
			// use real E_tot functional.
			//------------------------------
			/*
			GlobalC::pot.vr = GlobalC::pot.v_of_rho(GlobalC::CHR.rho_save, GlobalC::CHR.rho);
			GlobalC::en.calculate_etot();
			GlobalC::en.print_etot(conv_elec, istep, iter, dr2, 0.0, GlobalV::ETHR, avg_iter,0);
			GlobalC::pot.vr = GlobalC::pot.v_of_rho(GlobalC::CHR.rho, GlobalC::CHR.rho_core);
			GlobalC::en.delta_escf();
			*/
		}
		else
		{
			GlobalC::pot.vnew = GlobalC::pot.v_of_rho(GlobalC::CHR.rho, GlobalC::CHR.rho_core);
			//(used later for scf correction to the forces )
			GlobalC::pot.vnew -= GlobalC::pot.vr;
			GlobalC::en.descf = 0.0;
		}

		//-----------------------------------
		// output charge density for tmp
		//-----------------------------------
		for(int is=0; is<GlobalV::NSPIN; is++)
		{
			const int precision = 3;

			std::stringstream ssc;
			ssc << GlobalV::global_out_dir << "tmp" << "_SPIN" << is + 1 << "_CHG";
			GlobalC::CHR.write_rho(GlobalC::CHR.rho_save[is], is, iter, ssc.str(), precision );//mohan add 2007-10-17

			std::stringstream ssd;

			if(GlobalV::GAMMA_ONLY_LOCAL)
			{
				ssd << GlobalV::global_out_dir << "tmp" << "_SPIN" << is + 1 << "_DM";
			}
			else
			{
				ssd << GlobalV::global_out_dir << "tmp" << "_SPIN" << is + 1 << "_DM_R";
			}
			GlobalC::LOC.write_dm( is, iter, ssd.str(), precision );

			//LiuXh modify 20200701
			/*
			std::stringstream ssp;
			ssp << GlobalV::global_out_dir << "tmp" << "_SPIN" << is + 1 << "_POT";
			GlobalC::pot.write_potential( is, iter, ssp.str(), GlobalC::pot.vr, precision );
			*/
		}

		// (10) add Vloc to Vhxc.
		if(ELEC_evolve::td_vext == 0)
		{
			GlobalC::pot.set_vr_eff();
		}
		else
		{
			GlobalC::pot.set_vrs_tddft(istep);
		}

		//time_finish=std::time(NULL);
#ifdef __MPI
		double duration = (double)(MPI_Wtime() - clock_start);
#else
		double duration = (double)(std::chrono::system_clock::now() - clock_start) / CLOCKS_PER_SEC;
#endif
		//double duration_time = difftime(time_finish, time_start);
		//std::cout<<"Time_clock\t"<<"Time_time"<<std::endl;
		//std::cout<<duration<<"\t"<<duration_time<<std::endl;

		// (11) calculate the total energy.
		GlobalC::en.calculate_etot();

		// avg_iter is an useless variable in LCAO,
		// will fix this interface in future -- mohan 2021-02-10
		int avg_iter=0;
		GlobalC::en.print_etot(conv_elec, istep, iter, dr2, duration, GlobalV::ETHR, avg_iter);

		GlobalC::en.etot_old = GlobalC::en.etot;

		if (conv_elec || iter==GlobalV::NITER)
		{
			//--------------------------------------
			// output charge density for converged,
			// 0 means don't need to consider iter,
			//--------------------------------------
			if( GlobalC::chi0_hilbert.epsilon)                                    // pengfei 2016-11-23
			{
				std::cout <<"eta = "<<GlobalC::chi0_hilbert.eta<<std::endl;
				std::cout <<"domega = "<<GlobalC::chi0_hilbert.domega<<std::endl;
				std::cout <<"nomega = "<<GlobalC::chi0_hilbert.nomega<<std::endl;
				std::cout <<"dim = "<<GlobalC::chi0_hilbert.dim<<std::endl;
				//std::cout <<"oband = "<<GlobalC::chi0_hilbert.oband<<std::endl;
				GlobalC::chi0_hilbert.Chi();
			}

			//quxin add for DFT+U for nscf calculation
			if(INPUT.dft_plus_u)
			{
				if(GlobalC::CHR.out_charge)
				{
					std::stringstream sst;
					sst << GlobalV::global_out_dir << "onsite.dm";
					GlobalC::dftu.write_occup_m( sst.str() );
				}
			}

			for(int is=0; is<GlobalV::NSPIN; is++)
			{
				const int precision = 3;

				std::stringstream ssc;
				ssc << GlobalV::global_out_dir << "SPIN" << is + 1 << "_CHG";
				GlobalC::CHR.write_rho(GlobalC::CHR.rho_save[is], is, 0, ssc.str() );//mohan add 2007-10-17

				std::stringstream ssd;
				if(GlobalV::GAMMA_ONLY_LOCAL)
				{
					ssd << GlobalV::global_out_dir << "SPIN" << is + 1 << "_DM";
				}
				else
				{
					ssd << GlobalV::global_out_dir << "SPIN" << is + 1 << "_DM_R";
				}
				GlobalC::LOC.write_dm( is, 0, ssd.str(), precision );

				if(GlobalC::pot.out_potential == 1) //LiuXh add 20200701
				{
					std::stringstream ssp;
					ssp << GlobalV::global_out_dir << "SPIN" << is + 1 << "_POT";
					GlobalC::pot.write_potential( is, 0, ssp.str(), GlobalC::pot.vr_eff, precision );
				}

				//LiuXh modify 20200701
				/*
				//fuxiang add 2017-03-15
				std::stringstream sse;
				sse << GlobalV::global_out_dir << "SPIN" << is + 1 << "_DIPOLE_ELEC";
				GlobalC::CHR.write_rho_dipole(GlobalC::CHR.rho_save, is, 0, sse.str());
				*/
			}

			if(conv_elec)
			{
				GlobalV::ofs_running << "\n charge density convergence is achieved" << std::endl;
            	GlobalV::ofs_running << " final etot is " << GlobalC::en.etot * ModuleBase::Ry_to_eV << " eV" << std::endl;
			}

			if(GlobalV::OUT_LEVEL != "m") 
			{
				print_eigenvalue(GlobalV::ofs_running);
			}

			if(conv_elec)
			{
 				//xiaohui add "OUT_LEVEL", 2015-09-16
				if(GlobalV::OUT_LEVEL != "m") GlobalV::ofs_running << std::setprecision(16);
				if(GlobalV::OUT_LEVEL != "m") GlobalV::ofs_running << " EFERMI = " << GlobalC::en.ef * ModuleBase::Ry_to_eV << " eV" << std::endl;
				if(GlobalV::OUT_LEVEL=="ie")
				{
					GlobalV::ofs_running << " " << GlobalV::global_out_dir << " final etot is " << GlobalC::en.etot * ModuleBase::Ry_to_eV << " eV" << std::endl;
				}
#ifdef __DEEPKS
				if (GlobalV::out_descriptor)	//caoyu add 2021-06-04
				{
                    GlobalC::ld.save_npy_e(GlobalC::en.etot, "e_tot.npy");
                    if (GlobalV::deepks_scf) {
                        GlobalC::ld.save_npy_e(GlobalC::en.etot - GlobalC::ld.E_delta, "e_base.npy");//ebase :no deepks E_delta including
                    }
                    else
                    {
                        GlobalC::ld.save_npy_e(GlobalC::en.etot, "e_base.npy");  // no scf, e_tot=e_base
                    }

				}
#endif
			}
			else
			{
				GlobalV::ofs_running << " !! convergence has not been achieved @_@" << std::endl;
				if(GlobalV::OUT_LEVEL=="ie" || GlobalV::OUT_LEVEL=="m") //xiaohui add "m" option, 2015-09-16
				std::cout << " !! CONVERGENCE HAS NOT BEEN ACHIEVED !!" << std::endl;
			}

//			ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running,"ELECTRONS CONVERGED!");
			ModuleBase::timer::tick("ELEC_scf","scf");
			return;
		}
	}

	// fuxiang add, should be reconstructed in near future -- mohan note 2021-02-09
	if (ELEC_evolve::tddft==1)
	{
		delete[] WFC_init;
	}

	ModuleBase::timer::tick("ELEC_scf","scf");
	return;
}


void ELEC_scf::init_mixstep_final_scf(void)
{
    ModuleBase::TITLE("ELEC_scf","init_mixstep_final_scf");

    GlobalC::CHR.irstep=0;
    GlobalC::CHR.idstep=0;
    GlobalC::CHR.totstep=0;

    return;
}

