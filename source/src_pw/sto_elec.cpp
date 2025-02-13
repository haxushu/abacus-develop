#include "../module_base/global_function.h"
#include "../module_base/global_variable.h"
#include "global.h"
#include "sto_elec.h"
#include "occupy.h"
#include "../src_pw/symmetry_rho.h"
#include "../src_io/wf_io.h"
#include "H_Ewald_pw.h"
#include "../module_base/timer.h"

double Stochastic_Elec::avg_iter = 0;

Stochastic_Elec::Stochastic_Elec()
{
}

Stochastic_Elec::~Stochastic_Elec()
{
}


void Stochastic_Elec::scf_stochastic(const int &istep)
{
	ModuleBase::timer::tick("Elec_Stochastic","scf_stochastic");

	// mohan update 2021-02-25
	H_Ewald_pw::compute_ewald(GlobalC::ucell, GlobalC::pw);

    set_ethr();

	if(GlobalV::OUT_LEVEL=="ie")
	{
		std::cout << std::setprecision(12);
    	std::cout<< " " << std::setw(7)<< "ITER";

		if(GlobalV::NSPIN==2)
		{
			std::cout<<std::setw(10)<<"TMAG";
			std::cout<<std::setw(10)<<"AMAG";
		}

        std::cout<<std::setw(15)<< "ETOT(eV)"<<std::setw(15)<< "EDIFF(eV)"<<std::setw(11)<< "DRHO2"; // pengfei Li added 2015-1-31
		if(GlobalV::KS_SOLVER=="cg")
		{
			std::cout<<std::setw(11)<<"CG_ITER";
		}

		std::cout<<std::setw(11)<< "TIME(S)";

        std::cout<<std::endl;
	}

	Symmetry_rho srho;
	for(int is=0; is<GlobalV::NSPIN; is++)
	{
		srho.begin(is, GlobalC::CHR,GlobalC::pw, GlobalC::Pgrid, GlobalC::symm);
	}

	// conv_elec is a member of Threshold_Elec
	this->conv_elec = false;

	clock_t start,finish;
	double duration = 0.0;
	GlobalC::sto_wf.init(INPUT.nbands_sto, 
		INPUT.nche_sto, 
		INPUT.seed_sto, 
		INPUT.emax_sto, 
		INPUT.emin_sto, 
		INPUT.stotype);
	int chetype = 1;

	for (this->iter = 1;iter <= GlobalV::NITER;iter++)
    {
		GlobalV::ofs_running
		<< "\n PW-STOCHASTIC ALGO --------- ION=" << std::setw(4) << istep + 1
		<< "  ELEC=" << std::setw(4) << iter
		<< "--------------------------------\n";
		if(iter==1)
		{
			GlobalC::CHR.new_e_iteration = true;
        }
		else
		{
			GlobalC::CHR.new_e_iteration = false;
		}
		if(GlobalV::FINAL_SCF && iter==1)
        {
            GlobalC::CHR.irstep=0;
            GlobalC::CHR.idstep=0;
            GlobalC::CHR.totstep=0;
        }

		// record the start time.
        start=std::clock();


		//(1) set converged threshold,
		// automatically updated during self consistency.
        //this->update_ethr(iter);
        if(GlobalV::FINAL_SCF && iter==1)
		{
			GlobalV::ETHR = 1.0e-4/GlobalC::CHR.nelec; //smaller GlobalV::ETHR than KS-DFT
		}
        else
		{
			if (iter == 2)
        	{
            	GlobalV::ETHR = 1.0e-4/GlobalC::CHR.nelec;
        	}
			GlobalV::ETHR = std::min( GlobalV::ETHR, 0.1*dr2/ std::max(1.0, GlobalC::CHR.nelec));
        }



		// mohan move harris functional to here, 2012-06-05
		// use 'rho(in)' and 'v_h and v_xc'(in)
		//GlobalC::en.calculate_harris(1);

		// first_iter_again:					// Peize Lin delete 2019-05-01

		//(2) calculate band energy using cg or davidson method.
		// output the new eigenvalues and wave functions.
		if(GlobalV::NBANDS > 0 && GlobalV::MY_POOL == 0)
		{
			this->c_bands(istep+1);
		}
		else
		{
			GlobalC::hm.hpw.init_k(0); //only GAMMA
			//In fact, GlobalC::hm.hpw.init_k has been done in GlobalC::wf.wfcinit();
		}
		GlobalC::kv.wk[0] = 2;// GAMMA temporary

        if (check_stop_now()) return;

		GlobalC::en.eband  = 0.0;
        GlobalC::en.demet  = 0.0;
        GlobalC::en.ef     = 0.0;
        GlobalC::en.ef_up  = 0.0;
        GlobalC::en.ef_dw  = 0.0;

		//(3) save change density as previous charge,
		// prepared fox mixing.
		if(GlobalV::MY_POOL == 0)
		{
        	GlobalC::CHR.save_rho_before_sum_band();
		}

		//prepare wavefunction&eband for other pools
#ifdef __MPI
		if(GlobalV::NBANDS > 0)
		{
			MPI_Bcast(GlobalC::wf.evc[0].c, GlobalC::wf.npwx*GlobalV::NBANDS*2, MPI_DOUBLE , 0, PARAPW_WORLD);
			MPI_Bcast(GlobalC::wf.ekb[0], GlobalV::NBANDS, MPI_DOUBLE, 0, PARAPW_WORLD);
		}
#endif

		//(4) calculate fermi energy.
		int ndim;
    	if(GlobalC::sto_wf.stotype == "pw")
		{
    	    ndim = GlobalC::wf.npw;
		}
    	else
		{
    	    ndim = GlobalC::pw.nrxx;
		}

		if(iter == 1)
		{
			stoiter.init( ndim, chetype );
		}

		stoiter.orthog();
		stoiter.checkemm(iter);	//check and reset emax & emin
		stoiter.test();  //only for test
		stoiter.itermu(iter);


		//(5) calculate new charge density
		// calculate KS rho.
		if(GlobalV::NBANDS > 0)
		{
			if(GlobalV::MY_POOL == 0)
			{
				GlobalC::CHR.sum_band();
			}
			else
			{
				for(int is=0; is<GlobalV::NSPIN; is++)
				{
					ModuleBase::GlobalFunc::ZEROS(GlobalC::CHR.rho[is], GlobalC::pw.nrxx);
				}
			}
			//for(int is = 0; is < GlobalV::NSPIN; ++is)
			//{
			//	MPI_Bcast(GlobalC::CHR.rho[is], GlobalC::pw.nrxx, MPI_DOUBLE , 0,PARAPW_WORLD);
			//}
#ifdef __MPI
			MPI_Bcast(&GlobalC::en.eband,1, MPI_DOUBLE, 0,PARAPW_WORLD);
#endif
		}
		else
		{
			for(int is=0; is<GlobalV::NSPIN; is++)
			{
				ModuleBase::GlobalFunc::ZEROS(GlobalC::CHR.rho[is], GlobalC::pw.nrxx);
			}
		}

	// calculate stochastic rho
		stoiter.sum_stoband();


		//(6) calculate the delta_harris energy
		// according to new charge density.
		// mohan add 2009-01-23
		//GlobalC::en.calculate_harris(2);


		Symmetry_rho srho;
		for(int is=0; is<GlobalV::NSPIN; is++)
		{
			srho.begin(is, GlobalC::CHR,GlobalC::pw, GlobalC::Pgrid, GlobalC::symm);
		}


		//(8) deband is calculated from "output" charge density calculated
		// in sum_band
		// need 'rho(out)' and 'vr (v_h(in) and v_xc(in))'
        GlobalC::en.deband = GlobalC::en.delta_e();


		// tr2_min used only in first scf iteraton
		double diago_error = 0.0;
		if(iter==1)
		{
			// if 'dr2 < GlobalV::ETHR * nelec' happen,
			// in other word, 'dr2 < diago_error'
			// we update GlobalV::ETHR.
			diago_error = GlobalV::ETHR*std::max(1.0, GlobalC::CHR.nelec);
		}

		// if converged is achieved, or the self-consistent error(dr2)
		// is samller than the estimated error due to diagonalization(diago_error)
		// rhoin and rho are unchanged:
		// rhoin contain the input charge density and
		// rho contain the output charge density.
		// in other cases rhoin contains the mixed charge density
		// (the new input density) while rho is unchanged.
		if(GlobalV::MY_POOL == 0)
		{
			GlobalC::CHR.mix_rho(dr2,diago_error,GlobalV::DRHO2,iter,conv_elec);
		}

#ifdef __MPI
		MPI_Bcast(&dr2, 1, MPI_DOUBLE , 0, PARAPW_WORLD);
		MPI_Bcast(&conv_elec, 1, MPI_DOUBLE , 0, PARAPW_WORLD);
		MPI_Bcast(GlobalC::CHR.rho[0], GlobalC::pw.nrxx, MPI_DOUBLE, 0, PARAPW_WORLD);
#endif

		//			if(GlobalV::MY_RANK==0)
		//			{
		//				ofs_mix << std::setw(5) << iter << std::setw(20) << dr2 << std::endl;
		//			}

		if (iter==1)
		{
			if (dr2 < diago_error)
			{
				GlobalV::ofs_running << " Notice: Threshold on eigenvalues was too large.\n";

				ModuleBase::WARNING("scf","Threshold on eigenvalues was too large.");
				GlobalV::ofs_running << " dr2=" << dr2 << " < diago_error=" << diago_error << std::endl;

				// update GlobalV::ETHR.
				GlobalV::ofs_running << " Origin GlobalV::ETHR = " << GlobalV::ETHR << std::endl;
				GlobalV::ETHR = dr2 / GlobalC::CHR.nelec;
				GlobalV::ofs_running << " New    GlobalV::ETHR = " << GlobalV::ETHR << std::endl;
				//                  goto first_iter_again;
			}
		}

        if (!conv_elec)
        {
			// not converged yet, calculate new potential from mixed charge density
            GlobalC::pot.vr = GlobalC::pot.v_of_rho(GlobalC::CHR.rho, GlobalC::CHR.rho_core);

			// because <T+V(ionic)> = <eband+deband> are calculated after sum
			// band, using output charge density.
			// but E_Hartree(GlobalC::en.ehart) and Exc(GlobalC::en.etxc) are calculated in v_of_rho above,
			// using the mixed charge density.
			// so delta_escf corrects for this difference at first order.
            GlobalC::en.delta_escf();
        }
        else
        {
			for(int is=0; is<GlobalV::NSPIN; ++is)
			{
				for(int ir=0; ir<GlobalC::pw.nrxx; ++ir)
				{
					GlobalC::pot.vnew(is,ir) = GlobalC::pot.vr(is,ir);
				}
			}

			// the new potential V(PL)+V(H)+V(xc)
            GlobalC::pot.vr = GlobalC::pot.v_of_rho(GlobalC::CHR.rho, GlobalC::CHR.rho_core);

            //( vnew used later for scf correction to the forces )
	    	GlobalC::pot.vnew = GlobalC::pot.vr - GlobalC::pot.vnew;
            GlobalC::en.descf = 0.0;

        }



		// output for tmp.
		for(int is=0; is<GlobalV::NSPIN; is++)
		{
			std::stringstream ssc;
			ssc << GlobalV::global_out_dir << "tmp" << "_SPIN" << is + 1 << "_CHG";
			GlobalC::CHR.write_rho(GlobalC::CHR.rho_save[is], is, iter, ssc.str(), 3);//mohan add 2007-10-17
		}



			GlobalC::pot.set_vr_eff();

        //print_eigenvalue(GlobalV::ofs_running);
		GlobalC::en.calculate_etot();


        finish=clock();
        duration = (double)(finish - start) / CLOCKS_PER_SEC;

		GlobalC::en.print_etot(conv_elec, istep, iter, dr2, duration, GlobalV::ETHR, avg_iter);
        if (conv_elec || iter==GlobalV::NITER)
        {
			for(int is=0; is<GlobalV::NSPIN; is++)
			{
        		std::stringstream ssc;
        		ssc << GlobalV::global_out_dir << "SPIN" << is + 1 << "_CHG";
        		GlobalC::CHR.write_rho(GlobalC::CHR.rho_save[is], is, 0, ssc.str() );//mohan add 2007-10-17
			}

			if(conv_elec)
			{
				//GlobalV::ofs_running << " convergence is achieved" << std::endl;			
				//GlobalV::ofs_running << " !FINAL_ETOT_IS " << GlobalC::en.etot * ModuleBase::Ry_to_eV << " eV" << std::endl; 
				GlobalV::ofs_running << " charge density convergence is achieved" << std::endl;
                                GlobalV::ofs_running << " final etot is " << GlobalC::en.etot * ModuleBase::Ry_to_eV << " eV" << std::endl;
			}
			else
			{
				GlobalV::ofs_running << " convergence has NOT been achieved!" << std::endl;
			}

			if(GlobalV::OUT_LEVEL != "m") 
			{
				print_eigenvalue(GlobalV::ofs_running);
			}
			ModuleBase::timer::tick("Elec_Stochastic","scf_stochastic");
            return;
        }

        //if ( imix >= 0 )  GlobalC::CHR.rho = GlobalC::CHR.rho_save;
        //GlobalV::ofs_running << "\n start next iterate for idum ";
		
    } 
	
	ModuleBase::timer::tick("Elec_Stochastic","scf_stochastic");
    return;
} // end electrons


bool Stochastic_Elec::check_stop_now(void)
{
    bool check_stop_now = false;

    if (check_stop_now)
    {
        conv_elec = false;
    }

    return check_stop_now;
}


void Stochastic_Elec::c_bands(const int &istep)
{
	if (GlobalV::test_elec) ModuleBase::TITLE("electrons","c_bands");
	ModuleBase::timer::tick("Elec_Stochastic","c_bands");

	int precondition_type = 2;

	double *h_diag = new double[GlobalC::wf.npwx * GlobalV::NPOL];//added by zhengdy-soc
	ModuleBase::GlobalFunc::ZEROS(h_diag, GlobalC::wf.npwx * GlobalV::NPOL);
    
	avg_iter = 0.0;

	GlobalV::ofs_running << " "  <<std::setw(8) << "K-point" << std::setw(15) << "CG iter num" << std::setw(15) << "Time(Sec)"<< std::endl;
	GlobalV::ofs_running << std::setprecision(6) << std::setiosflags(ios::fixed) << std::setiosflags(ios::showpoint);

	for (int ik = 0;ik < GlobalC::kv.nks;ik++)
	{
		GlobalC::hm.hpw.init_k(ik);

        //===========================================
        // Conjugate-Gradient diagonalization
        // h_diag is the precondition matrix
        // h_diag(1:npw) = MAX( 1.0, g2kin(1:npw) );
        //===========================================
        if (precondition_type==1)
        {
            for (int ig = 0;ig < GlobalC::wf.npw; ig++)
			{
				h_diag[ig] = max(1.0, GlobalC::wf.g2kin[ig]);
				if(GlobalV::NPOL==2) h_diag[ig+GlobalC::wf.npwx] = h_diag[ig];
			}
        }
        else if (precondition_type==2)
        {
            for (int ig = 0;ig < GlobalC::wf.npw; ig++)
			{
				h_diag[ig] = 1 + GlobalC::wf.g2kin[ig] + sqrt( 1 + (GlobalC::wf.g2kin[ig] - 1) * (GlobalC::wf.g2kin[ig] - 1));
				if(GlobalV::NPOL==2) h_diag[ig+GlobalC::wf.npwx] = h_diag[ig];
			}
        }
		//h_diag can't be zero!  //zhengdy-soc
		if(GlobalV::NPOL==2)
		{
			for(int ig = GlobalC::wf.npw;ig < GlobalC::wf.npwx; ig++)
			{
				h_diag[ig] = 1.0;
				h_diag[ig+ GlobalC::wf.npwx] = 1.0;
			}
		}

		clock_t start=clock();

		double avg_iter_k = 0.0;
		GlobalC::hm.diagH_pw(istep, this->iter, ik, h_diag, avg_iter_k);

		avg_iter += avg_iter_k;

		GlobalC::en.print_band(ik);

		clock_t finish=clock();
		const double duration = static_cast<double>(finish - start) / CLOCKS_PER_SEC;


		GlobalV::ofs_running << " " << std::setw(8)
			<< ik+1 << std::setw(15)
			<< avg_iter_k << std::setw(15) << duration << std::endl;
	}//End K Loop


	if(GlobalV::BASIS_TYPE=="pw")
	{
		//		GlobalV::ofs_running << " avg_iteri " << avg_iter << std::endl;
		//Parallel_Reduce::reduce_double_allpool(avg_iter); //mohan fix bug 2012-06-05
		//		GlobalV::ofs_running << " avg_iter_after " << avg_iter << std::endl;
		//avg_iter /= static_cast<double>(GlobalC::kv.nkstot);
	}

	delete [] h_diag;

	ModuleBase::timer::tick("Elec_Stochastic","c_bands");

	return;
}
