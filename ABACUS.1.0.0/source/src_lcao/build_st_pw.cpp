#include "build_st_pw.h"
#include "../src_pw/global.h"

Build_ST_pw::Build_ST_pw()
{

}

Build_ST_pw::~Build_ST_pw()
{

}

// be called in Use_Hamilt_Matrix::calculate_STNR_k()
// FUNCTION: calculate the overlap and kinetic matrix
// in localized basis (expanded in plane wave basis).
void Build_ST_pw::set_ST(const int &ik, const char& dtype)
{
	switch (dtype)
	{
		case 'S':
		{
			for(int i=0; i<NLOCAL; i++)
			{
				const int mu = ParaO.trace_loc_row[i];
				if(mu < 0)continue;
				for(int j=0; j<NLOCAL; j++)
				{
					const int nu = ParaO.trace_loc_col[j];
					if(nu < 0)continue;
					
					complex<double> v = ZERO;
					for (int ig = 0; ig < kv.ngk[ik]; ig++) 
					{
						v += conj(wf.wanf2[ik](mu, ig)) * wf.wanf2[ik](nu, ig);
					}
					
	//				cout << "i=" << i << " j=" << j << " v=" << v << endl;
					//-----------------------------------
					// The results are saved in Sloc2.
					// 2 stands for k points.
					//-----------------------------------
					LM.Sloc2[ mu * ParaO.ncol + nu ] = v;
				}
			}
			break;
		}
		case 'T':
		{
			//------------------------------------
			//calculate the kinetic energy of ik.
			//------------------------------------
			wf.ekin(ik);

			for(int i=0; i<NLOCAL; i++)
			{
				const int mu = ParaO.trace_loc_row[i];
				if(mu < 0)continue;
				for(int j=0; j<NLOCAL; j++)
				{
					const int nu = ParaO.trace_loc_col[j];
					if(nu < 0)continue;
					
					complex<double> v = ZERO;
					for (int ig = 0; ig < kv.ngk[ik]; ig++) 
					{
						v += conj(wf.wanf2[ik](mu, ig)) * wf.wanf2[ik](nu, ig) * wf.g2kin[ig];
					}
					
	//				cout << "i=" << i << " j=" << j << " v=" << v << endl;
					//-----------------------------------------
					// The results are saved in Hloc_fixed2.
					//-----------------------------------------
					LM.Hloc_fixed2[ mu * ParaO.ncol + nu ] = v;
				}
			}
			break;
		}
	}

	return;
}

void Build_ST_pw::set_local(const int &ik)
{
	TITLE("Build_ST_pw","set_local");
	timer::tick("Build_ST_pw","set_local");
	assert(NLOCAL>0);
	assert(!GAMMA_ONLY_LOCAL);

    const int npw = kv.ngk[ik];
    complex<double> *psi_one = new complex<double>[npw];
    complex<double> *hpsi = new complex<double>[npw];
	complex<double> *psic = new complex<double>[pw.nrxx];
	int *fft_index = new int[npw];
	for(int ig=0; ig<npw; ig++)
	{
		fft_index[ig] = pw.ig2fftw[ wf.igk(ik, ig) ];
	}

//	ComplexMatrix vij(NLOCAL, NLOCAL);

	for(int i=0; i<NLOCAL; i++)
	{
		for(int ig=0; ig<npw; ig++)
		{
			psi_one[ig] = wf.wanf2[ik](i, ig);
		}

		ZEROS( psic, pw.nrxxs);
		// (1) set value
		for (int ig=0; ig< npw; ig++)
		{
			psic[ fft_index[ig]  ] = psi_one[ig];
		}

		// (2) fft to real space and doing things.
		pw.FFT_wfc.FFT3D( psic, 1);
		for (int ir=0; ir< pw.nrxx; ir++)
		{
			psic[ir] *= pot.vrs1[ir];
		}

		// (3) fft back to G space.
		pw.FFT_wfc.FFT3D( psic, -1);

		for(int ig=0; ig<npw; ig++)
		{
			hpsi[ig] = psic[ fft_index[ig] ];
		}

		for(int j=i; j<NLOCAL; j++)
		{
			complex<double> v = ZERO;
			for(int ig=0; ig<npw; ig++)
			{
				v += conj( wf.wanf2[ik](j,ig) ) * hpsi[ig];
			}
//			vij(j, i) = v;
			LM.set_HSk(j,i,v,'L');
			if(i!=j)
			{
				LM.set_HSk(i,j,conj(v),'L');
			}
		}
	}

//	out.printcm_norm("vij",vij,1.0e-5);

	delete[] fft_index;			
    delete[] psi_one;
    delete[] hpsi;
	timer::tick("Build_ST_pw","set_local");
	return;
}
