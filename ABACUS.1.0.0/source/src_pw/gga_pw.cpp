#include "gga_pw.h"
#include "global.h"
#include "xc_functional.h"
#include "../src_pw/myfunc.h"

// from gradcorr.f90
void GGA_PW::gradcorr(double &etxc, double &vtxc, matrix &v)
{
    if (xcf.igcx == 0  &&  xcf.igcc == 0)
    {
        return;
    }

	bool igcc_is_lyp = false;
	if( xcf.igcc == 3 || xcf.igcc == 7)
	{
		igcc_is_lyp = true;
	}

	assert(NSPIN>0);
	const double fac = 1.0/ NSPIN;

	// doing FFT to get rho in G space: rhog1 
    chr.set_rhog(chr.rho[0], chr.rhog[0]);
	if(NSPIN==2)//mohan fix bug 2012-05-28
	{
		chr.set_rhog(chr.rho[1], chr.rhog[1]);
	}
    chr.set_rhog(chr.rho_core, chr.rhog_core);
		
	// sum up (rho_core+rho) for each spin in real space
	// and reciprocal space.
	double* rhotmp1;
	double* rhotmp2;
	complex<double>* rhogsum1;
	complex<double>* rhogsum2;
	double** gdr1;
	double** gdr2;
	double** h1;
	double** h2;
	
	// for spin unpolarized case, 
	// calculate the gradient of (rho_core+rho) in reciprocal space.
	rhotmp1 = new double[pw.nrxx];
	rhogsum1 = new complex<double>[pw.ngmc];
	ZEROS(rhotmp1, pw.nrxx);
	ZEROS(rhogsum1, pw.ngmc);
	for(int ir=0; ir<pw.nrxx; ir++) rhotmp1[ir] = chr.rho[0][ir] + fac * chr.rho_core[ir];
	for(int ig=0; ig<pw.ngmc; ig++) rhogsum1[ig] = chr.rhog[0][ig] + fac * chr.rhog_core[ig];

	gdr1 = new double*[3];
	h1 = new double*[3];
	for(int i=0; i<3; i++) 
	{
		gdr1[i] = new double[pw.nrxx];
		h1[i] = new double[pw.nrxx];	
		ZEROS(gdr1[i], pw.nrxx);	
		ZEROS(h1[i], pw.nrxx);
	}
	
	GGA_PW::grad_rho( rhogsum1 , gdr1 );

	// for spin polarized case;
	// calculate the gradient of (rho_core+rho) in reciprocal space.
	if(NSPIN==2)
	{
		rhotmp2 = new double[pw.nrxx];
		rhogsum2 = new complex<double>[pw.ngmc];
		ZEROS(rhotmp2, pw.nrxx);
		ZEROS(rhogsum2, pw.ngmc);
		for(int ir=0; ir<pw.nrxx; ir++) rhotmp2[ir] = chr.rho[1][ir] + fac * chr.rho_core[ir];
		for(int ig=0; ig<pw.ngmc; ig++) rhogsum2[ig] = chr.rhog[1][ig] + fac * chr.rhog_core[ig];

		gdr2 = new double*[3];
		h2 = new double*[3];
		for(int i=0; i<3; i++) 
		{
			gdr2[i] = new double[pw.nrxx];	
			h2[i] = new double[pw.nrxx];	
			ZEROS(gdr2[i], pw.nrxx);
			ZEROS(h2[i], pw.nrxx);
		}
		
		GGA_PW::grad_rho( rhogsum2 , gdr2 );
	}

	// for test
	/*
	double sum[6]={0,0,0,0,0,0};
	for(int ir=0; ir<pw.nrxx; ir++)
	{
		sum[0] += abs(gdr1[0][ir]);
		sum[1] += abs(gdr1[1][ir]);
		sum[2] += abs(gdr1[2][ir]);	
		sum[3] += abs(rhotmp1[ir]);	
		sum[4] += rhotmp1[ir]*rhotmp1[ir];	
	}
	*/
	
	/*
	cout << "\n sum grad 1= " << sum[0] << " "  << sum[1] << " " << sum[2] << endl;
	cout << " sum rho = " << sum[3] << " "  << sum[4] << endl;
	ZEROS(sum,6);
	for(int ir=0; ir<pw.nrxx; ir++)
	{
		sum[0] += abs(gdr2[0][ir]);
		sum[1] += abs(gdr2[1][ir]);
		sum[2] += abs(gdr2[2][ir]);	
		sum[3] += abs(rhotmp2[ir]);	
		sum[4] += rhotmp2[ir]*rhotmp2[ir];	
	}
	cout << "\n sum grad 2= " << sum[0] << " "  << sum[1] << " " << sum[2] << endl;
	cout << " sum rho = " << sum[3] << " "  << sum[4] << endl;
	*/
	
	const double epsr = 1.0e-6;
	const double epsg = 1.0e-10;

	double grho2a = 0.0;
	double grho2b = 0.0;
	double sx = 0.0;
	double sc = 0.0;
	double v1x = 0.0;
	double v2x = 0.0;
	double v1c = 0.0;
	double v2c = 0.0;
	double vtxcgc = 0.0;
	double etxcgc = 0.0;

	if(NSPIN==1)
	{
		double segno;
		for(int ir=0; ir<pw.nrxx; ir++)
		{
			const double arho = std::abs( rhotmp1[ir] );
			for(int i=0; i<3; i++)
			{
				h1[i][ir]=0.0;
			}
			if(arho > epsr)
			{
				grho2a = 0.0;
				for(int i=0; i<3; i++)
				{
					grho2a += gdr1[i][ir]*gdr1[i][ir];
				}
				if( grho2a > epsg )
				{
					if( rhotmp1[ir] >= 0.0 ) segno = 1.0;
					if( rhotmp1[ir] < 0.0 ) segno = -1.0;
					
					XC_Functional::gcxc( arho, grho2a, sx, sc, v1x, v2x, v1c, v2c);
					
					// first term of the gradient correction:
					// D(rho*Exc)/D(rho)
					v(0, ir) += e2 * ( v1x + v1c );
					
					// h contains
					// D(rho*Exc) / D(|grad rho|) * (grad rho) / |grad rho|
					for(int j=0; j<3; j++)
					{
						h1[j][ir] = e2 * ( v2x + v2c ) * gdr1[j][ir];
					}
					
					vtxcgc += e2*( v1x + v1c ) * ( rhotmp1[ir] - chr.rho_core[ir] );
					etxcgc += e2*( sx + sc ) * segno;
				}
			} // end arho > epsr
		}
	}// end NSPIN == 1
	else // spin polarized case
	{
		double v1cup = 0.0;
		double v1cdw = 0.0;
		double v2cup = 0.0;
		double v2cdw = 0.0;
		double v1xup = 0.0;
		double v1xdw = 0.0;
		double v2xup = 0.0;
		double v2xdw = 0.0;
		double v2cud = 0.0;
		double v2c = 0.0;
		for(int ir=0; ir<pw.nrxx; ir++)
		{
			double rh = rhotmp1[ir] + rhotmp2[ir]; 
			grho2a = 0.0;
			grho2b = 0.0;
			for(int i=0; i<3; i++)
			{
				grho2a += gdr1[i][ir]*gdr1[i][ir];
				grho2b += gdr2[i][ir]*gdr2[i][ir];
			}
			//XC_Functional::gcx_spin();
			gcx_spin(rhotmp1[ir], rhotmp2[ir], grho2a, grho2b,
				sx, v1xup, v1xdw, v2xup, v2xdw);
			
			if(rh > epsr)
			{
				if(igcc_is_lyp)
				{
					WARNING_QUIT("gga_pw","igcc_is_lyp is not available now.");
				}
				else
				{
					double zeta = ( rhotmp1[ir] - rhotmp2[ir] ) / rh;
//					if(nspin==4)
					double grh2 = 0.0;
					for(int i=0; i<3; i++)
					{
						grh2 += pow((gdr1[i][ir]+gdr2[i][ir]),2);
					}
					//XC_Functional::gcc_spin(rh, zeta, grh2, sc, v1cup, v1cdw, v2c);
					gcc_spin(rh, zeta, grh2, sc, v1cup, v1cdw, v2c);
					v2cup = v2c;
					v2cdw = v2c;
					v2cud = v2c;
				}
			}
			else
			{
				sc = 0.0;
				v1cup = 0.0;
				v1cdw = 0.0;
				v2c = 0.0;
				v2cup = 0.0;
				v2cdw = 0.0;
				v2cud = 0.0;
			}


			// first term of the gradient correction : D(rho*Exc)/D(rho)
			v(0,ir) = v(0,ir) + e2 * ( v1xup + v1cup );
			v(1,ir) = v(1,ir) + e2 * ( v1xdw + v1cdw );

//			continue; //mohan tmp
			
			// h contains D(rho*Exc)/D(|grad rho|) * (grad rho) / |grad rho|
			for(int ipol=0; ipol<3; ++ipol)
			{
				double grup = gdr1[ipol][ir];
				double grdw = gdr2[ipol][ir];
				h1[ipol][ir] = e2 * ( ( v2xup + v2cup ) * grup + v2cud * grdw );
				h2[ipol][ir] = e2 * ( ( v2xdw + v2cdw ) * grdw + v2cud * grup );
			}	

			vtxcgc = vtxcgc + e2 * ( v1xup + v1cup ) * ( rhotmp1[ir] - chr.rho_core[ir] * fac );
			vtxcgc = vtxcgc + e2 * ( v1xdw + v1cdw ) * ( rhotmp2[ir] - chr.rho_core[ir] * fac );
			etxcgc = etxcgc + e2 * ( sx + sc );
			

		}// end ir

	}

	//cout << "\n vtxcgc=" << vtxcgc;
	//cout << "\n etxcgc=" << etxcgc << endl;

	for(int ir=0; ir<pw.nrxx; ir++) rhotmp1[ir] -= fac * chr.rho_core[ir];
	if(NSPIN==2) for(int ir=0; ir<pw.nrxx; ir++) rhotmp2[ir] -= fac * chr.rho_core[ir];
	
	// second term of the gradient correction :
	// \sum_alpha (D / D r_alpha) ( D(rho*Exc)/D(grad_alpha rho) )

	// dh is in real sapce.
	double* dh = new double[pw.nrxx];

	for(int is=0; is<NSPIN; is++)
	{
		ZEROS(dh, pw.nrxx);
		if(is==0)GGA_PW::grad_dot(h1,dh);
		if(is==1)GGA_PW::grad_dot(h2,dh);

		for(int ir=0; ir<pw.nrxx; ir++)
		{
			v(is, ir) -= dh[ir];
			// mohan test
//			if( v(is,ir)!=0.0 )
//			{
//				cout << " v=" << v(is,ir) << endl;
//			}
		}
		
		double sum = 0.0;
		if(is==0)
		{
			for(int ir=0; ir<pw.nrxx; ir++)
			{
				sum += dh[ir] * rhotmp1[ir];
			}
		}
		if(is==1)
		{
			for(int ir=0; ir<pw.nrxx; ir++)
			{
				sum += dh[ir] * rhotmp2[ir];
			}
		}
		

		vtxcgc -= sum;
	}

	delete[] dh;

	vtxc += vtxcgc;
	etxc += etxcgc;

	//mohan test
//	vtxc = 0.0;
//	etxc = 0.0;
	
	// deacllocate
	delete[] rhotmp1;
	delete[] rhogsum1;
	for(int i=0; i<3; i++) delete[] gdr1[i];
	for(int i=0; i<3; i++) delete[] h1[i];
	delete[] gdr1;
	delete[] h1;

	if(NSPIN==2)
	{
		delete[] rhotmp2;
		delete[] rhogsum2;
		for(int i=0; i<3; i++) delete[] gdr2[i];
		for(int i=0; i<3; i++) delete[] h2[i];
		delete[] gdr2;
		delete[] h2;
	}

	return;
}

void GGA_PW::grad_rho( const complex<double> *rhog, double **gdr )
{
	complex<double> *gdrtmpg = new complex<double>[pw.ngmc];
	ZEROS(gdrtmpg, pw.ngmc);

	complex<double> *Porter = UFFT.porter;

	// the formula is : rho(r)^prime = \int iG * rho(G)e^{iGr} dG
	for(int ig=0; ig<pw.ngmc; ig++)
	{
		gdrtmpg[ig] = IMAG_UNIT * rhog[ig];
	}

	for(int i=0; i<3; i++)
	{
		// calculate the charge density gradient in reciprocal space.
		ZEROS(Porter, pw.nrxx);
		if(i==0)for(int ig=0; ig<pw.ngmc; ig++)Porter[ pw.ig2fftc[ig] ] = gdrtmpg[ig]* complex<double>(pw.gcar[ig].x, 0.0);
		if(i==1)for(int ig=0; ig<pw.ngmc; ig++)Porter[ pw.ig2fftc[ig] ] = gdrtmpg[ig]* complex<double>(pw.gcar[ig].y, 0.0);
		if(i==2)for(int ig=0; ig<pw.ngmc; ig++)Porter[ pw.ig2fftc[ig] ] = gdrtmpg[ig]* complex<double>(pw.gcar[ig].z, 0.0);
		
		// bring the gdr from G --> R
		pw.FFT_chg.FFT3D(Porter, 1);

		// remember to multily 2pi/a0, which belongs to G vectors.
		double sum = 0.0;
		for(int ir=0; ir<pw.nrxx; ir++)
		{
			gdr[i][ir] = Porter[ir].real() * ucell.tpiba;
			sum += gdr[i][ir];
		}
	}

	delete[] gdrtmpg;
	return;
}


void GGA_PW::grad_dot(double **h, double *dh)
{
	complex<double> *aux = new complex<double>[pw.nrxx];
	complex<double> *gaux = new complex<double>[pw.ngmc];
	ZEROS(gaux, pw.ngmc);
	
	complex<double> tmp; 
	for(int i=0; i<3; i++)
	{
		ZEROS(aux, pw.nrxx);
		for(int ir=0; ir<pw.nrxx; ir++)
		{
			aux[ir] = complex<double>( h[i][ir], 0.0);
		}
		// bring to G space.
		pw.FFT_chg.FFT3D(aux, -1);
		
		switch(i)
		{
			// the only difference is 'pw.g'
			case 0:
			for(int ig=0; ig<pw.ngmc; ig++)
			{
				tmp = pw.gcar[ig].x * IMAG_UNIT;
				gaux[ig] += tmp * aux[ pw.ig2fftc[ig] ]; 
			}
			break;

			case 1:	
			for(int ig=0; ig<pw.ngmc; ig++)
			{
				tmp = pw.gcar[ig].y * IMAG_UNIT;
				gaux[ig] += tmp * aux[ pw.ig2fftc[ig] ]; 
			}
			break;

			case 2:
			for(int ig=0; ig<pw.ngmc; ig++)
			{
				tmp = pw.gcar[ig].z * IMAG_UNIT;
				gaux[ig] += tmp * aux[ pw.ig2fftc[ig] ]; 
			}
			break;
		}
	}


	ZEROS(aux, pw.nrxx);
	for(int ig=0; ig<pw.ngmc; ig++)
	{
		aux[ pw.ig2fftc[ig] ] = gaux[ig];
	}

	// bring back to R space
	pw.FFT_chg.FFT3D(aux, 1);
	
	for(int ir=0; ir<pw.nrxx; ir++)
	{
		dh[ir] = aux[ir].real() * ucell.tpiba;
	}
	
	delete[] aux;	
	delete[] gaux; //mohan fix 2012-04-02
	return;
}
