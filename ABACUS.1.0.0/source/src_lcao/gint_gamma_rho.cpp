#include "gint_gamma.h"
#include "grid_technique.h"
#include "lcao_orbitals.h"
#include "../src_pw/global.h"
#include "blas_interface.h"

inline void setVindex(const int ncyz, const int ibx, const int jby, const int kbz, int* vindex)
{				
	int bindex = 0;
	// z is the fastest, 
	for(int ii=0; ii<pw.bx; ii++)
	{
		const int ipart = (ibx + ii) * ncyz + kbz;
		for(int jj=0; jj<pw.by; jj++)
		{
			const int jpart = (jby + jj) * pw.nczp + ipart;
			for(int kk=0; kk<pw.bz; kk++)
			{
				vindex[bindex] = kk + jpart; 
				++bindex;
			}
		}
	}
}

inline void cal_psir_ylm(int size, int grid_index, double delta_r, double phi, 
                        double* mt, double*** dr, double** distance, 
                        const Numerical_Orbital_Lm* pointer, double* ylma,
                        int* colidx, int* block_iw, int* bsize, 
                        double** psir_ylm)
{
	//timer::tick("Gint_Gamma","cal_psir_ylm",'J');
	colidx[0]=0;
	for (int id=0; id<size; id++)
	{
		// there are two parameters we want to know here:
		// in which bigcell of the meshball the atom in?
		// what's the cartesian coordinate of the bigcell?
		const int mcell_index = GridT.bcell_start[grid_index] + id;
		const int imcell = GridT.which_bigcell[mcell_index];

		int iat = GridT.which_atom[mcell_index];

		const int it = ucell.iat2it[ iat ];
		const int ia = ucell.iat2ia[ iat ];
		const int start = ucell.itiaiw2iwt(it, ia, 0);
		block_iw[id]=GridT.trace_lo[start];
		Atom* atom = &ucell.atoms[it];
		bsize[id]=atom->nw;
		colidx[id+1]=colidx[id]+atom->nw;
		// meshball_positions should be the bigcell position in meshball
		// to the center of meshball.
		// calculated in cartesian coordinates
		// the vector from the grid which is now being operated to the atom position.
		// in meshball language, is the vector from imcell to the center cel, plus
		// tau_in_bigcell.
		mt[0] = GridT.meshball_positions[imcell][0] - GridT.tau_in_bigcell[iat][0];
		mt[1] = GridT.meshball_positions[imcell][1] - GridT.tau_in_bigcell[iat][1];
		mt[2] = GridT.meshball_positions[imcell][2] - GridT.tau_in_bigcell[iat][2];

		for(int ib=0; ib<pw.bxyz; ib++)
		{
			double *p=&psir_ylm[ib][colidx[id]];
			// meshcell_pos: z is the fastest
			dr[ib][id][0] = GridT.meshcell_pos[ib][0] + mt[0]; 
			dr[ib][id][1] = GridT.meshcell_pos[ib][1] + mt[1]; 
			dr[ib][id][2] = GridT.meshcell_pos[ib][2] + mt[2]; 	

			distance[ib][id] = std::sqrt(dr[ib][id][0]*dr[ib][id][0] + dr[ib][id][1]*dr[ib][id][1] + dr[ib][id][2]*dr[ib][id][2]);
			if(distance[ib][id] > ORB.Phi[it].getRcut()) 
			{
				ZEROS(p, bsize[id]);
				continue;
			}
			
			//if(distance[id] > GridT.orbital_rmax) continue;
			//	Ylm::get_ylm_real(this->nnn[it], this->dr[id], ylma);
			if (distance[ib][id] < 1.0E-9) distance[ib][id] += 1.0E-9;
			
			Ylm::sph_harm (	ucell.atoms[it].nwl,
					dr[ib][id][0] / distance[ib][id],
					dr[ib][id][1] / distance[ib][id],
					dr[ib][id][2] / distance[ib][id],
					ylma);
			// these parameters are about interpolation
			// because once we know the distance from atom to grid point,
			// we can get the parameters we need to do interpolation and
			// store them first!! these can save a lot of effort.
			const double position = distance[ib][id] / delta_r;
			/*
				this->iq[id] = static_cast<int>(position);
				this->x0[id] = position - static_cast<double>(iq[id]);
				this->x1[id] = 1.0 - x0[id];
				this->x2[id] = 2.0 - x0[id];
				this->x3[id] = 3.0 - x0[id];
				this->x12[id] = x1[id]*x2[id] / 6.0;
				this->x03[id] = x0[id]*x3[id] / 2.0;
			 */
			int ip;
			double dx, dx2, dx3;
			double c1, c2, c3, c4;

			ip = static_cast<int>(position);
			dx = position - ip;
			dx2 = dx * dx;
			dx3 = dx2 * dx;

			c3 = 3.0*dx2-2.0*dx3;
			c1 = 1.0-c3;
			c2 = (dx-2.0*dx2+dx3)*delta_r;
			c4 = (dx3-dx2)*delta_r;

			for (int iw=0; iw< atom->nw; ++iw, ++p)
			{
				if ( atom->iw2_new[iw] )
				{
					pointer = &ORB.Phi[it].PhiLN(
							atom->iw2l[iw],
							atom->iw2n[iw]);
					phi = c1*pointer->psi_uniform[ip]+c2*pointer->dpsi_uniform[ip]
						+ c3*pointer->psi_uniform[ip+1] + c4*pointer->dpsi_uniform[ip+1];
				}
				*p = phi * ylma[atom->iw2_ylm[iw]];
			}
		}// end ib
	}// end id
	//timer::tick("Gint_Gamma","cal_psir_ylm",'J');
	//OUT(ofs_running, "psir_ylm is done");
}

// can be done by GPU
inline void cal_band_rho(int size, int LD_pool, int* block_iw, int* bsize, int* colidx,
                        double ** psir_ylm, double **psir_DM, double* psir_DM_pool, 
                        int* vindex
)
{
	//parameters for dsymm, dgemm and ddot
	char side='L', uplo='U';
	char transa='N', transb='N';
	double alpha_symm=1, alpha_gemm=2, beta=1;	
	int inc=1;
	
	for(int is=0; is<NSPIN; ++is)
	{
		ZEROS(psir_DM_pool, pw.bxyz*LD_pool);
		//xiaohui move 2015-09-25, fix lcao+k bug;
		//timer::tick("Gint_Gamma","cal_band_psir",'J');
		for (int ia1=0; ia1<size; ++ia1)
		{
			//OUT(ofs_running, "calculate psir_DM, atom", ia1);
			int iw1_lo=block_iw[ia1];
			//ia1==ia2, diagonal part
			//cblas_dsymm(CblasRowMajor, CblasRight, CblasUpper, pw.bxyz, bsize[ia1], 1, &LOC.DM[is][iw1_lo][iw1_lo], GridT.lgd, &psir_ylm[0][colidx[ia1]], LD_pool, 1, &psir_DM[0][colidx[ia1]], LD_pool);
			dsymm_(&side, &uplo, &bsize[ia1], &pw.bxyz, &alpha_symm, &LOC.DM[is][iw1_lo][iw1_lo], &GridT.lgd, &psir_ylm[0][colidx[ia1]], &LD_pool, &beta, &psir_DM[0][colidx[ia1]], &LD_pool);
			//ia2>ia1
			
			//OUT(ofs_running, "diagonal part of psir_DM done");
			for (int ia2=ia1+1; ia2<size; ++ia2)
			{
				int iw2_lo=block_iw[ia2];
				dgemm_(&transa, &transb, &bsize[ia2], &pw.bxyz, &bsize[ia1], &alpha_gemm, &LOC.DM[is][iw1_lo][iw2_lo], &GridT.lgd, &psir_ylm[0][colidx[ia1]], &LD_pool, &beta, &psir_DM[0][colidx[ia2]], &LD_pool);
				//OUT(ofs_running, "upper triangle part of psir_DM done, atom2", ia2);
			}// ia2
		} // ia1
		//xiaohui move 2015-09-25, fix lcao+k bug;
		//timer::tick("Gint_Gamma","cal_band_psir",'J');
	
		// calculate rho	
		//xiaohui move 2015-09-25, fix lcao+k bug;
		//timer::tick("Gint_Gamma","cal_band_rho",'J');	
		double *rhop = chr.rho[is];
		for(int ib=0; ib<pw.bxyz; ++ib)
		{
			//double r = cblas_ddot(colidx[size], psir_ylm[ib], 1, psir_DM[ib], 1);
			double r = ddot_(&colidx[size], psir_ylm[ib], &inc, psir_DM[ib], &inc);
			const int grid = vindex[ib];
			rhop[ grid ] += r;
		}
		//xiaohui move 2015-09-25, fix lcao+k bug;
		//timer::tick("Gint_Gamma","cal_band_rho",'J');	
	}
}

double Gint_Gamma::cal_rho(void)
{
	TITLE("Gint_Gamma","cal_rho");
	timer::tick("Gint_Gamma","cal_rho",'F');

	this->job = cal_charge;
	this->save_atoms_on_grid(GridT);

	double ne = 0.0;
	if(BFIELD)
	{
		this->gamma_charge_B();
	}
	else
	{
		ne = this->gamma_charge();
	}
	timer::tick("Gint_Gamma","cal_rho",'F');
	return ne;
}

#include "bfield.h"
double Gint_Gamma::gamma_charge_B(void)
{
	TITLE("Grid_Integral","gamma_charge_B");
	timer::tick("Gint_Gamma","gamma_charge_B",'I');
	
	//get cartesian lattice vectors, in bohr, zhiyuan add 2012-01-13//
	double latvec1[3],latvec2[3],latvec3[3];
	latvec1[0]=ucell.latvec.e11*ucell.lat0;
	latvec1[1]=ucell.latvec.e12*ucell.lat0;
	latvec1[2]=ucell.latvec.e13*ucell.lat0;
	latvec2[0]=ucell.latvec.e21*ucell.lat0;
	latvec2[1]=ucell.latvec.e22*ucell.lat0;
	latvec2[2]=ucell.latvec.e23*ucell.lat0;
	latvec3[0]=ucell.latvec.e31*ucell.lat0;
	latvec3[1]=ucell.latvec.e32*ucell.lat0;
	latvec3[2]=ucell.latvec.e33*ucell.lat0;

	//This is to store the position of grids, zhiyuan add 2012-01-13//
	double **Rbxyz = new double *[pw.bxyz]; // bxyz*3//
	for(int i=0;i<pw.bxyz;i++)
	{
		Rbxyz[i]=new double[3];
		ZEROS(Rbxyz[i], 3);
	}

	//This is for the phase, zhiyuan add 2012-01-13//
	double A2_ms_A1[3]={0,0,0}; //used to store A2-A1//

	// it's a uniform grid to save orbital values, so the delta_r is a constant.
	const double delta_r = ORB.dr_uniform;
	const Numerical_Orbital_Lm* pointer;

	// allocate 1
	int nnnmax=0;
	for(int T=0; T<ucell.ntype; T++)
	{
		nnnmax = max(nnnmax, nnn[T]);
	}

	double*** dr; // vectors between atom and grid: [bxyz, maxsize, 3]
	double** distance; // distance between atom and grid: [bxyz, maxsize]
	double*** psir_ylm;
	bool** cal_flag;
	if(max_size!=0)
	{
		dr = new double**[pw.bxyz];
		distance = new double*[pw.bxyz];
		psir_ylm = new double**[pw.bxyz];
		cal_flag = new bool*[pw.bxyz];

		for(int i=0; i<pw.bxyz; i++)
		{
			dr[i] = new double*[max_size];
			distance[i] = new double[max_size];
			psir_ylm[i] = new double*[max_size];
			cal_flag[i] = new bool[max_size];

			ZEROS(distance[i], max_size);
			ZEROS(cal_flag[i], max_size);

			for(int j=0; j<max_size; j++)
			{
				dr[i][j] = new double[3];
				psir_ylm[i][j] = new double[ucell.nwmax];
				ZEROS(dr[i][j],3);
				ZEROS(psir_ylm[i][j],ucell.nwmax);
			}
		}
	}

	double* ylma = new double[nnnmax]; // Ylm for each atom: [bxyz, nnnmax]
	ZEROS(ylma, nnnmax);

	double mt[3]={0,0,0};
	double v1 = 0.0;
	int* vindex=new int[pw.bxyz];
	ZEROS(vindex, pw.bxyz);
	double phi=0.0;

	const int nbx = GridT.nbx;
	const int nby = GridT.nby;
	const int nbz_start = GridT.nbzp_start;
	const int nbz = GridT.nbzp;

	const int ncyz = pw.ncy*pw.nczp; 
	for (int i=0; i<nbx; i++)
	{
		const int ibx = i*pw.bx; 
		for (int j=0; j<nby; j++)
		{
			const int jby = j*pw.by;
			for (int k=nbz_start; k<nbz_start+nbz; k++) // FFT grid
			{
				const int kbz = k*pw.bz-pw.nczp_start; 
				this->grid_index = (k-nbz_start) + j * nbz + i * nby * nbz;

				// get the value: how many atoms has orbital value on this grid.
				const int size = GridT.how_many_atoms[ this->grid_index ];
				if(size==0) continue;

				// (1) initialized the phi * Ylm.
				for (int id=0; id<size; id++)
				{
					// there are two parameters we want to know here:
					// in which bigcell of the meshball the atom in?
					// what's the cartesian coordinate of the bigcell?
					const int mcell_index = GridT.bcell_start[grid_index] + id;
					const int imcell = GridT.which_bigcell[mcell_index];

					int iat = GridT.which_atom[mcell_index];

					const int it = ucell.iat2it[ iat ];
					const int ia = ucell.iat2ia[ iat ];

					// meshball_positions should be the bigcell position in meshball
					// to the center of meshball.
					// calculated in cartesian coordinates
					// the vector from the grid which is now being operated to the atom position.
					// in meshball language, is the vector from imcell to the center cel, plus
					// tau_in_bigcell.
					mt[0] = GridT.meshball_positions[imcell][0] - GridT.tau_in_bigcell[iat][0];
					mt[1] = GridT.meshball_positions[imcell][1] - GridT.tau_in_bigcell[iat][1];
					mt[2] = GridT.meshball_positions[imcell][2] - GridT.tau_in_bigcell[iat][2];

					for(int ib=0; ib<pw.bxyz; ib++)
					{
						// meshcell_pos: z is the fastest
						dr[ib][id][0] = GridT.meshcell_pos[ib][0] + mt[0];
						dr[ib][id][1] = GridT.meshcell_pos[ib][1] + mt[1];
						dr[ib][id][2] = GridT.meshcell_pos[ib][2] + mt[2];

						distance[ib][id] = std::sqrt(dr[ib][id][0]*dr[ib][id][0] + dr[ib][id][1]*dr[ib][id][1] + dr[ib][id][2]*dr[ib][id][2]);
						if(distance[ib][id] <= ORB.Phi[it].getRcut())
						{
							cal_flag[ib][id]=true;
						}
						else
						{
							cal_flag[ib][id]=false;
							continue;
						}
						
						//if(distance[id] > GridT.orbital_rmax) continue;
						//  Ylm::get_ylm_real(this->nnn[it], this->dr[id], ylma);
						if (distance[ib][id] < 1.0E-9) distance[ib][id] += 1.0E-9;

						Ylm::sph_harm ( ucell.atoms[it].nwl,
								dr[ib][id][0] / distance[ib][id],
								dr[ib][id][1] / distance[ib][id],
								dr[ib][id][2] / distance[ib][id],
								ylma);
						// these parameters are about interpolation
						// because once we know the distance from atom to grid point,
						// we can get the parameters we need to do interpolation and
						// store them first!! these can save a lot of effort.
						const double position = distance[ib][id] / delta_r;
						/*
							this->iq[id] = static_cast<int>(position);
							this->x0[id] = position - static_cast<double>(iq[id]);
							this->x1[id] = 1.0 - x0[id];
							this->x2[id] = 2.0 - x0[id];
							this->x3[id] = 3.0 - x0[id];
							this->x12[id] = x1[id]*x2[id] / 6.0;
							this->x03[id] = x0[id]*x3[id] / 2.0;
						 */
						int ip;
						double dx, dx2, dx3;
						double c1, c2, c3, c4;

				
						ip = static_cast<int>(position);
						dx = position - ip;
						dx2 = dx * dx;
						dx3 = dx2 * dx;

						c3 = 3.0*dx2-2.0*dx3;
						c1 = 1.0-c3;
						c2 = (dx-2.0*dx2+dx3)*delta_r;
						c4 = (dx3-dx2)*delta_r;

						//	  int ip = this->iq[id];
						//	  double A = ip+1.0-position/delta_r;
						//	  double B = 1.0-A;
						//	  double coef1 = (A*A*A-A)/6.0*delta_r*delta_r;
						//	  double coef2 = (B*B*B-B)/6.0*delta_r*delta_r;

						Atom* atom1 = &ucell.atoms[it];
						for (int iw=0; iw< atom1->nw; iw++)
						{
							if ( atom1->iw2_new[iw] )
							{
								pointer = &ORB.Phi[it].PhiLN(
										atom1->iw2l[iw],
										atom1->iw2n[iw]);
								phi = c1*pointer->psi_uniform[ip]+c2*pointer->dpsi_uniform[ip]
									+ c3*pointer->psi_uniform[ip+1] + c4*pointer->dpsi_uniform[ip+1];
							}
							psir_ylm[ib][id][iw] = phi * ylma[atom1->iw2_ylm[iw]];
							//psir_ylm[ib][id][iw] = 1;//for test
						}
					}// end ib
				}// end id

				setVindex(ncyz, ibx, jby, kbz, vindex);
				/*
				int bindex = 0;
				// z is the fastest,
				for(int ii=0; ii<pw.bx; ii++)
				{
					const int ipart = (i*pw.bx + ii) * pw.ncy*pw.nczp;
					for(int jj=0; jj<pw.by; jj++)
					{
						const int jpart = (j*pw.by + jj) * pw.nczp;
						for(int kk=0; kk<pw.bz; kk++)
						{
							vindex[bindex] = (k*pw.bz + kk-pw.nczp_start) + jpart + ipart;
							//  assert(vindex[bindex] < pw.nrxx);
							++bindex;
						}
					}
				}
				*/
	
				//get the information about position r of grid//
				int index=0;
				double rA[3]={0,0,0};
				for(int ii=0; ii<pw.bx; ii++)
				{
					const double iii = (i*pw.bx + ii)/(double)pw.ncx;
					for(int jj=0; jj<pw.by; jj++)
					{
						const double jjj = (j*pw.by + jj)/(double)pw.ncy;
						for(int kk=0; kk<pw.bz; kk++)
						{
							const double kkk = (k*pw.bz + kk)/(double)pw.ncz;
							rA[0]=iii*latvec1[0]+jjj*latvec2[0]+kkk*latvec3[0];
							rA[1]=iii*latvec1[1]+jjj*latvec2[1]+kkk*latvec3[1];
							rA[2]=iii*latvec1[2]+jjj*latvec2[2]+kkk*latvec3[2];
							Rbxyz[index][0]=rA[0];
							Rbxyz[index][1]=rA[1];
							Rbxyz[index][2]=rA[2];
							index++;
						}
					}
				}

				for (int ia1=0; ia1<size; ia1++)
				{
					const int mcell_index1 = GridT.bcell_start[grid_index] + ia1;
					const int iat1=GridT.which_atom[mcell_index1];
					const int T1 = ucell.iat2it[iat1];
					Atom *atom1 = &ucell.atoms[T1];
					const int I1 = ucell.iat2ia[iat1];
					// get the start index of local orbitals.
					const int start1 = ucell.itiaiw2iwt(T1, I1, 0);

					// attention! assume all rcut are same for this atom type now.
					//if (distance[ia1] > ORB.Phi[T1].getRcut())continue;

					//for(int ia2=ia1; ia2<size; ia2++)
					for (int ia2=0; ia2<size; ia2++)
					{
						const int mcell_index2 = GridT.bcell_start[grid_index] + ia2;
						const int iat2=GridT.which_atom[mcell_index2];
						const int T2 = ucell.iat2it[iat2];
						Atom *atom2 = &ucell.atoms[T2];
						const int I2 = ucell.iat2ia[iat2];
						const int start2 = ucell.itiaiw2iwt(T2, I2, 0);

						//get A2-A1//
						A2_ms_A1[0]=bfid.A_of_Atom[iat2][0]-bfid.A_of_Atom[iat1][0];
						A2_ms_A1[1]=bfid.A_of_Atom[iat2][1]-bfid.A_of_Atom[iat1][1];
						A2_ms_A1[2]=bfid.A_of_Atom[iat2][2]-bfid.A_of_Atom[iat1][2];
						// only do half part of matrix(including diago part)
						// for T2 > T1, we done all elements, for T2 == T1,
						// we done about half.
						if (T2 >= T1)
						{
							for(int is=0; is<NSPIN; is++)
							{
								double *rhop = chr.rho[is];
								for (int ib=0; ib<pw.bxyz; ib++)
								{
									if(cal_flag[ib][ia1] && cal_flag[ib][ia2])
									{
										//get phase//
										complex<double> phase=0;
										phase=bfid.cal_phase(A2_ms_A1,Rbxyz[ib],-1);
										int iw1_lo = GridT.trace_lo[start1];
										double* psi1 = psir_ylm[ib][ia1];
										double* psi2 = psir_ylm[ib][ia2];
										double tmp = 0.0;

										// how many orbitals in this type: SZ or DZP or TZP...
										for (int iw=0; iw< atom1->nw; iw++, ++iw1_lo)
										{
											v1=psi1[iw]+psi1[iw];
											int iw2_lo = GridT.trace_lo[start2];

											//zhiyuan add 2012-01-03//
											complex<double> *DMp_B;
											DMp_B= &LOC.DM_B[is][iw1_lo][iw2_lo];

											double *psi2p = psi2;
											double *psi2p_end = psi2 + atom2->nw;

											for (; psi2p < psi2p_end;
													++iw2_lo, ++psi2p, ++DMp_B)
											{
												if ( iw1_lo > iw2_lo)
												{
													//	  ++iw2_lo;
													continue;
												}
												// for diago part, the charge density be accumulated once.
												// for off-diago part, the charge density be accumulated twice.
												// (easy to understand, right? because we only calculate half
												// of the matrix).

												//  double tmp = v1*psir_ylm[ib][ia2][iw2]*DMp[0];
												double tmp1=(phase*DMp_B[0]).real();
												double tmp2=tmp1*v1*psi2p[0];

												if (iw1_lo<iw2_lo)
												{
													tmp += tmp2;
												}
												else
												{
													tmp += tmp2/2.0;
												}
												//  ++iw2_lo;
											}//iw2
										}//iw
										const int grid = vindex[ib];
										rhop[ grid ] += tmp;
									}// cal_flag
								}//ib
							}
						}//T
					}// ia2
				}// ia1
			}// k
		}// j
	}// i

	delete[] vindex;
	delete[] ylma;

	if(max_size!=0)
	{
		for(int i=0; i<pw.bxyz; i++)
		{
			for(int j=0; j<max_size; j++)
			{
				delete[] dr[i][j];
				delete[] psir_ylm[i][j];
			}
			delete[] dr[i];
			delete[] distance[i];
			delete[] psir_ylm[i];
			delete[] cal_flag[i];
		}
		delete[] dr;
		delete[] distance;
		delete[] psir_ylm;
		delete[] cal_flag;
	}

	//delete Rbxyz//
	for(int i=0;i<pw.bxyz;i++)
	{
		delete [] Rbxyz[i];
	}
	delete [] Rbxyz;

	double sum = 0.0;
	for(int is=0; is<NSPIN; is++)
	{
		for (int ir=0; ir<pw.nrxx; ir++)
		{
			sum += chr.rho[is][ir];
		}
	}
#ifdef __MPI
	Parallel_Reduce::reduce_double_pool( sum );
#endif
	OUT(ofs_warning,"charge density sumed from grid (Bfield)", sum * ucell.omega/ pw.ncxyz);
	
	double ne = sum * ucell.omega / pw.ncxyz;
	timer::tick("Gint_Gamma","gamma_charge_B",'I');

	return ne;
}



// this subroutine lies in the heart of LCAO algorithms.
// so it should be done very efficiently, very carefully.
// I might repeat again to emphasize this: need to optimize
// this code very efficiently, very carefully.
double Gint_Gamma::gamma_charge(void)
{
	TITLE("Gint_Gamma","gamma_charge");
	timer::tick("Gint_Gamma","gamma_charge",'I');	
	if(max_size==0) 
	{
		double sum = 0.0;
		goto ENDandRETURN;
	}

	// it's a uniform grid to save orbital values, so the delta_r is a constant.
	const double delta_r = ORB.dr_uniform;
	const Numerical_Orbital_Lm* pointer;

	// allocate 1
	int nnnmax=0;
	for(int T=0; T<ucell.ntype; T++)
	{
		nnnmax = max(nnnmax, nnn[T]);
	}
	
	double*** dr; // vectors between atom and grid: [bxyz, maxsize, 3]
	double** distance; // distance between atom and grid: [bxyz, maxsize]
	// set up band matrix psir_ylm and psir_DM
	int LD_pool=max_size*ucell.nwmax;
	int nblock;
	int *bsize; //band size: number of columns of a band
	int *colidx;
	double *psir_ylm_pool, **psir_ylm;
	double *psir_DM_pool, **psir_DM;

	dr = new double**[pw.bxyz];
	distance = new double*[pw.bxyz];
	bsize=new int[max_size];
	colidx=new int[max_size+1];
	psir_ylm_pool=new double[pw.bxyz*LD_pool];
	psir_ylm=new double *[pw.bxyz];
	psir_DM_pool=new double[pw.bxyz*LD_pool];
	psir_DM=new double *[pw.bxyz];
	ZEROS(psir_ylm_pool, pw.bxyz*LD_pool);
	ZEROS(psir_DM_pool, pw.bxyz*LD_pool);
	for(int i=0; i<pw.bxyz; ++i)
	{
		psir_ylm[i] = &psir_ylm_pool[i*LD_pool];
		psir_DM[i] = &psir_DM_pool[i*LD_pool];
		dr[i] = new double*[max_size];
		distance[i] = new double[max_size];
		ZEROS(distance[i], max_size);
		for(int j=0; j<max_size; j++) 
		{
			dr[i][j] = new double[3];
			ZEROS(dr[i][j],3);
		}
	}
	
	int *block_iw; // index of wave functions of each block;
	block_iw=new int[max_size];

	double* ylma = new double[nnnmax]; // Ylm for each atom: [bxyz, nnnmax]
	ZEROS(ylma, nnnmax);

	double mt[3]={0,0,0};
	double v1 = 0.0;
	int* vindex=new int[pw.bxyz];
	ZEROS(vindex, pw.bxyz);
	double phi=0.0;

	const int nbx = GridT.nbx;
	const int nby = GridT.nby;
	const int nbz_start = GridT.nbzp_start;
	const int nbz = GridT.nbzp;

	const int ncyz = pw.ncy*pw.nczp; // mohan add 2012-03-25
	//OUT(ofs_running, "nbx", nbx);
	//OUT(ofs_running, "nby", nby);
	//OUT(ofs_running, "nbz", nbz);
	for (int i=0; i<nbx; i++)
	{
		const int ibx = i*pw.bx; // mohan add 2012-03-25
		for (int j=0; j<nby; j++)
		{
			const int jby = j*pw.by; // mohan add 2012-03-25
			for (int k=nbz_start; k<nbz_start+nbz; k++) // FFT grid
			{
				const int kbz = k*pw.bz-pw.nczp_start; //mohan add 2012-03-25

				this->grid_index = (k-nbz_start) + j * nbz + i * nby * nbz;

				// get the value: how many atoms has orbital value on this grid.
				const int size = GridT.how_many_atoms[ this->grid_index ];
				if(size==0) continue;
				setVindex(ncyz, ibx, jby, kbz, vindex);
				cal_psir_ylm(size, this->grid_index, delta_r, phi, mt, dr, distance, pointer, ylma, colidx, block_iw, bsize,  psir_ylm);
				cal_band_rho(size, LD_pool, block_iw, bsize, colidx, psir_ylm, psir_DM, psir_DM_pool, vindex);
			}// k
		}// j
	}// i
	
	delete[] vindex;
	delete[] ylma;
	
	delete[] block_iw;
	//OUT(ofs_running, "block_iw deleted");
	for(int i=0; i<pw.bxyz; i++)
	{
		for(int j=0; j<max_size; j++) 
		{
			delete[] dr[i][j];
		}
		delete[] dr[i];
		delete[] distance[i];
	}
	delete[] dr;
	delete[] distance;
	delete[] psir_DM;
	delete[] psir_DM_pool;
	delete[] psir_ylm;
	delete[] psir_ylm_pool;
	delete[] colidx;
	delete[] bsize;
	
	double sum = 0.0;
	for(int is=0; is<NSPIN; is++)
	{
		for (int ir=0; ir<pw.nrxx; ir++)
		{
			sum += chr.rho[is][ir];
		}
	}
	
ENDandRETURN:
	//xiaohui add 'OUT_LEVEL', 2015-09-16
	if(OUT_LEVEL != "m") OUT(ofs_running, "sum", sum);

	timer::tick("Gint_Gamma","reduce_charge",'J');
#ifdef __MPI
	Parallel_Reduce::reduce_double_pool( sum );
#endif
	timer::tick("Gint_Gamma","reduce_charge",'J');
	OUT(ofs_warning,"charge density sumed from grid", sum * ucell.omega/ pw.ncxyz);

	double ne = sum * ucell.omega / pw.ncxyz;
	//xiaohui add 'OUT_LEVEL', 2015-09-16
	if(OUT_LEVEL != "m") OUT(ofs_running, "ne", ne);
	timer::tick("Gint_Gamma","gamma_charge",'I');

	return ne;
}

