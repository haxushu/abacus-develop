#include "parallel_grid.h"
#include "parallel_global.h"

Parallel_Grid::Parallel_Grid()
{
	this->allocate = false;
}

Parallel_Grid::~Parallel_Grid()
{
	if(this->allocate)
	{
		for(int ip=0; ip<NPOOL; ip++)
		{
			delete[] numz[ip];
			delete[] startz[ip];
			delete[] whichpro[ip];
		}
		delete[] numz;
		delete[] startz;
		delete[] whichpro;
	}
}

void Parallel_Grid::init(const int &ncx_in, const int &ncy_in, const int &ncz_in, const int &nczp_in, 
const int &nrxx_in, const int &nbz_in, const int &bz_in)
{

#ifndef __MPI
	return;
#endif

	TITLE("Parallel_Grid","init");
	
	this->ncx = ncx_in;
	this->ncy = ncy_in;
	this->ncz = ncz_in;
	this->nczp = nczp_in;
	this->nrxx = nrxx_in;
	this->nbz = nbz_in;
	this->bz = bz_in;

	if(nczp<0)
	{
		ofs_warning << " nczp = " << nczp << endl;
		WARNING_QUIT("Parallel_Grid::init","nczp<0");
	}

	assert(ncx > 0);
	assert(ncy > 0);
	assert(ncz > 0);

	this->ncxy = ncx * ncy;
	this->ncxyz = ncxy * ncz;

	// (2)
	assert(allocate==false);
	assert(NPOOL > 0);

	this->nproc_in_pool = new int[NPOOL];
	const int remain_pro = NPROC%NPOOL;
	for(int i=0; i<NPOOL; i++)
	{
		nproc_in_pool[i] = NPROC/NPOOL;
		if(i<remain_pro) this->nproc_in_pool[i]++;
	}	

	this->numz = new int*[NPOOL];
	this->startz = new int*[NPOOL];
	this->whichpro = new int*[NPOOL];
	this->numdata = new int*[NPOOL];
	this->startdata = new int*[NPOOL];

	for(int ip=0; ip<NPOOL; ip++)
	{
		const int nproc = nproc_in_pool[ip];
		this->numz[ip] = new int[nproc];
		this->startz[ip] = new int[nproc];
		this->whichpro[ip] = new int[this->ncz];
		this->numdata[ip] = new int[nproc];
		this->startdata[ip] = new int[nproc];
		ZEROS(this->numz[ip], nproc);
		ZEROS(this->startz[ip], nproc);
		ZEROS(this->whichpro[ip], this->ncz);
		ZEROS(this->numdata[ip], nproc);
		ZEROS(this->startdata[ip], nproc);
	}

	this->allocate = true;
	this->z_distribution();
	
	return;
}

void Parallel_Grid::z_distribution(void)
{
	assert(allocate);	

	int* startp = new int[NPOOL];
	startp[0] = 0;
	for(int ip=0; ip<NPOOL; ip++)
	{
//		ofs_running << "\n now POOL=" << ip;
		const int nproc = nproc_in_pool[ip];
		
		if(ip>0) startp[ip] = startp[ip-1] + nproc_in_pool[ip-1];
		
		// (1) how many z on each 'proc' in each 'pool'
		for(int iz=0; iz<nbz; iz++)
		{
			const int proc = iz % nproc;
			numz[ip][proc]+=bz;
		}

//		for(int proc=0; proc<nproc; proc++)
//		{
//			ofs_running << "\n proc=" << proc << " numz=" << numz[ip][proc];
//		}

		// (2) start position of z in each 'proc' in each 'pool'
		startz[ip][0] = 0;
		for (int proc=1; proc<nproc; proc++)
		{
			startz[ip][proc] = startz[ip][proc-1] + numz[ip][proc-1];
		}

//		for(int proc=0; proc<nproc; proc++)
//		{
//			ofs_running << "\n proc=" << proc << " startz=" << startz[ip][proc];
//		}

		// (3) each z belongs to which 'proc' ( global index )
		for(int iz=0; iz<ncz; iz++)
		{
			for(int proc=0; proc<nproc; proc++)
			{
				if(iz>=startz[ip][nproc-1])
				{
					whichpro[ip][iz] = startp[ip] + nproc-1;
					break;
				}
				else if(iz>=startz[ip][proc] && iz<startz[ip][proc+1])
				{
					whichpro[ip][iz] = startp[ip] + proc;
					break;
				}
			}
		}

//		for(int iz=0; iz<ncz; iz++)
//		{
//			ofs_running << "\n iz=" << iz << " whichpro=" << whichpro[ip][iz];
//		}

		//(4)
		for(int proc=0; proc<nproc; proc++)
		{
			numdata[ip][proc] = numz[ip][proc]*ncxy;
		}

		//(5)
		startdata[ip][0]=0;
		for(int proc=0; proc<nproc; proc++)
		{
			startdata[ip][proc]=startdata[ip][proc-1]+numdata[ip][proc-1];
		}
		
	}

	delete[] startp;
	return;
}


#ifdef __MPI
void Parallel_Grid::zpiece_to_all(double *zpiece, const int &iz, double *rho)
{
	assert(allocate);	
	//TITLE("Parallel_Grid","zpiece_to_all");
	MPI_Status ierror;

	const int znow = iz - this->startz[MY_POOL][RANK_IN_POOL];
	const int proc = this->whichpro[MY_POOL][iz];
	
	if(MY_POOL==0)
	{
		// case 1: the first part of rho in processor 0.
		// and send zpeice to to other pools.
		if(proc == 0 && MY_RANK ==0)
		{
			for(int ir=0; ir<ncxy; ir++)
			{
				rho[ir*nczp+znow] = zpiece[ir];
			}
			for(int ipool=1; ipool < NPOOL; ipool++)
			{
				MPI_Send(zpiece, ncxy, MPI_DOUBLE, this->whichpro[ipool][iz], iz, MPI_COMM_WORLD);
			}
		}

		// case 2: processor n (n!=0) receive rho from processor 0.
		// and the receive tag is iz.
		else if(proc == RANK_IN_POOL )
		{
			MPI_Recv(zpiece, ncxy, MPI_DOUBLE, 0, iz, MPI_COMM_WORLD,&ierror);
			for(int ir=0; ir<ncxy; ir++)
			{
				rho[ir*nczp + znow] = zpiece[ir];
			}
		}
				
		// case 2: > first part rho: processor 0 send the rho
		// to all pools. The tag is iz, because processor may
		// send more than once, and the only tag to distinguish
		// them is iz.
		else if(RANK_IN_POOL==0)
		{
			for(int ipool=0; ipool < NPOOL; ipool++)
			{
				MPI_Send(zpiece, ncxy, MPI_DOUBLE, this->whichpro[ipool][iz], iz, MPI_COMM_WORLD);
			}
		}
	}// MY_POOL == 0
	else
	{
		//ofs_running << "\n Receive charge density iz=" << iz << endl;
		// the processors in other pools always receive rho from
		// processor 0. the tag is 'iz'
		if(proc == MY_RANK )
		{
			MPI_Recv(zpiece, ncxy, MPI_DOUBLE, 0, iz, MPI_COMM_WORLD,&ierror);
			for(int ir=0; ir<ncxy; ir++)
			{
				rho[ir*nczp+znow] = zpiece[ir];
			}
		}
	}

	//ofs_running << "\n iz = " << iz << " Done.";
	return;	
}
#endif

#ifdef __MPI
void Parallel_Grid::reduce_to_fullrho(double *rhotot, double *rhoin)
{
	//TITLE("Parallel_Grid","reduce_to_fullrho");

	// if not the first pool, wait here until processpr 0
	// send the Barrier command.
	if(MY_POOL!=0) 
	{
		MPI_Barrier(MPI_COMM_WORLD);
		return;
	}

	double* zpiece = new double[this->ncxy];
	
	for(int iz=0; iz<this->ncz; iz++)
	{
		const int znow = iz - this->startz[MY_POOL][RANK_IN_POOL];
		const int proc = this->whichpro[MY_POOL][iz];
		ZEROS(zpiece, this->ncxy);
		int tag = iz;
		MPI_Status ierror;

		// case 1: the first part of rho in processor 0.
		if(proc == 0 && RANK_IN_POOL ==0)
		{
			for(int ir=0; ir<ncxy; ir++)
			{
				zpiece[ir] = rhoin[ir*this->nczp + znow];
			}
		}

		// case 2: > first part rho: send the rho to
		// processor 0.
		else if(proc == RANK_IN_POOL )
		{
			for(int ir=0; ir<ncxy; ir++)
			{
				zpiece[ir] = rhoin[ir*this->nczp + znow];
			}
			MPI_Send(zpiece, ncxy, MPI_DOUBLE, 0, tag, POOL_WORLD);
		}

		 // case 2: > first part rho: processor 0 receive the rho
		 // from other processors
		else if(RANK_IN_POOL==0)
		{
			MPI_Recv(zpiece, ncxy, MPI_DOUBLE, proc, tag, POOL_WORLD, &ierror);
		}

		if(MY_RANK==0)
		{
			for(int ix=0; ix<this->ncx; ix++)
			{
				for(int iy=0; iy<this->ncy; iy++)
				{
					const int ir = ix * this->ncy + iy;
					rhotot[ix * ncy * ncz + iy * ncz + iz] = zpiece[ir];
				}
			}	
		}
	}

	delete[] zpiece;	

	MPI_Barrier(MPI_COMM_WORLD);

	return;
}
#endif
