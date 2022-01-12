#include "../pw_basis.h"
#ifdef __MPI
#include "test_tool.h"
#include "../../src_parallel/parallel_global.h"
#include "mpi.h"
#endif
#include "../../module_base/global_function.h"
#include "../../module_base/constants.h"
#include "pw_test.h"
extern int nproc_in_pool,rank_in_pool;
using namespace std;

TEST_F(PWTEST,test2_1_2)
{
    cout<<"dividemthd 2, gamma_only: on, check gcar,gdirect,gg,istot2bigixy,ig2isz"<<endl;
    //--------------------------------------------------
    ModuleBase::Matrix3 latvec(1,0,0,0,1,1,0,0,2);
    bool gamma_only = true;
    double wfcecut = 30;
    double lat0 = 10;
    int distribution_type = 2;
    //--------------------------------------------------

    ModulePW::PW_Basis pwtest;

    pwtest.initgrids(lat0, latvec, 2*wfcecut,nproc_in_pool, rank_in_pool);
    pwtest.initparameters(gamma_only, wfcecut, distribution_type);
    pwtest.distribute_r();
    pwtest.distribute_g();
    pwtest.collect_local_pw();
    ModuleBase::Matrix3 GT,G,GGT;
    GT = latvec.Inverse();
	G  = GT.Transpose();
	GGT = G * GT;
    double tpiba2 = ModuleBase::TWO_PI * ModuleBase::TWO_PI / lat0 / lat0;
    double ggecut = wfcecut / tpiba2;

    //ref
    const int totnpw_ref = 2931;
    const int totnst_ref = 175;
    const int nx_ref = 25;
    const int bigny_ref = 36;
    const int ny_ref = 19;
    const int nz_ref = 50;
    //some results for different number of processors
    int npw_per_ref[12][12]={
        {2931},
        {1548,1383},
        {1133,744,1054},
        {870,678,586,797},
        {699,617,397,559,659},
        {598,535,415,329,521,533},
        {501,457,421,267,371,431,483},
        {458,412,384,294,230,356,406,391},
        {418,364,364,304,181,267,319,367,347},
        {374,340,328,286,234,165,277,291,329,307},
        {328,314,298,276,234,160,200,240,310,296,275},
        {303,295,291,257,233,183,151,204,236,262,254,262}
    };
    int nst_per_ref[12][12]={
        {175},
        {88,87},
        {59,58,58},
        {44,44,44,43},
        {35,35,35,35,35},
        {30,29,29,29,29,29},
        {25,25,25,25,25,25,25},
        {22,22,22,22,22,22,22,21},
        {20,20,20,20,19,19,19,19,19},
        {18,18,18,18,18,17,17,17,17,17},
        {16,16,16,16,16,16,16,16,16,16,15},
        {15,15,15,15,15,15,15,14,14,14,14,14}
    };
    int *npw_per;
    if(rank_in_pool == 0)
    {
        npw_per = new int [nproc_in_pool];
    }
#ifdef __MPI
    MPI_Gather(&pwtest.npw,1,MPI_INT,npw_per,1,MPI_INT,0,POOL_WORLD);
#else
    if(rank_in_pool == 0) npw_per[0] = pwtest.npw;
#endif
    if(rank_in_pool == 0)
    {
        if(nproc_in_pool <= 12)
        {
            for(int ip = 0 ; ip < nproc_in_pool ; ++ip)
            {
                ASSERT_EQ(npw_per_ref[nproc_in_pool-1][ip], npw_per[ip]);
                ASSERT_EQ(nst_per_ref[nproc_in_pool-1][ip], pwtest.nst_per[ip]);
            }
        }
        else
        {
            cout<<"Please use mpi processors no more than 12."<<endl;
        }
        delete []npw_per;
    }
    

    //results
    int tot_npw = 0;
#ifdef __MPI
    MPI_Allreduce(&pwtest.npw, &tot_npw, 1, MPI_INT, MPI_SUM, POOL_WORLD);
#else
    tot_npw = pwtest.npw;
#endif
    ASSERT_EQ(pwtest.nx, nx_ref);
    ASSERT_EQ(pwtest.ny, ny_ref);
    ASSERT_EQ(pwtest.bigny, bigny_ref);
    ASSERT_EQ(pwtest.nz, nz_ref);
    ASSERT_EQ(tot_npw, totnpw_ref);
    ASSERT_EQ(pwtest.nstot,totnst_ref);
    ASSERT_EQ(pwtest.bignxyz, nx_ref*bigny_ref*nz_ref);


    int *tmpx = new int[pwtest.nx*pwtest.ny*pwtest.nz];
    int *tmpy = new int[pwtest.nx*pwtest.ny*pwtest.nz];
    int *tmpz = new int[pwtest.nx*pwtest.ny*pwtest.nz];
    ModuleBase::GlobalFunc::ZEROS(tmpx,pwtest.nx*pwtest.ny*pwtest.nz);
    ModuleBase::GlobalFunc::ZEROS(tmpy,pwtest.nx*pwtest.ny*pwtest.nz);
    ModuleBase::GlobalFunc::ZEROS(tmpz,pwtest.nx*pwtest.ny*pwtest.nz);
    
    int * startnst = new int [nproc_in_pool];
    startnst[0] = 0;
    for(int ip = 1 ; ip < nproc_in_pool; ++ip)
    {
        startnst[ip] = startnst[ip-1] + pwtest.nst_per[ip-1];
    }

    for(int ig = 0 ; ig < pwtest.npw; ++ig)
    {
        int istot = pwtest.ig2isz[ig] / pwtest.nz + startnst[rank_in_pool];
        // int is = pwtest.ig2isz[ig] / pwtest.nz;
        int iz = pwtest.ig2isz[ig] % pwtest.nz;
        int iy = pwtest.istot2bigixy[istot] % pwtest.bigny;
        int ix = pwtest.istot2bigixy[istot] / pwtest.bigny;
        // int iy = pwtest.is2ixy[is] % pwtest.ny;
        // int ix = pwtest.is2ixy[is] / pwtest.ny;

        tmpx[iz+(iy+ix*pwtest.ny)*pwtest.nz] = int(pwtest.gdirect[ig].x);
        tmpy[iz+(iy+ix*pwtest.ny)*pwtest.nz] = int(pwtest.gdirect[ig].y);
        tmpz[iz+(iy+ix*pwtest.ny)*pwtest.nz] = int(pwtest.gdirect[ig].z);
    }
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE,tmpx,pwtest.nxyz,MPI_INT,MPI_SUM,POOL_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,tmpy,pwtest.nxyz,MPI_INT,MPI_SUM,POOL_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,tmpz,pwtest.nxyz,MPI_INT,MPI_SUM,POOL_WORLD);
#endif
    if(rank_in_pool==0)
    {
        for(int iz = 0 ; iz < pwtest.nz; ++iz)
        {
            for(int iy = 0 ; iy < pwtest.ny ; ++iy)
            {
                for(int ix = 0 ; ix < pwtest.nx ; ++ix)
                {
                    ModuleBase::Vector3<double> f;
                    f.x = ix;
                    f.y = iy;
                    f.z = iz;
                    if(iz >= int(pwtest.nz/2) +1) f.z -= pwtest.nz;
                    if(ix >= int(pwtest.nx/2) +1) f.x -= pwtest.nx;
                    double modulus = f * (GGT * f);
                    if (modulus <= ggecut)
                    {
                        EXPECT_EQ(tmpx[iz + iy*pwtest.nz + ix*pwtest.ny*pwtest.nz], int(f.x));
                        EXPECT_EQ(tmpy[iz + iy*pwtest.nz + ix*pwtest.ny*pwtest.nz], int(f.y));
                        EXPECT_EQ(tmpz[iz + iy*pwtest.nz + ix*pwtest.ny*pwtest.nz], int(f.z));
                    }
                    
                }
            }
        }
    }
    for(int ig = 0 ;ig < pwtest.npw ; ++ig)
    {
        ModuleBase::Vector3<double> f;
        f.x = pwtest.gdirect[ig].x;
        f.y = pwtest.gdirect[ig].y;
        f.z = pwtest.gdirect[ig].z;
        ModuleBase::Vector3<double> gcar;
        gcar = f * G;
        double modulus = f*GGT*f;
        EXPECT_NEAR(gcar.x,pwtest.gcar[ig].x,1e-6);
        EXPECT_NEAR(gcar.y,pwtest.gcar[ig].y,1e-6);
        EXPECT_NEAR(gcar.z,pwtest.gcar[ig].z,1e-6);
        EXPECT_NEAR(modulus,pwtest.gg[ig],1e-6);
    }
    delete [] startnst;
    delete [] tmpx;
    delete [] tmpy;
    delete [] tmpz;
}