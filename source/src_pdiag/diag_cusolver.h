#include "../src_pw/global.h"
#include "../src_parallel/parallel_reduce.h"
void save_H(const double *H, double *H_full)
{

        int ir,ic;
        for (int i=0; i<GlobalV::NLOCAL; i++)
        {
            double* lineH = new double[GlobalV::NLOCAL-i];
            ModuleBase::GlobalFunc::ZEROS(lineH, GlobalV::NLOCAL-i);

            ir = GlobalC::ParaO.trace_loc_row[i];
            if (ir>=0)
            {
                // data collection
                for (int j=i; j<GlobalV::NLOCAL; j++)
                {
                    ic = GlobalC::ParaO.trace_loc_col[j];
                    if (ic>=0)
                    {
                        int iic;
                        iic=ir+ic*GlobalC::ParaO.nrow;
                        lineH[j-i] = H[iic];
                    }
                }
            }
            else
            {
                //do nothing
            }

            Parallel_Reduce::reduce_double_all(lineH,GlobalV::NLOCAL-i);

            if (GlobalV::DRANK==0)
            {

                for (int j=i; j<GlobalV::NLOCAL; j++)
                {
                    int index = i + j*GlobalV::NLOCAL;
                    H_full[index] = lineH[j-i];
                }

            }
            delete[] lineH;
        }

		if (GlobalV::DRANK==0){
        for (int i=0; i<GlobalV::NLOCAL; i++)
            for (int j=0; j<i; j++) {
                int index_L = i + j*GlobalV::NLOCAL;
                int index_U = j + i*GlobalV::NLOCAL;
                H_full[index_L] = H_full[index_U];
            }
		}

    return;
}



void save_S(const double *S, double *S_full)
{

        int ir,ic;
        for (int i=0; i<GlobalV::NLOCAL; i++)
        {
            double* lineS = new double[GlobalV::NLOCAL-i];
            ModuleBase::GlobalFunc::ZEROS(lineS, GlobalV::NLOCAL-i);

            ir = GlobalC::ParaO.trace_loc_row[i];
            if (ir>=0)
            {
                // data collection
                for (int j=i; j<GlobalV::NLOCAL; j++)
                {
                    ic = GlobalC::ParaO.trace_loc_col[j];
                    if (ic>=0)
                    {
                        int iic;
                        iic=ir+ic*GlobalC::ParaO.nrow;
                        lineS[j-i] = S[iic];
                    }
                }
            }
            else
            {
                //do nothing
            }

            Parallel_Reduce::reduce_double_all(lineS,GlobalV::NLOCAL-i);

            if (GlobalV::DRANK==0)
            {

                for (int j=i; j<GlobalV::NLOCAL; j++)
                {
                    int index = i + j*GlobalV::NLOCAL;
                    S_full[index] = lineS[j-i];
                }

            }
            delete[] lineS;
        }

		if (GlobalV::DRANK==0){
        for (int i=0; i<GlobalV::NLOCAL; i++)
            for (int j=0; j<i; j++) {
                int index_L = i + j*GlobalV::NLOCAL;
                int index_U = j + i*GlobalV::NLOCAL;
                S_full[index_L] = S_full[index_U];
            }
		}

    return;
}