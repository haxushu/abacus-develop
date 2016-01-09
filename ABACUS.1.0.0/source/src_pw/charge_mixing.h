//=================================
// Mohan add 2010-02-01
//=================================
#ifndef CHARGE_MIXING_H
#define CHARGE_MIXING_H

//==================================
// (1) Plain Mixing
// (2) KerKer Mixing
// (3) Pulay Mixing
// (4) Modified Broden Mixing
//===================================
#include "tools.h"
#include "charge.h"
class Charge_Mixing: public Charge
{
	public:
	Charge_Mixing();
	~Charge_Mixing();

    string mixing_mode;
    double mixing_beta;
    int mixing_ndim;
	double mixing_gg0; //mohan add 2014-09-27

    void set_mixing
    (
        const string &mixing_mode_in,
        const double &mixing_beta_in,
        const int &mixing_ndim_in,
	const double &mixing_gg0_in
    );//mohan add mixing_gg0_in 2014-09-27
	
	// also be used by namespace: GGA_pw
    void set_rhog(const double *rho_in, complex<double> *rhog_in)const;
	
	protected:

	// simple plain mixing method.	
    void plain_mixing( double *rho_in, double *rho_save_in ) const;
	
	// Kerker mixing method.
	// Kerker method must use plane wave basis, change the high energy plane wave 
	// more than low energy plane wave.
	void Kerker_mixing( double *rho, const complex<double> *residual_g, double *rho_save);
	
	// tools
	void set_rhor(complex<double> *rhog, double *rho)const;
	double rhog_dot_product(
		const complex<double> *rhog11,const complex<double> *rhog12,
		const complex<double> *rhog21,const complex<double> *rhog22) const;
	
};

#endif
