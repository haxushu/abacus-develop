#ifndef MATHZONE_H
#define MATHZONE_H

#include "realarray.h"
using namespace std;

class Mathzone
{
public:

    Mathzone();
    ~Mathzone();

    template<class T>
    static T Max3(const T &a,const T &b,const T &c)
    {
        T v;
        if (a>=b && a>=c) v = a;
        else if (b>=a && b>=c) v = b;
        else if (c>=a && c>=b) v = c;
        return v;
    }

    // be careful, this can only be used for plane wave
    // during parallel calculation
    static void norm_pw(complex<double> *u, const int n);

    //========================================================
    // Polynomial_Interpolation
    //========================================================
    static void Polynomial_Interpolation
    (
        const realArray &table,
        const int &dim1,
        const int &dim2,
        realArray &y,
        const int &dim_y,
        const int &table_length,
        const double &table_interval,
        const double &x
    );
    static double Polynomial_Interpolation
    (
        const realArray &table,
        const int &dim1,
        const int &dim2,
        const int &table_length,
        const double &table_interval,
        const double &x				// input value
    );
    static double Polynomial_Interpolation
    (
        const double *table,
        const int &table_length,
        const double &table_interval,
        const double &x				// input value
    );
    static double Polynomial_Interpolation_xy
    (
        const double *xpoint,
        const double *ypoint,
        const int table_length,
        const double &x             // input value
    );

    //========================================================
    // Spherical Bessel
    //========================================================
    static void Spherical_Bessel
    (
        const int &msh,	//number of grid points
        const double *r,//radial grid
        const double &q,	//
        const int &l,	//angular momentum
        double *jl	//jl(1:msh) = j_l(q*r(i)),spherical bessel function
    );

	static void Spherical_Bessel
	(           
	    const int &msh, //number of grid points
		const double *r,//radial grid
		const double &q,    //
		const int &l,   //angular momentum
		double *sj,     //jl(1:msh) = j_l(q*r(i)),spherical bessel function
		double *sjp
	);

	
    static void Spherical_Bessel_Roots
    (
        const int &num,
        const int &l,
        const double &epsilon,
        double* eigenvalue,
        const double &rcut
    );
private:

    static double Spherical_Bessel_7(const int n, const double &x);
    static double BESSJY(double x, double xnu, double *rj, double *ry, double *rjp, double *ryp);
    static void BESCHB(double x, double *gam1, double *gam2, double *gampl, double *gammi);
    static double CHEBEV(double a, double b, double c[], int m, double x);
    static int IMAX(int a, int b);

public:

    static void Ylm_Real
    (
        const int lmax2, 			// lmax2 = (lmax+1)^2
        const int ng,				//
        const Vector3<double> *g, 	// g_cartesian_vec(x,y,z)
        matrix &ylm 				// output
    );
	
	static void Ylm_Real2
	(
    	const int lmax2, 			// lmax2 = (lmax+1)^2
    	const int ng,				//
    	const Vector3<double> *g, 	// g_cartesian_vec(x,y,z)
    	matrix &ylm 				// output
	);

	static void rlylm
	(
    	const int lmax, 	
    	const double& x,				
    	const double& y,
		const double& z, // g_cartesian_vec(x,y,z)
    	double* rly 	 // output
	);

    static long double Fact(const int n);
    static int Semi_Fact(const int n);

    static void Simpson_Integral
    (
        const int mesh,
        const double *func,
        const double *rab,
        double &asum
    );
//==========================================================
// MEMBER FUNCTION :
// NAME : Direct_to_Cartesian
// use lattice vector matrix R
// change the direct vector (dx,dy,dz) to cartesuab vectir
// (cx,cy,cz)
// (dx,dy,dz) = (cx,cy,cz) * R
//
// NAME : Cartesian_to_Direct
// the same as above
// (cx,cy,cz) = (dx,dy,dz) * R^(-1)
//==========================================================
    static inline void Direct_to_Cartesian
    (
        const double &dx,const double &dy,const double &dz,
        const double &R11,const double &R12,const double &R13,
        const double &R21,const double &R22,const double &R23,
        const double &R31,const double &R32,const double &R33,
        double &cx,double &cy,double &cz)
    {
        static Matrix3 lattice_vector;
        static Vector3<double> direct_vec, cartesian_vec;
        lattice_vector.e11 = R11;
        lattice_vector.e12 = R12;
        lattice_vector.e13 = R13;
        lattice_vector.e21 = R21;
        lattice_vector.e22 = R22;
        lattice_vector.e23 = R23;
        lattice_vector.e31 = R31;
        lattice_vector.e32 = R32;
        lattice_vector.e33 = R33;

        direct_vec.x = dx;
        direct_vec.y = dy;
        direct_vec.z = dz;

        cartesian_vec = direct_vec * lattice_vector;
        cx = cartesian_vec.x;
        cy = cartesian_vec.y;
        cz = cartesian_vec.z;
        return;
    }

    static inline void Cartesian_to_Direct
    (
        const double &cx,const double &cy,const double &cz,
        const double &R11,const double &R12,const double &R13,
        const double &R21,const double &R22,const double &R23,
        const double &R31,const double &R32,const double &R33,
        double &dx,double &dy,double &dz)
    {
        static Matrix3 lattice_vector, inv_lat;
        lattice_vector.e11 = R11;
        lattice_vector.e12 = R12;
        lattice_vector.e13 = R13;
        lattice_vector.e21 = R21;
        lattice_vector.e22 = R22;
        lattice_vector.e23 = R23;
        lattice_vector.e31 = R31;
        lattice_vector.e32 = R32;
        lattice_vector.e33 = R33;

        inv_lat = lattice_vector.Inverse();

        static Vector3<double> direct_vec, cartesian_vec;
        cartesian_vec.x = cx;
        cartesian_vec.y = cy;
        cartesian_vec.z = cz;
        direct_vec = cartesian_vec * inv_lat;
        dx = direct_vec.x;
        dy = direct_vec.y;
        dz = direct_vec.z;
        return;
    }


    static void To_Polar_Coordinate
    (
        const double &x_cartesian,
        const double &y_cartesian,
        const double &z_cartesian,
        double &r,
        double &theta,
        double &phi
    );

};

#endif
