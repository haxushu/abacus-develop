#ifndef HAMILTPW_H
#define HAMILTPW_H

#include "hamilt.h"

namespace hamilt
{

class HamiltPW : public Hamilt
{
  public:
    HamiltPW(){};
    ~HamiltPW(){};

    // construct Hamiltonian matrix with inputed electonic density
    void constructHamilt(const int iter, const MatrixBlock<double> rho) override
    {
        this->ch_mock();
    };

    // for target K point, update consequence of hPsi() and matrix()
    void updateHk(int ik) override
    {
        this->hk_mock();
    };

    // core function: for solving eigenvalues of Hamiltonian with iterative method
    virtual void hPsi(const psi::Psi<std::complex<double>>& psi, psi::Psi<std::complex<double>>& hpsi) const override
    {
        this->hpsi_mock(psi, hpsi);
    };



  private:
    void ch_mock();
    void hk_mock();
    void hpsi_mock(const psi::Psi<std::complex<double>>& psi, psi::Psi<std::complex<double>>& hpsi) const;
};

} // namespace hamilt

#endif