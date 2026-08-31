#ifndef B1Run_h
#define B1Run_h 1

#include "G4Run.hh"
#include "globals.hh"

#include <algorithm>
#include <cmath>
#include <vector>

/// Накопители одного прогона (run), потокобезопасно сливаются в master-Run
/// (паттерн примера B4). Для каждого детектора хранит:
///   fNin[i]  - число фотонов, вошедших в сферу i (через границу);
///   fSpec[i] - спектр вошедших фотонов по логарифмическим бинам энергии;
///   fEdep[i] - суммарное энерговыделение в сфере i (нормируется на массу
///              детектора в B1RunAction -> керма в Gy);
///   fEdep2[i]- сумма квадратов (для статистической погрешности).
///
/// Энергетическая сетка спектра: NBINS логарифмических бинов
/// от EMIN до EMAX (МэВ). Идентичная сетка воспроизведена в
/// python/b1soil_io.py — при изменении синхронизировать обе стороны.
class B1Run : public G4Run
{
  public:
    static constexpr G4int    kNBins = 48;
    static constexpr G4double kEMin  = 0.01;   // MeV
    static constexpr G4double kEMax  = 3.0;    // MeV

    B1Run() = default;

    // индекс бина для энергии E (МэВ), с зажимом на краях
    static G4int BinIndex(G4double E_MeV)
    {
      if (E_MeV <= kEMin) return 0;
      if (E_MeV >= kEMax) return kNBins - 1;
      const G4double w = std::log(E_MeV / kEMin)
                       / std::log(kEMax / kEMin);
      return std::min(kNBins - 1, static_cast<G4int>(w * kNBins));
    }

    void Reset(G4int nDet);
    void Grow(G4int nDet);
    void AddEntering(G4int detIdx, G4double E_MeV);
    void AddEdep(G4int detIdx, G4double edep);

    void Merge(const G4Run* run) override;

    const G4long&                        GetNin(G4int i) const { return fNin[i]; }
    const std::vector<G4double>&         GetSpectrum(G4int i) const { return fSpec[i]; }
    G4double                             GetEdep(G4int i) const { return fEdep[i]; }
    G4double                             GetEdep2(G4int i) const { return fEdep2[i]; }
    G4int                                GetNDet() const { return (G4int)fNin.size(); }

  private:
    std::vector<G4long>                  fNin;
    std::vector<std::vector<G4double>>   fSpec;
    std::vector<G4double>                fEdep;
    std::vector<G4double>                fEdep2;
};

#endif
