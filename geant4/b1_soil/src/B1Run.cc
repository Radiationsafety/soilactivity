#include "B1Run.hh"

#include <algorithm>
#include <cmath>

void B1Run::Reset(G4int nDet)
{
  fNin.assign(nDet, 0);
  fSpec.assign(nDet, std::vector<G4double>(kNBins, 0.0));
  fEdep.assign(nDet, 0.0);
  fEdep2.assign(nDet, 0.0);
}

void B1Run::AddEntering(G4int detIdx, G4double E_MeV)
{
  if (detIdx >= (G4int)fNin.size()) Grow(detIdx + 1);
  ++fNin[detIdx];
  ++fSpec[detIdx][BinIndex(E_MeV)];
}

void B1Run::AddEdep(G4int detIdx, G4double edep)
{
  if (detIdx >= (G4int)fEdep.size()) Grow(detIdx + 1);
  fEdep[detIdx]  += edep;
  fEdep2[detIdx] += edep * edep;
}

void B1Run::Grow(G4int nDet)
{
  // расширение БЕЗ очистки уже накопленных данных
  if (nDet <= (G4int)fNin.size()) return;
  fNin.resize(nDet, 0);
  fSpec.resize(nDet, std::vector<G4double>(kNBins, 0.0));
  fEdep.resize(nDet, 0.0);
  fEdep2.resize(nDet, 0.0);
}

void B1Run::Merge(const G4Run* runPtr)
{
  const B1Run* run = static_cast<const B1Run*>(runPtr);
  G4Run::Merge(runPtr);

  const G4int n = run->fNin.size();
  if (fNin.size() < (size_t)n) {
    fNin.resize(n, 0);
    fSpec.resize(n, std::vector<G4double>(kNBins, 0.0));
    fEdep.resize(n, 0.0);
    fEdep2.resize(n, 0.0);
  }
  for (G4int i = 0; i < n; ++i) {
    fNin[i] += run->fNin[i];
    for (G4int b = 0; b < kNBins; ++b) fSpec[i][b] += run->fSpec[i][b];
    fEdep[i]  += run->fEdep[i];
    fEdep2[i] += run->fEdep2[i];
  }
}
