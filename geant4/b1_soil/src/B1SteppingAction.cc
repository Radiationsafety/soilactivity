#include "B1SteppingAction.hh"
#include "B1Run.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Gamma.hh"
#include "G4LogicalVolume.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"

void B1SteppingAction::UserSteppingAction(const G4Step* step)
{
  const auto* pre = step->GetPreStepPoint();
  const auto* lv = pre->GetTouchableHandle()->GetVolume()->GetLogicalVolume();

  // Большинство шагов отсекаем здесь: объём не детекторный
  if (lv == nullptr || lv->GetName() != "DetLV") return;

  auto* run = static_cast<B1Run*>(
      G4RunManager::GetRunManager()->GetNonConstCurrentRun());
  if (run == nullptr) return;

  const G4int copyNo = pre->GetTouchableHandle()->GetCopyNumber();

  // 1) энерговыделение внутри сферы (керма/доза, cross-check)
  const G4double edep = step->GetTotalEnergyDeposit();
  if (edep > 0.) run->AddEdep(copyNo, edep);

  // 2) фотон, вошедший в сферу через границу -> спектр флюенса
  const auto* track = step->GetTrack();
  if (pre->GetStepStatus() == fGeomBoundary &&
      track->GetDefinition() == G4Gamma::GammaDefinition()) {
    run->AddEntering(copyNo, track->GetKineticEnergy() / MeV);
  }
}
