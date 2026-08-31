#include "B1PrimaryGeneratorAction.hh"
#include "B1DetectorConstruction.hh"

#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4Gamma.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include "G4RandomDirection.hh"
#include "G4RunManager.hh"
#include "G4Exception.hh"

#include <algorithm>
#include <cctype>

B1PrimaryGeneratorAction::B1PrimaryGeneratorAction()
{
  fGun = new G4ParticleGun(1);
  fGun->SetParticleDefinition(G4Gamma::Definition());
  SetNuclide("Cs137"); // линия 661.7 кэВ, выход 0.851 (по умолчанию)
}

B1PrimaryGeneratorAction::~B1PrimaryGeneratorAction()
{
  delete fGun;
}

void B1PrimaryGeneratorAction::SetNuclide(const G4String& name)
{
  G4String n = name;
  std::transform(n.begin(), n.end(), n.begin(), ::tolower);
  fLines.clear();
  if (n == "cs137") {
    fLines.push_back({ 0.6617 * MeV, 0.851 });              // Ba-137m
  } else if (n == "co60") {
    fLines.push_back({ 1.173228 * MeV, 0.9985 });
    fLines.push_back({ 1.332501 * MeV, 0.99998 });
  } else if (n == "am241") {
    fLines.push_back({ 0.059541 * MeV, 0.359 });
  } else {
    G4ExceptionDescription msg;
    msg << "Unknown nuclide '" << name
        << "' (allowed: Cs137 | Co60 | Am241), or use /b1soil/addLine";
    G4Exception("B1PrimaryGeneratorAction::SetNuclide", "B1SOIL04",
                FatalException, msg);
  }
}

void B1PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
  // --- координата распада ----------------------------------------------------
  G4ThreeVector decayPos(fSrcX, fSrcY, fSrcZ);
  if (fUseModel) {
    if (fModelCells.empty()) {
      G4Exception("B1PrimaryGeneratorAction::GeneratePrimaries", "B1SOIL05",
                  FatalException,
                  "Model source requested but no cells defined "
                  "(use /b1soil/addModelCell)");
    }
    if (fModelCum.size() != fModelCells.size()) {
      fModelCum.clear();
      G4double sum = 0.0;
      for (const auto& c : fModelCells) { sum += c.activity; fModelCum.push_back(sum); }
    }
    const G4double u = G4UniformRand() * fModelCum.back();
    const size_t k = (size_t)(std::upper_bound(fModelCum.begin(),
                                               fModelCum.end(), u)
                              - fModelCum.begin());
    decayPos = fModelCells[std::min(k, fModelCells.size() - 1)].pos;
  }
  else if (fSrcIndex >= 0) {
    // положение ячейки берём из текущей геометрии (актуально и в MT)
    auto* dc = static_cast<B1DetectorConstruction*>(
        G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    decayPos = dc->SourcePosition(fSrcIndex);
  }

  // --- эмиссия фотонов согласно линиям нуклида --------------------------------
  for (const auto& line : fLines) {
    G4int nPh = static_cast<G4int>(line.yield);           // целая часть выхода
    if (G4UniformRand() < line.yield - nPh) ++nPh;        // дробная часть
    for (G4int k = 0; k < nPh; ++k) {
      fGun->SetParticleEnergy(line.energy);
      fGun->SetParticlePosition(decayPos);
      fGun->SetParticleDirection(G4RandomDirection());
      fGun->GeneratePrimaryVertex(event);
    }
  }
  // События без вертикали (например, 14.9 % распадов Cs-137 без гаммы)
  // допустимы: они корректно входят в нормировку "на распад".
}
