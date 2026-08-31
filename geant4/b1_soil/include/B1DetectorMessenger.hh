#ifndef B1DetectorMessenger_h
#define B1DetectorMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

class B1DetectorConstruction;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithAString;

/// UI-команды /b1soil/* для геометрии и метаданных прогона.
/// Все геометрические параметры перестраивают геометрию автоматически.
class B1DetectorMessenger : public G4UImessenger
{
  public:
    explicit B1DetectorMessenger(B1DetectorConstruction* dc);
    ~B1DetectorMessenger() override;

    void SetNewValue(G4UIcommand* cmd, G4String value) override;

  private:
    B1DetectorConstruction* fDC = nullptr;

    G4UIcmdWithAnInteger*       fSetNx = nullptr;
    G4UIcmdWithAnInteger*       fSetNy = nullptr;
    G4UIcmdWithADoubleAndUnit*  fSetCell = nullptr;
    G4UIcmdWithADoubleAndUnit*  fSetSrcDepth = nullptr;
    G4UIcmdWithADoubleAndUnit*  fSetSoilDepth = nullptr;
    G4UIcmdWithADoubleAndUnit*  fSetDetHeight = nullptr;
    G4UIcmdWithADoubleAndUnit*  fSetDetRadius = nullptr;
    G4UIcmdWithADoubleAndUnit*  fSetSoilDensity = nullptr;
    G4UIcmdWithAString*         fSetDetMode = nullptr;
    G4UIcmdWithAnInteger*       fSetDetIndex = nullptr;
    G4UIcmdWithAString*         fSetRunType = nullptr;
    G4UIcmdWithAString*         fSetOutputFile = nullptr;
};

#endif
