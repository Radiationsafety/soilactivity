#ifndef B1PrimaryGeneratorMessenger_h
#define B1PrimaryGeneratorMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

class B1PrimaryGeneratorAction;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithABool;
class G4UIcmdWithAString;
class G4UIcommand;

/// UI-команды /b1soil/* генератора частиц:
///   setGammaLine Cs137|Co60|Am241, addLine E yield, clearLines,
///   setSrcIndex / setSrcX|Y|Z, useModelSource, addModelCell, clearModelCells.
class B1PrimaryGeneratorMessenger : public G4UImessenger
{
  public:
    explicit B1PrimaryGeneratorMessenger(B1PrimaryGeneratorAction* pg);
    ~B1PrimaryGeneratorMessenger() override;

    void SetNewValue(G4UIcommand* cmd, G4String value) override;

  private:
    B1PrimaryGeneratorAction* fPG = nullptr;

    G4UIcmdWithAString*        fSetLine = nullptr;
    G4UIcommand*               fAddLine = nullptr;
    G4UIcmdWithAString*        fClearLines = nullptr;
    G4UIcmdWithAnInteger*      fSetSrcIndex = nullptr;
    G4UIcmdWithADoubleAndUnit* fSetSrcX = nullptr;
    G4UIcmdWithADoubleAndUnit* fSetSrcY = nullptr;
    G4UIcmdWithADoubleAndUnit* fSetSrcZ = nullptr;
    G4UIcmdWithABool*          fUseModel = nullptr;
    G4UIcommand*               fAddCell = nullptr;
    G4UIcmdWithAString*        fClearCells = nullptr;
};

#endif
