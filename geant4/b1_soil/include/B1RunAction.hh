#ifndef B1RunAction_h
#define B1RunAction_h 1

#include "G4UserRunAction.hh"
#include "globals.hh"

#include <vector>

/// Метаданные текущего прогона, общие для master-потока.
/// Заполняются UI-командами (мессенджеры B1DetectorMessenger /
/// B1PrimaryGeneratorMessenger) до /run/beamOn и читаются в
/// EndOfRunAction на master при записи CSV-строки.
struct B1RunMeta
{
  static G4String runType;    // "SENSITIVITY" | "MODEL" (произвольная метка)
  static G4String outputFile; // путь к CSV
  static G4int    srcIndex;   // >=0 - индекс ячейки; -1 - модельный источник; -2 - ручные координаты
  static G4bool   useModel;   // источник распределён по модельным ячейкам
  static G4double srcX, srcY, srcZ; // ручные координаты источника (CLHEP-единицы)
};

class B1RunAction : public G4UserRunAction
{
  public:
    B1RunAction() = default;
    ~B1RunAction() override = default;

    G4Run* GenerateRun() override;
    void   EndOfRunAction(const G4Run* run) override;
};

#endif
