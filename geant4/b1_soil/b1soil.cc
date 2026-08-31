/// b1soil - развитие базового примера Geant4 basic/B1:
/// матрица чувствительности "источник в грунте - детектор на 1 м"
/// и реконструкция активности средствами пакета soilactivity.
///
/// Сборка и запуск: см. README.md и macros/*.mac
#include "B1ActionInitialization.hh"
#include "B1DetectorConstruction.hh"

#include "FTFP_BERT.hh"
#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4UIExecutive.hh"
#include "G4VisExecutive.hh"

int main(int argc, char** argv)
{
  auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);

  runManager->SetUserInitialization(new B1DetectorConstruction());
  runManager->SetUserInitialization(new FTFP_BERT);
  runManager->SetUserInitialization(new B1ActionInitialization());

  auto* visManager = new G4VisExecutive;
  visManager->Initialize();

  auto* UImanager = G4UImanager::GetUIpointer();

  if (argc > 1) {
    // пакетный режим: b1soil macros/sensitivity.mac
    G4String command = "/control/execute ";
    UImanager->ApplyCommand(command + argv[1]);
  }
  else {
    // интерактивный режим с визуализацией
    auto* ui = new G4UIExecutive(argc, argv);
    UImanager->ApplyCommand("/control/execute macros/vis.mac");
    ui->SessionStart();
    delete ui;
  }

  delete visManager;
  delete runManager;
  return 0;
}
