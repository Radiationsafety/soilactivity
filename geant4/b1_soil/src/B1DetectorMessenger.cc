#include "B1DetectorMessenger.hh"
#include "B1DetectorConstruction.hh"
#include "B1RunAction.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"

B1DetectorMessenger::B1DetectorMessenger(B1DetectorConstruction* dc)
  : fDC(dc)
{
  auto* dir = new G4UIdirectory("/b1soil/");
  dir->SetGuidance("Управление примером b1_soil (грунт - сетка источников - детектор на 1 м)");

  fSetNx = new G4UIcmdWithAnInteger("/b1soil/setGridNx", this);
  fSetNx->SetGuidance("Число ячеек сетки по X (по умолчанию 5)");
  fSetNx->SetParameterName("nx", false);
  fSetNx->SetRange("nx > 0");

  fSetNy = new G4UIcmdWithAnInteger("/b1soil/setGridNy", this);
  fSetNy->SetGuidance("Число ячеек сетки по Y (по умолчанию 5)");
  fSetNy->SetParameterName("ny", false);
  fSetNy->SetRange("ny > 0");

  fSetCell = new G4UIcmdWithADoubleAndUnit("/b1soil/setCellSize", this);
  fSetCell->SetGuidance("Шаг сетки (по умолчанию 2 м)");
  fSetCell->SetParameterName("cell", false);
  fSetCell->SetDefaultUnit("m");

  fSetSrcDepth = new G4UIcmdWithADoubleAndUnit("/b1soil/setSrcDepth", this);
  fSetSrcDepth->SetGuidance("Глубина источника под поверхностью грунта (по умолчанию 10 см)");
  fSetSrcDepth->SetParameterName("depth", false);
  fSetSrcDepth->SetDefaultUnit("cm");

  fSetSoilDepth = new G4UIcmdWithADoubleAndUnit("/b1soil/setSoilDepth", this);
  fSetSoilDepth->SetGuidance("Толщина слоя грунта (по умолчанию 2 м)");
  fSetSoilDepth->SetParameterName("soil", false);
  fSetSoilDepth->SetDefaultUnit("m");

  fSetDetHeight = new G4UIcmdWithADoubleAndUnit("/b1soil/setDetHeight", this);
  fSetDetHeight->SetGuidance("Высота центра детектора над поверхностью (по умолчанию 1 м)");
  fSetDetHeight->SetParameterName("h", false);
  fSetDetHeight->SetDefaultUnit("m");

  fSetDetRadius = new G4UIcmdWithADoubleAndUnit("/b1soil/setDetRadius", this);
  fSetDetRadius->SetGuidance("Радиус сферического детектора (по умолчанию 15 см - сфера ICRU)");
  fSetDetRadius->SetParameterName("r", false);
  fSetDetRadius->SetDefaultUnit("cm");

  fSetSoilDensity = new G4UIcmdWithADoubleAndUnit("/b1soil/setSoilDensity", this);
  fSetSoilDensity->SetGuidance("Плотность грунта (по умолчанию 1.6 г/см3)");
  fSetSoilDensity->SetParameterName("rho", false);
  fSetSoilDensity->SetDefaultUnit("g/cm3");
  fSetSoilDensity->SetUnitCandidates("g/cm3 kg/m3");

  fSetDetMode = new G4UIcmdWithAString("/b1soil/setDetMode", this);
  fSetDetMode->SetGuidance("all      - все nx*ny детекторов одновременно (быстро, физически эквивалентно)");
  fSetDetMode->SetGuidance("single   - один детектор над ячейкой setDetIndex (буквально поочерёдное размещение)");
  fSetDetMode->SetParameterName("mode", false);
  fSetDetMode->SetCandidates("all single");

  fSetDetIndex = new G4UIcmdWithAnInteger("/b1soil/setDetIndex", this);
  fSetDetIndex->SetGuidance("Индекс ячейки детектора для режима single (row-major: idx = iy*nx + ix)");
  fSetDetIndex->SetParameterName("idx", false);

  fSetRunType = new G4UIcmdWithAString("/b1soil/setRunType", this);
  fSetRunType->SetGuidance("Метка типа прогона в CSV: SENSITIVITY (матрица A) | MODEL (модельный источник)");
  fSetRunType->SetParameterName("type", false);

  fSetOutputFile = new G4UIcmdWithAString("/b1soil/setOutputFile", this);
  fSetOutputFile->SetGuidance("Имя CSV-файла результатов (дописывается в конец; по умолчанию b1soil_runs.csv)");
  fSetOutputFile->SetParameterName("file", false);
}

B1DetectorMessenger::~B1DetectorMessenger()
{
  delete fSetNx; delete fSetNy; delete fSetCell; delete fSetSrcDepth;
  delete fSetSoilDepth; delete fSetDetHeight; delete fSetDetRadius;
  delete fSetSoilDensity; delete fSetDetMode; delete fSetDetIndex;
  delete fSetRunType; delete fSetOutputFile;
}

void B1DetectorMessenger::SetNewValue(G4UIcommand* cmd, G4String value)
{
  if (cmd == fSetNx)            { fDC->SetGridNx(fSetNx->GetNewIntValue(value)); }
  else if (cmd == fSetNy)       { fDC->SetGridNy(fSetNy->GetNewIntValue(value)); }
  else if (cmd == fSetCell)     { fDC->SetCellSize(fSetCell->GetNewDoubleValue(value)); }
  else if (cmd == fSetSrcDepth) { fDC->SetSrcDepth(fSetSrcDepth->GetNewDoubleValue(value)); }
  else if (cmd == fSetSoilDepth){ fDC->SetSoilDepth(fSetSoilDepth->GetNewDoubleValue(value)); }
  else if (cmd == fSetDetHeight){ fDC->SetDetHeight(fSetDetHeight->GetNewDoubleValue(value)); }
  else if (cmd == fSetDetRadius){ fDC->SetDetRadius(fSetDetRadius->GetNewDoubleValue(value)); }
  else if (cmd == fSetSoilDensity) { fDC->SetSoilDensity(fSetSoilDensity->GetNewDoubleValue(value)); }
  else if (cmd == fSetDetMode)  { fDC->SetDetMode(value); }
  else if (cmd == fSetDetIndex) { fDC->SetDetIndex(fSetDetIndex->GetNewIntValue(value)); }
  else if (cmd == fSetRunType)  { B1RunMeta::runType = value; }
  else if (cmd == fSetOutputFile) { B1RunMeta::outputFile = value; }
}
