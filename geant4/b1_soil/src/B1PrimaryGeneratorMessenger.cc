#include "B1PrimaryGeneratorMessenger.hh"
#include "B1PrimaryGeneratorAction.hh"
#include "B1RunAction.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIparameter.hh"
#include "G4SystemOfUnits.hh"

#include <sstream>

B1PrimaryGeneratorMessenger::B1PrimaryGeneratorMessenger(B1PrimaryGeneratorAction* pg)
  : fPG(pg)
{
  auto* dir = new G4UIdirectory("/b1soil/");
  dir->SetGuidance("Управление источником b1_soil");

  fSetLine = new G4UIcmdWithAString("/b1soil/setGammaLine", this);
  fSetLine->SetGuidance("Выбор нуклида: Cs137 (по умолчанию) | Co60 | Am241");
  fSetLine->SetParameterName("nuclide", false);
  fSetLine->SetCandidates("Cs137 Co60 Am241");

  fAddLine = new G4UIcommand("/b1soil/addLine", this);
  fAddLine->SetGuidance("Добавить гамма-линию: /b1soil/addLine E yield");
  auto* pE = new G4UIparameter("E", 'd', false);
  pE->SetDefaultValue("0.6617");
  fAddLine->SetParameter(pE);
  auto* pEu = new G4UIparameter("unit", 's', false);
  pEu->SetDefaultValue("MeV");
  fAddLine->SetParameter(pEu);
  auto* pY = new G4UIparameter("yield", 'd', false);
  pY->SetDefaultValue("1.0");
  fAddLine->SetParameter(pY);

  fClearLines = new G4UIcmdWithAString("/b1soil/clearLines", this);
  fClearLines->SetGuidance("Очистить список гамма-линий (аргумент игнорируется)");

  fSetSrcIndex = new G4UIcmdWithAnInteger("/b1soil/setSrcIndex", this);
  fSetSrcIndex->SetGuidance("Источник в ячейку сетки с индексом idx (row-major: idx = iy*nx + ix)");
  fSetSrcIndex->SetParameterName("idx", false);

  fSetSrcX = new G4UIcmdWithADoubleAndUnit("/b1soil/setSrcX", this);
  fSetSrcX->SetGuidance("Ручная координата источника X (вместе с Y и Z; вне сетки)");
  fSetSrcX->SetParameterName("x", false);
  fSetSrcX->SetDefaultUnit("m");
  fSetSrcY = new G4UIcmdWithADoubleAndUnit("/b1soil/setSrcY", this);
  fSetSrcY->SetGuidance("Ручная координата источника Y");
  fSetSrcY->SetParameterName("y", false);
  fSetSrcY->SetDefaultUnit("m");
  fSetSrcZ = new G4UIcmdWithADoubleAndUnit("/b1soil/setSrcZ", this);
  fSetSrcZ->SetGuidance("Ручная координата источника Z (отрицательная - в грунте)");
  fSetSrcZ->SetParameterName("z", false);
  fSetSrcZ->SetDefaultUnit("m");

  fUseModel = new G4UIcmdWithABool("/b1soil/useModelSource", this);
  fUseModel->SetGuidance("true: ячейка распада выбирается случайно пропорционально активности "
                         "(модельный источник); метка прогона автоматически MODEL");
  fUseModel->SetParameterName("flag", false);
  fUseModel->SetDefaultValue(true);

  fAddCell = new G4UIcommand("/b1soil/addModelCell", this);
  fAddCell->SetGuidance("Добавить ячейку модельного источника: x y z activity");
  fAddCell->SetGuidance("Например: /b1soil/addModelCell 0 0 -0.1 m 2e8 Bq");
  const char* names[4] = {"x", "y", "z", "activity"};
  const char* units[4] = {"m", "m", "m", "Bq"};
  for (G4int i = 0; i < 4; ++i) {
    auto* p = new G4UIparameter(names[i], 'd', false);
    p->SetDefaultValue("0.0");
    fAddCell->SetParameter(p);
    auto* pu = new G4UIparameter((G4String(names[i]) + "_unit").c_str(), 's', true);
    pu->SetDefaultValue(units[i]);
    fAddCell->SetParameter(pu);
  }

  fClearCells = new G4UIcmdWithAString("/b1soil/clearModelCells", this);
  fClearCells->SetGuidance("Очистить ячейки модельного источника (аргумент игнорируется)");
}

B1PrimaryGeneratorMessenger::~B1PrimaryGeneratorMessenger()
{
  delete fSetLine; delete fAddLine; delete fClearLines;
  delete fSetSrcIndex; delete fSetSrcX; delete fSetSrcY; delete fSetSrcZ;
  delete fUseModel; delete fAddCell; delete fClearCells;
}

void B1PrimaryGeneratorMessenger::SetNewValue(G4UIcommand* cmd, G4String value)
{
  if (cmd == fSetLine)        { fPG->SetNuclide(value); }
  else if (cmd == fClearLines){ fPG->ClearLines(); }
  else if (cmd == fAddLine) {
    G4double E, y;
    G4String unit;
    std::istringstream is(value);
    is >> E >> unit >> y;
    if (unit == "keV") E *= keV;
    else               E *= MeV;
    fPG->AddLine(E, y);
  }
  else if (cmd == fSetSrcIndex) {
    fPG->SetSrcIndex(fSetSrcIndex->GetNewIntValue(value));
    B1RunMeta::srcIndex = fPG->GetSrcIndex();
    B1RunMeta::useModel = false;
    if (B1RunMeta::runType == "MODEL") B1RunMeta::runType = "SENSITIVITY";
  }
  else if (cmd == fSetSrcX) {
    fPG->SetSrcX(fSetSrcX->GetNewDoubleValue(value));
    B1RunMeta::srcX = fPG->GetSrcIndex() == -2 ? fSetSrcX->GetNewDoubleValue(value) : 0.0;
    B1RunMeta::srcIndex = -2;
    B1RunMeta::useModel = false;
  }
  else if (cmd == fSetSrcY) {
    fPG->SetSrcY(fSetSrcY->GetNewDoubleValue(value));
    B1RunMeta::srcY = fSetSrcY->GetNewDoubleValue(value);
    B1RunMeta::srcIndex = -2;
  }
  else if (cmd == fSetSrcZ) {
    fPG->SetSrcZ(fSetSrcZ->GetNewDoubleValue(value));
    B1RunMeta::srcZ = fSetSrcZ->GetNewDoubleValue(value);
    B1RunMeta::srcIndex = -2;
  }
  else if (cmd == fUseModel) {
    fPG->SetUseModel(fUseModel->GetNewBoolValue(value));
    B1RunMeta::useModel = fPG->IsModel();
    B1RunMeta::runType  = fPG->IsModel() ? "MODEL" : "SENSITIVITY";
  }
  else if (cmd == fAddCell) {
    G4double x, y, z, a;
    G4String xu, yu, zu, au;
    std::istringstream is(value);
    is >> x >> xu >> y >> yu >> z >> zu >> a >> au;
    x *= (xu == "cm") ? cm : m;
    y *= (yu == "cm") ? cm : m;
    z *= (zu == "cm") ? cm : m;
    if      (au == "kBq") a *= kBq;
    else if (au == "MBq") a *= MBq;
    else if (au == "GBq") a *= GBq;   // иначе Бк
    fPG->AddModelCell(G4ThreeVector(x, y, z), a);
  }
  else if (cmd == fClearCells) { fPG->ClearModelCells(); }
}
