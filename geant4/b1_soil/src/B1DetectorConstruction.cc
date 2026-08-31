#include "B1DetectorConstruction.hh"

#include "G4Box.hh"
#include "G4Orb.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4SystemOfUnits.hh"
#include "G4RunManager.hh"
#include "G4Exception.hh"
#include "CLHEP/Units/PhysicalConstants.h"

#include <cmath>
#include <iomanip>
#include <sstream>

B1DetectorConstruction::B1DetectorConstruction() = default;

// --- сеттеры ----------------------------------------------------------------
void B1DetectorConstruction::SetGridNx(G4int v)   { fNx = std::max(1, v); GeometryChanged(); }
void B1DetectorConstruction::SetGridNy(G4int v)   { fNy = std::max(1, v); GeometryChanged(); }
void B1DetectorConstruction::SetCellSize(G4double v)  { fCellSize  = v; GeometryChanged(); }
void B1DetectorConstruction::SetSrcDepth(G4double v)  { fSrcDepth  = v; GeometryChanged(); }
void B1DetectorConstruction::SetSoilDepth(G4double v) { fSoilDepth = v; GeometryChanged(); }
void B1DetectorConstruction::SetDetHeight(G4double v) { fDetHeight = v; GeometryChanged(); }
void B1DetectorConstruction::SetDetRadius(G4double v) { fDetRadius = v; GeometryChanged(); }
void B1DetectorConstruction::SetSoilDensity(G4double v){ fSoilDensity = v; GeometryChanged(); }

void B1DetectorConstruction::SetDetMode(const G4String& mode)
{
  if (mode != "single" && mode != "all") {
    G4ExceptionDescription msg;
    msg << "Unknown det mode '" << mode << "' (allowed: single | all)";
    G4Exception("B1DetectorConstruction::SetDetMode", "B1SOIL01",
                FatalException, msg);
    return;
  }
  fDetMode = mode;
  GeometryChanged();
}

void B1DetectorConstruction::SetDetIndex(G4int v)
{
  const G4int n = fNx * fNy;
  if (v < 0 || v >= n) {
    G4ExceptionDescription msg;
    msg << "Detector index " << v << " out of range [0, " << n - 1 << "]";
    G4Exception("B1DetectorConstruction::SetDetIndex", "B1SOIL02",
                FatalException, msg);
    return;
  }
  fDetIndex = v;
  GeometryChanged();
}

void B1DetectorConstruction::GeometryChanged()
{
  auto* rm = G4RunManager::GetRunManager();
  if (rm != nullptr) rm->GeometryHasBeenModified();
}

// --- координаты сетки -------------------------------------------------------
G4double B1DetectorConstruction::CellX(G4int idx) const
{
  const G4int ix = idx % fNx;
  return (ix - 0.5 * (fNx - 1)) * fCellSize;
}

G4double B1DetectorConstruction::CellY(G4int idx) const
{
  const G4int iy = idx / fNx;
  return (iy - 0.5 * (fNy - 1)) * fCellSize;
}

G4ThreeVector B1DetectorConstruction::SourcePosition(G4int idx) const
{
  return { CellX(idx), CellY(idx), -fSrcDepth };
}

G4ThreeVector B1DetectorConstruction::DetectorPosition(G4int idx) const
{
  return { CellX(idx), CellY(idx), fDetHeight };
}

// --- построение геометрии ---------------------------------------------------
G4VPhysicalVolume* B1DetectorConstruction::Construct()
{
  auto* nist = G4NistManager::Instance();

  // Воздух (NIST) и фиксированный материал детектора
  G4Material* air = nist->FindOrBuildMaterial("G4_AIR");

  // Грунт: ICRU 53 sandy loam (тот же состав, что NIST_SOIL_COMPOSITION
  // в пакете soilactivity: H 2.1, C 2.0, O 56.0, Al 6.0, Si 29.0,
  // K 1.2, Ca 1.8, Fe 1.9 % по массе). Плотность задаётся командой.
  // Имя материала зависит от плотности, чтобы повторное построение
  // геометрии не конфликтовало с уже созданным материалом.
  std::ostringstream nameSS;
  nameSS << "Soil_" << std::fixed << std::setprecision(2)
         << (fSoilDensity / (g / cm3));
  G4String soilName = nameSS.str();
  G4Material* soil = G4Material::GetMaterial(soilName, /*quiet=*/false);
  if (soil == nullptr) {
    soil = new G4Material(soilName, fSoilDensity, 8);
    soil->AddElement(nist->FindOrBuildElement("H"),  0.021);
    soil->AddElement(nist->FindOrBuildElement("C"),  0.020);
    soil->AddElement(nist->FindOrBuildElement("O"),  0.560);
    soil->AddElement(nist->FindOrBuildElement("Al"), 0.060);
    soil->AddElement(nist->FindOrBuildElement("Si"), 0.290);
    soil->AddElement(nist->FindOrBuildElement("K"),  0.012);
    soil->AddElement(nist->FindOrBuildElement("Ca"), 0.018);
    soil->AddElement(nist->FindOrBuildElement("Fe"), 0.019);
  }

  // --- Мир (воздух): XY с запасом 5 м вокруг сетки, Z от -(soilDepth+1 м) до +(detHeight+2 м)
  const G4double worldHalfXY =
      0.5 * std::max(fNx, fNy) * fCellSize + 5.0 * m;
  const G4double zMin = -(fSoilDepth + 1.0 * m);
  const G4double zMax = fDetHeight + 2.0 * m;
  const G4double worldHalfZ = 0.5 * (zMax - zMin);
  const G4double worldZc    = 0.5 * (zMax + zMin);

  auto* worldSolid = new G4Box("World", worldHalfXY, worldHalfXY, worldHalfZ);
  auto* worldLV = new G4LogicalVolume(worldSolid, air, "WorldLV");
  auto* worldPV = new G4PVPlacement(nullptr, G4ThreeVector(0, 0, worldZc),
                                    worldLV, "World", nullptr, false, 0);

  // --- Слой грунта (z от -fSoilDepth до 0), вплотную к стенкам мира по XY
  auto* soilSolid = new G4Box("Soil", worldHalfXY, worldHalfXY,
                              0.5 * fSoilDepth);
  auto* soilLV = new G4LogicalVolume(soilSolid, soil, "SoilLV");
  new G4PVPlacement(nullptr, G4ThreeVector(0, 0, -0.5 * fSoilDepth),
                    soilLV, "Soil", worldLV, false, 0);

  // --- Детектор: воздушная сфера (R = fDetRadius) над центром ячейки
  auto* detSolid = new G4Orb("Det", fDetRadius);
  auto* detLV = new G4LogicalVolume(detSolid, air, "DetLV");

  fDetMass = (4.0 / 3.0) * CLHEP::pi * std::pow(fDetRadius / m, 3)
             * (air->GetDensity() / (kg / m3));  // кг

  if (fDetMode == "all") {
    for (G4int i = 0; i < fNx * fNy; ++i) {
      new G4PVPlacement(nullptr, DetectorPosition(i), detLV,
                        "DetPV", worldLV, false, i);
    }
  }
  else { // single: один детектор, copy number = индекс позиции
    new G4PVPlacement(nullptr, DetectorPosition(fDetIndex), detLV,
                      "DetPV", worldLV, false, fDetIndex);
  }

  return worldPV;
}
