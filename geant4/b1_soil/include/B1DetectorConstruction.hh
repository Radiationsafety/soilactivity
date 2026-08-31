#ifndef B1DetectorConstruction_h
#define B1DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "globals.hh"

class G4VPhysicalVolume;
class G4LogicalVolume;

/// Геометрия примера b1_soil (развите basic/B1):
///   - плоский слой грунта (z <= 0) с материалом Soil (ICRU 53 sandy loam);
///   - сетка nx x ny ячеек с шагом cellSize;
///   - точечный источник внутри грунта на глубине fSrcDepth под центром ячейки;
///   - сферический детектор (воздух, R = fDetRadius) на высоте fDetHeight
///     над центром той же ячейки.
///
/// Режимы размещения детектора (setDetMode):
///   "single" - ровно один детектор над ячейкой fDetIndex
///              (буквальное "поочерёдное" размещение по сетке);
///   "all"    - все nx*ny детекторов одновременно (физически эквивалентно
///              поочерёдному размещению: воздух на 662 кэВ почти прозрачен,
///              искажение < 0.01 %), один прогон даёт сразу строку матрицы A.
///
/// Ускоряющая структура не нужна: скоринг выполняет B1SteppingAction
/// по имени логического объёма "DetLV" и copy-number физического объёма.
class B1DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    B1DetectorConstruction();
    ~B1DetectorConstruction() override = default;

    G4VPhysicalVolume* Construct() override;

    // ---- параметры сетки (доступны также из B1RunAction/B1PrimaryGeneratorAction)
    G4int    GetGridNx() const     { return fNx; }
    G4int    GetGridNy() const     { return fNy; }
    G4double GetCellSize() const   { return fCellSize; }
    G4double GetSrcDepth() const   { return fSrcDepth; }
    G4double GetSoilDepth() const  { return fSoilDepth; }
    G4double GetDetHeight() const  { return fDetHeight; }
    G4double GetDetRadius() const  { return fDetRadius; }
    G4double GetSoilDensity() const { return fSoilDensity; }
    G4int    GetDetIndex() const   { return fDetIndex; }
    const G4String& GetDetMode() const { return fDetMode; }
    G4int    GetNDetectors() const { return fNx * fNy; }
    G4double GetDetectorMass() const { return fDetMass; }

    // индексы ячеек row-major: idx = iy*nx + ix
    G4double      CellX(G4int idx) const;
    G4double      CellY(G4int idx) const;
    G4ThreeVector SourcePosition(G4int idx) const;    // в грунте (z = -fSrcDepth)
    G4ThreeVector DetectorPosition(G4int idx) const;  // на высоте fDetHeight

    // ---- сеттеры (вызываются из B1DetectorMessenger)
    void SetGridNx(G4int v);
    void SetGridNy(G4int v);
    void SetCellSize(G4double v);
    void SetSrcDepth(G4double v);
    void SetSoilDepth(G4double v);
    void SetDetHeight(G4double v);
    void SetDetRadius(G4double v);
    void SetSoilDensity(G4double v);
    void SetDetMode(const G4String& mode); // "single" | "all"
    void SetDetIndex(G4int v);

  private:
    G4int    fNx = 5;
    G4int    fNy = 5;
    G4double fCellSize   = 2.0 * m;
    G4double fSrcDepth   = 10.0 * cm;
    G4double fSoilDepth  = 2.0 * m;     // толщина слоя грунта
    G4double fDetHeight  = 1.0 * m;
    G4double fDetRadius  = 15.0 * cm;   // сфера ICRU (диаметр 30 см)
    G4double fSoilDensity = 1.6 * g / cm3;
    G4String fDetMode    = "all";
    G4int    fDetIndex   = 0;
    G4double fDetMass    = 0.0;         // масса детектора (кг), заполняется в Construct()

    void GeometryChanged();
};

#endif
