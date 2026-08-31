#ifndef B1PrimaryGeneratorAction_h
#define B1PrimaryGeneratorAction_h 1

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "globals.hh"

#include <vector>

class G4ParticleGun;

/// Генератор первичных частиц b1_soil (развитие basic/B1).
///
/// Одно MC-событие == один распад источника:
///   - для каждой гамма-линии (E_i, yield_i) эмитируется floor(yield_i)
///     фотонов плюс один с вероятностью frac(yield_i)
///     (Cs-137: 1 фотон 661.7 кэВ с вероятностью 0.851, остальные события
///     пустые - это корректная нормировка "на распад");
///   - координата распада: ячейка сетки (setSrcIndex), ручные координаты
///     (setSrcX/Y/Z) или модельное распределение (addModelCell +
///     useModelSource: ячейка выбирается случайно пропорционально активности);
///   - направление фотонов изотропное (4*пи).
class B1PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    struct Line { G4double energy; G4double yield; };  // МэВ, фотон/распад

    B1PrimaryGeneratorAction();
    ~B1PrimaryGeneratorAction() override;

    void GeneratePrimaries(G4Event* event) override;

    // --- нуклидные линии
    void SetNuclide(const G4String& name);      // "Cs137" | "Co60" | "Am241"
    void AddLine(G4double E, G4double yield)    { fLines.push_back({E, yield}); }
    void ClearLines()                           { fLines.clear(); }
    const std::vector<Line>& GetLines() const   { return fLines; }

    // --- положение источника
    void SetSrcIndex(G4int idx) { fSrcIndex = idx; fUseModel = false; }
    void SetSrcX(G4double v)    { fSrcX = v; fSrcIndex = -2; fUseModel = false; }
    void SetSrcY(G4double v)    { fSrcY = v; fSrcIndex = -2; fUseModel = false; }
    void SetSrcZ(G4double v)    { fSrcZ = v; fSrcIndex = -2; fUseModel = false; }
    void SetUseModel(G4bool v)  { fUseModel = v; if (v) fSrcIndex = -1; }
    G4bool IsModel() const      { return fUseModel; }
    G4int  GetSrcIndex() const  { return fSrcIndex; }

    // --- модельный источник (ячейки с активностью, Бк)
    struct ModelCell { G4ThreeVector pos; G4double activity; };
    void ClearModelCells()                    { fModelCells.clear(); fModelCum.clear(); }
    void AddModelCell(G4ThreeVector p, G4double a) { fModelCells.push_back({p, a}); }
    const std::vector<ModelCell>& GetModelCells() const { return fModelCells; }

  private:
    G4ParticleGun* fGun = nullptr;
    std::vector<Line>       fLines;
    std::vector<ModelCell>  fModelCells;
    std::vector<G4double>   fModelCum;   // кумулятивные веса ячеек (кэш)
    G4int         fSrcIndex = 0;         // >=0 сетка; -1 модель; -2 ручные координаты
    G4bool        fUseModel = false;
    G4double      fSrcX = 0.0, fSrcY = 0.0, fSrcZ = -10.0 * cm;
};

#endif
