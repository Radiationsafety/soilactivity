#ifndef B1SteppingAction_h
#define B1SteppingAction_h 1

#include "G4UserSteppingAction.hh"

/// Скоринг отклика детекторов (аналог подхода basic/B1, но для набора сфер):
///  1) флюенс: каждый ФОТОН, пересёкший границу внутрь сферы DetLV,
///     учитывается в спектре вошедших фотонов (copy number = индекс детектора).
///     Отклик H*(10) на распад вычисляется из спектра в python-обработке
///     через ICRP 74: H*(10)/распад = sum_b N_b * h*(10)/Phi(E_b) / (pi*R^2);
///  2) керма: энерговыделение внутри сфер (cross-check, доза/распад).
///
/// Проверка объёма по ИМЕНИ ("DetLV"), а не по кэшированному указателю:
/// геометрия перестраивается при смене setDetIndex/setDetMode, и указатель
/// на логический объём мог бы устареть.
class B1SteppingAction : public G4UserSteppingAction
{
  public:
    B1SteppingAction() = default;
    ~B1SteppingAction() override = default;

    void UserSteppingAction(const G4Step* step) override;
};

#endif
