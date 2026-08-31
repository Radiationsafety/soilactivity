#include "B1RunAction.hh"
#include "B1Run.hh"
#include "B1DetectorConstruction.hh"

#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4Exception.hh"
#include "G4ThreeVector.hh"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

// --- метаданные прогона (значения по умолчанию) -----------------------------
G4String B1RunMeta::runType    = "SENSITIVITY";
G4String B1RunMeta::outputFile = "b1soil_runs.csv";
G4int    B1RunMeta::srcIndex   = 0;
G4bool   B1RunMeta::useModel   = false;
G4double B1RunMeta::srcX = 0.0, B1RunMeta::srcY = 0.0, B1RunMeta::srcZ = -10.0 * cm;

G4Run* B1RunAction::GenerateRun()
{
  return new B1Run();
}

// ----------------------------------------------------------------------------
void B1RunAction::EndOfRunAction(const G4Run* run)
{
  if (!IsMaster()) return;  // CSV пишет только master

  const B1Run* b1run = static_cast<const B1Run*>(run);
  const G4int  nDecays = run->GetNumberOfEvent();

  auto* dc = static_cast<B1DetectorConstruction*>(
      G4RunManager::GetRunManager()->GetUserDetectorConstruction());

  const G4int nDet = dc->GetDetMode() == "all" ? dc->GetNDetectors() : 1;

  // --- шапка (пишется один раз, если файла ещё нет) --------------------------
  const G4String& fname = B1RunMeta::outputFile;
  std::ifstream probe(fname);
  const bool needHeader =
      !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
  probe.close();

  std::ofstream out(fname, std::ios::app);
  if (!out.good()) {
    G4ExceptionDescription msg;
    msg << "Cannot open output CSV '" << fname << "'";
    G4Exception("B1RunAction::EndOfRunAction", "B1SOIL03",
                FatalException, msg);
    return;
  }
  out << std::setprecision(10);

  if (needHeader) {
    out << "# b1soil_version=1.0"
        << " nx="          << dc->GetGridNx()
        << " ny="          << dc->GetGridNy()
        << " cellSize_m="  << dc->GetCellSize() / m
        << " srcDepth_m="  << dc->GetSrcDepth() / m
        << " soilDepth_m=" << dc->GetSoilDepth() / m
        << " detHeight_m=" << dc->GetDetHeight() / m
        << " detRadius_m=" << dc->GetDetRadius() / m
        << " soilDensity_g_cm3=" << dc->GetSoilDensity() / (g / cm3)
        << " nBins="       << B1Run::kNBins
        << " emin_MeV="    << B1Run::kEMin
        << " emax_MeV="    << B1Run::kEMax
        << "\n";
    out << "# спектр: логарифмические бины вошедших фотонов, сетка в python/b1soil_io.py\n";
    out << "#run_type,src_index,src_x_m,src_y_m,src_z_m,det_mode,det_index,"
           "det_x_m,det_y_m,det_z_m,n_decays,n_in,edep_sum_MeV,edep_rms_MeV";
    for (G4int b = 0; b < B1Run::kNBins; ++b) out << ",sp_" << std::setw(3) << std::setfill('0') << b;
    out << std::setfill(' ') << "\n";
  }

  // --- координаты источника --------------------------------------------------
  G4double sx = -1.0, sy = -1.0, sz = -1.0;
  if (!B1RunMeta::useModel) {
    if (B1RunMeta::srcIndex >= 0) {
      const G4ThreeVector p = dc->SourcePosition(B1RunMeta::srcIndex);
      sx = p.x() / m; sy = p.y() / m; sz = p.z() / m;
    } else { // ручные координаты
      sx = B1RunMeta::srcX / m; sy = B1RunMeta::srcY / m; sz = B1RunMeta::srcZ / m;
    }
  }

  // --- одна строка на активный детектор --------------------------------------
  for (G4int i = 0; i < nDet; ++i) {
    const G4int detIdx = (dc->GetDetMode() == "all") ? i : dc->GetDetIndex();
    const G4ThreeVector dpos = dc->DetectorPosition(detIdx);

    const G4double edepMeV  = b1run->GetEdep(detIdx);    // суммарное энерговыделение, МэВ (сырые единицы)
    const G4double edep2MeV = b1run->GetEdep2(detIdx);   // сумма квадратов, МэВ^2
    const G4double rms = (edepMeV > 0.0 && nDecays > 0)
        ? std::sqrt(std::max(edep2MeV / nDecays - std::pow(edepMeV / nDecays, 2), 0.0))
        : 0.0;

    out << B1RunMeta::runType << ','
        << (B1RunMeta::useModel ? -1 : B1RunMeta::srcIndex) << ','
        << sx << ',' << sy << ',' << sz << ','
        << dc->GetDetMode() << ',' << detIdx << ','
        << dpos.x() / m << ',' << dpos.y() / m << ',' << dpos.z() / m << ','
        << nDecays << ',' << b1run->GetNin(detIdx) << ','
        << std::scientific << edepMeV << ',' << rms << std::fixed;
    for (G4int b = 0; b < B1Run::kNBins; ++b)
      out << ',' << std::scientific << b1run->GetSpectrum(detIdx)[b];
    out << std::fixed << "\n";
  }
  out.close();

  // --- сводка в лог -----------------------------------------------------------
  G4double nTot = 0.0;
  for (G4int i = 0; i < nDet; ++i) nTot += (G4double)b1run->GetNin(i);
  G4cout << "\n[B1RunAction] run '" << B1RunMeta::runType << "': "
         << nDecays << " decays, photons entered detectors: "
         << nTot << "  (rows appended to " << fname << ")" << G4endl;
}
