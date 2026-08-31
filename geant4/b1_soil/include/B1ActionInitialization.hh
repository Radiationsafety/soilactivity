#ifndef B1ActionInitialization_h
#define B1ActionInitialization_h 1

#include "G4VUserActionInitialization.hh"

class B1ActionInitialization : public G4VUserActionInitialization
{
  public:
    B1ActionInitialization() = default;
    ~B1ActionInitialization() override = default;

    void Build() const override;          // worker / sequential
    void BuildForMaster() const override; // master (MT): только RunAction
};

#endif
