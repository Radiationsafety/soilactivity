#include "B1ActionInitialization.hh"
#include "B1PrimaryGeneratorAction.hh"
#include "B1RunAction.hh"
#include "B1SteppingAction.hh"

void B1ActionInitialization::Build() const
{
  SetUserAction(new B1PrimaryGeneratorAction());
  SetUserAction(new B1RunAction());
  SetUserAction(new B1SteppingAction());
}

void B1ActionInitialization::BuildForMaster() const
{
  // на master нужен только RunAction (слияние накопителей и запись CSV)
  SetUserAction(new B1RunAction());
}
