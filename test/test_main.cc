// Shared entry point for the test binaries: initializes Abseil logging before
// gtest runs so CHECK/LOG output is formatted normally and Abseil's one-time
// "InitializeLog() not called" warning doesn't appear in test output.

#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

int main(int argc, char** argv) {
  absl::InitializeLog();
  // Tests are run interactively; show everything on stderr.
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
