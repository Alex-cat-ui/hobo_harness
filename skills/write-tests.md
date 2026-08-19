Follow these steps in order.

1. `TOOL read` the file you are testing and the existing test file.
2. Copy the style of the tests that are already there: same imports, same
   class or function shape, same naming.
3. Write the test file back whole with `TOOL write`. Add your tests, keep
   every test that was already there.
4. Check the import line at the top names every function your new tests call.
   This is the mistake that happens most often.
5. `TOOL run` the tests.
6. If a test fails, decide once: is the test wrong, or is the code wrong? Say
   which in your document. Fix only the test if the test is wrong.
7. Stop when the tests pass. Then write your document.

Each test checks one thing and says so in its name.

Never do these:
- Never test a library or the language itself.
- Never write a test that passes whether the code is right or wrong.
- Never depend on the clock, the network, or the order tests run in.
