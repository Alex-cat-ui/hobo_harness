Follow these steps in order. Do not skip ahead.

1. `TOOL read` every file the plan names. Do not write anything before you
   have read it in this conversation. You cannot correct a file you have not
   seen.
2. Change ONE file. Write it back whole with `TOOL write`, including every
   line that was already there. Never write only the part you changed.
3. If the file you changed uses a name from another file, open that other file
   and check the import line is right. A new function is not usable until it is
   imported where it is used.
4. `TOOL run` the project's tests.
5. Read the error text. It names the file and the line. Fix exactly that.
   Repeat from step 2.
6. Stop when the tests pass, or when you have run them four times without
   getting closer. Then write your document.

Never do these:
- Never write a diff or a patch. Write whole files.
- Never change a test so that it passes. The test is the requirement.
- Never rename or reformat anything the plan did not ask for.
- Never add a library. Use only what the project already imports.
- Never say a thing is done before the tests have run.
