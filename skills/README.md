# Skills

Procedures a role follows, named by `skill` in the role definition.

Written for a 14B model, which means they obey rules a human-facing procedure
does not have to:

- **Steps, not principles.** A weak model cannot derive an action from a
  principle. Every line is something to do.
- **Short.** Past roughly forty lines the model starts answering the beginning
  and forgetting the end.
- **One literal output shape.** Shown, not described.
- **Explicit stop conditions.** Without them the model keeps going, or stops
  after one step believing it is finished.
- **Say what not to do, by name.** "Do not refactor" works; "keep the change
  minimal" does not.
