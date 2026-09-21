# Contributing to Tensor Engine NMT

Thank you for contributing to `tensor-engine-nmt`!

To maintain a clean, readable, and reproducible history across this from-scratch deep learning engine, all commits must follow our official guidelines.

---

## Commit Guidelines Summary

We follow the **Conventional Commits v1.0.0** specification:

```text
<type>(<scope>): <short imperative summary>
```

### Allowed Types
- `feat`: A new feature or capability
- `fix`: A bug fix or mathematical correction
- `docs`: Documentation updates only
- `test`: Adding or modifying tests
- `perf`: Performance or memory optimizations
- `refactor`: Restructuring code without changing functionality
- `chore`: Tooling, packaging, or maintenance updates

### Allowed Scopes
- Neural Engine: `(model)`, `(encoder)`, `(decoder)`, `(attention)`, `(loss)`, `(activations)`, `(backend)`
- Data Pipeline: `(bpe)`, `(dataset)`, `(config)`
- Execution: `(train)`, `(optimizer)`, `(inference)`, `(evaluate)`
- Utilities: `(tools)`, `(bench)`, `(project)`

### Full Documentation
For the complete guide, forbidden file types, and pre-commit verification checklists, see:
👉 **[docs/commit_rules.md](docs/commit_rules.md)**
