# Pull Request Workflow

Use one focused branch and pull request per GitHub issue.

## Branch Naming

Use:

```text
codex/issue-<number>-<short-description>
```

Examples:

- `codex/issue-21-readme-quickstart`
- `codex/issue-25-pr-workflow`

## Flow

1. Start from the correct base branch.
2. Create one branch for one issue.
3. Keep the commit focused to that issue.
4. Run the relevant local tests.
5. Push the branch.
6. Open a pull request with `Closes #<issue-number>`.
7. Include validation logs in the PR body.

## Stacked Work

When several dependent issues are built in sequence, stack PRs by targeting the
previous issue branch. After earlier PRs merge, retarget later PRs to `main`
before merging them.

## Commit Scope

Do:

- Commit source, tests, docs, and config needed for the issue.
- Keep generated artifacts out of Git.
- Mention tests and docs in the PR checklist.

Do not:

- Mix unrelated issue work in the same PR.
- Commit `.env`, secrets, `chroma_db/`, logs, or local virtualenv files.
