# Security policy

Task Agent reads untrusted content (web pages, emails, files) and can act on the outside
world, so we treat security reports as a priority.

## Reporting a vulnerability

Please **do not open a public issue**. Report privately through
[GitHub security advisories](https://github.com/SpandanNagale/Task-Agent/security/advisories/new)
with steps to reproduce and the impact you expect. We aim to acknowledge reports within 72
hours.

Especially in scope:

- Prompt injection that leads to an unapproved side effect (sending, posting, paying, deleting)
- Bypasses of the approval gate or the tool policy file
- Sandbox escapes, or access to files or network beyond the sandbox's limits
- Secrets leaking into prompts, logs, traces or eval reports
- Cross-task or cross-user data exposure

## Supported versions

The project is pre-1.0. Fixes land on `main`.
