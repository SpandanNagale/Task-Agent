## What and why

## How it was tested

- [ ] `ruff`, `mypy` and `pytest` pass locally
- [ ] Eval suite still runs (`task-agent eval --models config/models.mock.yaml`)

## Eval impact

<!-- If agent behaviour changed: success rate, cost per task, and steps before/after. Otherwise "none". -->

## Checklist

- [ ] New tools declare an honest side-effect level; write actions go through approval
- [ ] New capabilities come with eval tasks
- [ ] No secrets in code, config, fixtures or logs
- [ ] Docs updated (README / docs / evals README) where behaviour changed
