@AGENTS.md

## Claude Code 전용

- 작업은 `.claude/worktrees/` 아래 worktree 에서 한다. `.claude/` 폴더(worktree·로컬 설정)는 커밋하지 않는다.
- 커밋 메시지 끝에는 Claude Code 가 붙이는 `Co-Authored-By: Claude … <noreply@anthropic.com>` 줄을 그대로 둔다.
- 변경 검토는 내장 `/simplify`(단순화)·`/code-review` 를 쓰거나, [docs/AI_검토_프롬프트.md](docs/AI_검토_프롬프트.md) 기준으로 검토한다.
