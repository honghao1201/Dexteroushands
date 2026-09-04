# Dexteroushands

项目目标和技术栈尚待确定。目前包含 Codex 协作与项目管理基础文件。

## 在 VS Code 中开始
1. 使用“文件 → 打开文件夹”打开 `D:\Projects\Dexteroushands`。
2. 打开 Codex 侧边栏并按提示登录；也可在命令面板中执行 `Codex: Open Codex Sidebar`。
3. 在 Codex 中发送下面的启动指令。

```text
请先阅读 AGENTS.md、PROJECT.md 和 TASKS.md，概述当前项目状态和适用规则。
检查这些文件是否存在；如果能正确读取，记录 T002 的验证结果。
接着帮助我明确项目目标、首版功能和验收标准，再更新项目说明和任务清单。
```

## 日常协作
- 查看进度：“请根据 TASKS.md 汇报已完成、进行中、阻塞及下一步。”
- 开始任务：“请执行 Txxx，完成后验证结果并更新任务清单。”
- 新增需求：“请把以下需求拆成有验收标准的任务，并更新里程碑：……”
- 结束工作：“请同步项目进展、验证结果和下一次继续的位置。”

## 文件导航
- [项目说明与里程碑](PROJECT.md)
- [任务清单](TASKS.md)
- [Codex 协作规则](AGENTS.md)

## 运行与测试
尚无业务代码。技术栈确定后补充实际环境、依赖安装、启动和测试命令。

## 版本管理
当前未初始化 Git 仓库。Git 可用后，在项目目录执行 `git init`，再通过 `git status` 检查状态。
是否提交代码和连接远程仓库由用户决定。

## 参考
- [Codex IDE 官方说明](https://learn.chatgpt.com/docs/codex/ide)
- [AGENTS.md 官方说明](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
